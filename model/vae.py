import torch
from torch import nn
from dataclasses import dataclass


class DownSamplingResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(DownSamplingResidualBlock, self).__init__()

        down_sample_stride = 2
        inplace_stride = 1

        self.activation = nn.ELU()
        self.skip_conv = nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   stride=down_sample_stride,
                                   padding=kernel_size // 2)

        self.latent_conv1 = nn.Conv2d(in_channels,
                                      out_channels // 2,
                                      kernel_size=kernel_size,
                                      padding=kernel_size//2,
                                      stride=down_sample_stride)
        self.latent_conv2 = nn.Conv2d(out_channels // 2,
                                      out_channels,
                                      kernel_size=kernel_size,
                                      padding=kernel_size//2,
                                      stride=inplace_stride)

        self.latent_bn = nn.BatchNorm2d(out_channels // 2)
        self.output_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x_ = self.skip_conv(x)

        y = self.latent_bn(self.latent_conv1(x))
        y = self.activation(y)
        y = self.latent_conv2(y)

        return self.activation(self.output_bn(x_ + y))


class UpSamplingResidualBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(UpSamplingResidualBlock, self).__init__()

        self.activation = nn.ELU()
        self.upsample = nn.Upsample(scale_factor=2,
                                    mode='nearest')

        self.skip_conv = nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   stride=1,
                                   padding=kernel_size // 2)

        self.latent_conv1 = nn.Conv2d(in_channels,
                                      in_channels // 2,
                                      kernel_size=kernel_size,
                                      padding=kernel_size // 2,
                                      stride=1)
        self.latent_conv2 = nn.Conv2d(in_channels // 2,
                                      out_channels,
                                      kernel_size=kernel_size,
                                      padding=kernel_size // 2,
                                      stride=1)

        self.latent_bn = nn.BatchNorm2d(in_channels // 2)
        self.output_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x_ = self.upsample(x)
        skip = self.skip_conv(x_)

        y = self.activation(self.latent_bn(self.latent_conv1(x_)))
        y = self.latent_conv2(y)

        return self.activation(self.output_bn(skip + y))


class Encoder(nn.Module):

    def __init__(self, input_size,
                 base_num_features,
                 input_channels,
                 kernel_size,
                 final_conv_image_size,
                 latent_size):

        super(Encoder, self).__init__()

        assert input_size[0] == input_size[1]
        size = input_size[0]
        num_channels = input_channels
        d_blocks = []

        while size > final_conv_image_size:

            if size % 2 == 0:

                out_channels = num_channels * 2 if num_channels > input_channels else base_num_features
                d_blocks.append(
                    DownSamplingResidualBlock(in_channels=num_channels,
                                              out_channels=out_channels,
                                              kernel_size=kernel_size)
                )

                size //= 2
                num_channels = out_channels
            else:

                padded_size = size + 2
                desired_size = int(2**torch.floor(torch.log2(torch.as_tensor(padded_size))).item())
                adapt_kernel = padded_size - desired_size + 1

                d_blocks.append(
                    nn.Conv2d(in_channels=num_channels, out_channels=num_channels, padding=1, kernel_size=adapt_kernel)
                )

                size = desired_size

        self.d_blocks = nn.Sequential(*d_blocks)
        self.out_layer = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=2*latent_size, kernel_size=final_conv_image_size),
            nn.Flatten()
        )
        self.latent_size = latent_size

    def forward(self, x):
        out = self.d_blocks(x)
        out = self.out_layer(out)
        latent_mean, latent_log_std = torch.split(out, self.latent_size, dim=-1)
        return latent_mean, latent_log_std

    def code_distribution(self, x):
        latent_mean, latent_log_std = self(x)
        return torch.distributions.Normal(latent_mean, torch.exp(latent_log_std))


class Decoder(nn.Module):

    def __init__(self,
                 final_size,
                 base_num_features,
                 input_channels,
                 kernel_size,
                 final_conv_image_size,
                 latent_size):
        super(Decoder, self).__init__()

        assert final_size[0] == final_size[1]
        size = final_size[0]
        num_channels = input_channels

        u_blocks = []
        while size > final_conv_image_size:
            if size % 2 == 0:
                in_channels = num_channels * 2 if num_channels > 1 else base_num_features

                u_blocks.append(
                    UpSamplingResidualBlock(in_channels=in_channels,
                                            out_channels=num_channels,
                                            kernel_size=kernel_size)
                )
                num_channels = in_channels
                size //= 2
            else:
                padded_size = size + 2
                desired_size = int(2 ** torch.floor(torch.log2(torch.as_tensor(padded_size))).item())
                adapt_kernel = padded_size - desired_size + 1

                u_blocks.append(
                    nn.ConvTranspose2d(in_channels=num_channels,
                                       out_channels=num_channels,
                                       kernel_size=adapt_kernel,
                                       padding=1)
                )

                size = desired_size

        u_blocks.append(
            nn.Sequential(
                nn.ConvTranspose2d(in_channels=latent_size, out_channels=num_channels,
                                   kernel_size=final_conv_image_size),
                nn.LeakyReLU()
            )
        )

        self.u_blocks = nn.Sequential(*reversed(u_blocks))
        self.out_layer = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=input_channels, kernel_size=1),
            nn.Tanh() #nn.Sigmoid()
        )

        self.latent_size = latent_size
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=[self.latent_size, 1, 1])

    def forward(self, x):
        x = self.unflatten(x)
        y = self.u_blocks(x)
        y = self.out_layer(y)
        return y


@dataclass
class ELBO:

    elbo: torch.Tensor
    rec_ll: torch.Tensor
    kl: torch.Tensor

    def __repr__(self):
        return f"log p(x) >= { self.elbo.item():.4f} | E[p(x|z)] = { self.rec_ll.item():.4f} | KL(q||p) = { self.kl.item():.4f}"


class VAE(nn.Module):

    def __init__(self, input_size,
                 kernel_size,
                 input_channels,
                 base_num_features,
                 final_conv_image_size,
                 latent_size):
        super().__init__()

        self.encoder = Encoder(
            input_size=input_size,
            base_num_features=base_num_features,
            input_channels=input_channels,
            kernel_size=kernel_size,
            final_conv_image_size=final_conv_image_size,
            latent_size=latent_size
        )

        self.latent_size = self.encoder.latent_size

        self.decoder = Decoder(
            final_size=input_size,
            base_num_features=base_num_features,
            input_channels=input_channels,
            kernel_size=kernel_size,
            final_conv_image_size=final_conv_image_size,
            latent_size=latent_size
        )

        assert self.encoder.latent_size == self.decoder.latent_size

        self.latent_mean = nn.Parameter(torch.zeros(self.latent_size), requires_grad=False)
        self.latent_std = nn.Parameter(torch.ones(self.latent_size), requires_grad=False)
        self.latent_prior_distribution = torch.distributions.Normal(self.latent_mean, self.latent_std)

    def forward(self, x):
        return self.encoder(x)

    def reconstruction(self, x):
        latent_code, _ = self.encoder(x)
        reconstruction = self.decoder(latent_code)
        return reconstruction

    def elbo(self, x, num_encoder_samples=1, kl_weight=1.0):

        loss = torch.zeros(x.shape[0]).to(x.device)
        rec_ll = torch.zeros_like(loss)

        latent_dist = self.encoder.code_distribution(x)

        for i in range(num_encoder_samples):
            latent_code = latent_dist.rsample()
            """
            rec_ll -= torch.nn.functional.binary_cross_entropy(
                self.decoder(latent_code),
                x
            )
            """
            rec_ll -= torch.mean(torch.square(self.decoder(latent_code) - x))

        rec_ll /= num_encoder_samples
        rec_ll = torch.mean(rec_ll)
        kl = torch.mean(torch.distributions.kl_divergence(latent_dist, self.latent_prior_distribution))
        loss = rec_ll - kl_weight * kl
        return ELBO(
            elbo=loss,
            rec_ll=rec_ll,
            kl=kl
        )

    def sample(self, num_samples):
        latent_codes = self.latent_prior_distribution.sample((num_samples,))
        samples = self.decoder(latent_codes)
        return samples
