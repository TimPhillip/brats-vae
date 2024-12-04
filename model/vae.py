import torch
from torch import nn
from dataclasses import dataclass


class Encoder(nn.Sequential):

    def __init__(self, input_size,
                 num_layers,
                 num_filters,
                 kernel_size,
                 latent_size):

        self.latent_size = latent_size
        layers = []
        in_channels = 1

        assert input_size[0] == input_size[1]
        final_size = input_size[0] - (kernel_size - 1) * num_layers

        print(f"Encoder Final Size: {final_size}")

        for i in range(num_layers):
            conv  = nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_filters,
                kernel_size=kernel_size,
                stride=1,
            )
            layers.append(conv)
            in_channels = num_filters
            layers.append(nn.ReLU())

        layers.append(
            torch.nn.Flatten()
        )
        layers.append(
            torch.nn.Linear(final_size * final_size * num_filters, 2 * latent_size)
        )

        super(Encoder, self).__init__(*layers)

    def forward(self, x):
        out = super(Encoder, self).forward(x)
        latent_mean, latent_log_std = torch.split(out, self.latent_size, dim=-1)
        return latent_mean, latent_log_std

    def code_distribution(self, x):
        latent_mean, latent_log_std = self(x)
        return torch.distributions.Normal(latent_mean, torch.exp(latent_log_std))


class Decoder(nn.Sequential):

    def __init__(self, final_size, num_layers, num_filters, kernel_size, latent_size):

        layers = []
        in_channels = num_filters
        out_channels = 1

        assert final_size[0] == final_size[1]
        input_size = final_size[0] - (kernel_size - 1) * (num_layers)

        print(f"Decoder Initial Size: { input_size }")

        layers.append(
            torch.nn.Linear(latent_size, input_size * input_size * in_channels)
        )

        layers.append(
            torch.nn.Unflatten(dim=1, unflattened_size=[in_channels, input_size, input_size])
        )

        for i in range(num_layers):

            conv = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=num_filters if i < num_layers - 1 else out_channels,
                kernel_size=kernel_size
            )
            layers.append(conv)
            in_channels = num_filters
            layers.append(nn.ReLU() if i < num_layers - 1 else nn.Sigmoid())

        super(Decoder, self).__init__(*layers)


@dataclass
class ELBO:

    elbo: torch.Tensor
    rec_ll: torch.Tensor
    kl: torch.Tensor

    def __repr__(self):
        return f"log p(x) >= { self.elbo.item():.4f} | E[p(x|z)] = { self.rec_ll.item():.4f} | KL(q||p) = { self.kl.item():.4f}"


class VAE(nn.Module):

    def __init__(self, input_size,
                 latent_size,
                 num_encoder_layers,
                 num_decoder_layers):
        super().__init__()

        kernel_size = 32
        num_filters = 32

        self.latent_size = latent_size
        self.encoder = Encoder(input_size,
                               num_layers=num_encoder_layers,
                               num_filters=num_filters,
                               kernel_size=kernel_size,
                               latent_size=latent_size)
        self.decoder = Decoder(final_size=input_size,
                               num_layers=num_decoder_layers,
                               num_filters=num_filters,
                               kernel_size=kernel_size,
                               latent_size=latent_size)

        self.latent_mean = nn.Parameter(torch.zeros(latent_size))
        self.latent_std = nn.Parameter(torch.ones(latent_size))
        self.latent_prior_distribution = torch.distributions.Normal(self.latent_mean, self.latent_std)

    def forward(self, x):
        return self.encoder(x)

    def reconstruction(self, x):
        latent_code, _ = self.encoder(x)
        reconstruction = self.decoder(latent_code)
        return reconstruction

    def elbo(self, x, num_encoder_samples= 1):

        loss = torch.zeros(x.shape[0]).to(x.device)
        rec_ll = torch.zeros_like(loss)

        latent_dist = self.encoder.code_distribution(x)

        for i in range(num_encoder_samples):
            latent_code = latent_dist.sample()
            rec_ll -= torch.nn.functional.binary_cross_entropy(
                self.decoder(latent_code),
                x
            )

        rec_ll /= num_encoder_samples
        kl = torch.sum(torch.distributions.kl_divergence(latent_dist, self.latent_prior_distribution))
        loss = rec_ll - kl
        return ELBO(
            elbo=loss,
            rec_ll=rec_ll,
            kl=kl
        )

    def sample(self, num_samples):
        latent_codes = self.latent_prior_distribution.sample((num_samples,))
        samples = self.decoder(latent_codes)
        return samples


if __name__ == "__main__":

    x = torch.randn((1, 3, 31, 31))

    conv = torch.nn.ConvTranspose2d(
        in_channels=3,
        out_channels=1,
        kernel_size=24,
    )

    print(conv(x).shape)
    print(240 - (32 - 1) * 7)