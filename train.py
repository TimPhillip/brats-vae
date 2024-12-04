import torch
import torch.utils.data
import logging
from matplotlib import pyplot as plt
import math
import argparse

from data.brain import BrainDataset
from model.vae import VAE


def plot_vae_samples(samples):

    num_samples = len(samples)
    n_rows = int(math.sqrt(num_samples))
    assert n_rows * n_rows == num_samples, "Number of samples must be a square number."

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_rows)
    for ax, sample in zip(axs, samples):
        img = torch.squeeze(sample).cpu().numpy()
        ax[0].imshow(img, cmap='Greys_r')


def train(model, data, device, num_epochs=100):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    data_loader = torch.utils.data.DataLoader(
        dataset=data,
        batch_size=64,
        shuffle=True,
    )

    for epoch in range(num_epochs):

        with torch.no_grad():
            model.eval()
            samples = model.sample(num_samples=9)
            plot_vae_samples(samples)
            plt.savefig(f'samples/samples_epoch_{epoch}.png')

        model.train()
        for batch in data_loader:

            batch = batch.to(device)

            optimizer.zero_grad()

            elbo = model.elbo(batch)
            (-elbo.elbo).backward()
            optimizer.step()

            print(f"Epoch { epoch }: {elbo}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to brain dataset.")
    parser.add_argument("--num_layers", type=int, default=7, help="Number of layers.")
    parser.add_argument("--latent_size", type=int, default=32, help="Size of latent space.")
    parser.add_argument("--device", type=str, choices=['cpu', 'gpu'], default='gpu', help="Which device to use.")
    args = parser.parse_args()

    if args.device == 'gpu' and torch.cuda.is_available() and torch.cuda.device_count() >= 1:
        device = torch.device('cuda', index=0)
        print(f"Running experiment on { torch.cuda.get_device_name(device) }")
    else:
        device = torch.device('cpu')
        print(f"Running experiment on CPU")

    dataset = BrainDataset(args.dataset)
    model = VAE(input_size=(240, 240),
                latent_size=args.latent_size,
                num_encoder_layers=args.num_layers,
                num_decoder_layers=args.num_layers)
    model.to(device)
    train(model, dataset, device, num_epochs=args.epochs)


if __name__ == '__main__':
    main()