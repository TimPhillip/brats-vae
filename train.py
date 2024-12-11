import torch
import torch.utils.data
import logging
from matplotlib import pyplot as plt
import math
import argparse
import numpy as np

from data.brain import BrainDataset
from model.vae import VAE


def plot_vae_samples(samples):

    num_samples = len(samples)
    n_rows = int(math.sqrt(num_samples))
    assert n_rows * n_rows == num_samples, "Number of samples must be a square number."

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_rows)
    for ax, sample in zip(np.reshape(axs, -1), samples):
        img = torch.squeeze(sample).cpu().numpy()
        ax.imshow(img, cmap='Greys_r')


def train(model, data, device,
          num_epochs=100,
          batch_size=64,
          num_encoder_samples=4,
          plot_samples=True):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    data_loader = torch.utils.data.DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=True
    )

    for epoch in range(num_epochs):

        if plot_samples:
            with torch.no_grad():
                model.eval()
                samples = model.sample(num_samples=9)
                plot_vae_samples(samples)
                plt.savefig(f'samples/samples_epoch_{epoch}.png')

        model.train()
        for batch in data_loader:

            batch = batch.to(device)

            optimizer.zero_grad()

            elbo = model.elbo(batch, num_encoder_samples=num_encoder_samples)
            (-elbo.elbo).backward()
            optimizer.step()

            logging.info(f"Epoch { epoch }: {elbo}")


def parse_training_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to brain dataset.")
    parser.add_argument("--conv-kernel-size", type=int, default=5, help="Convolution kernel size.")
    parser.add_argument("--conv-filters", type=int, default=16, help="Number of convolution filters.")
    parser.add_argument("--device",
                        type=str, choices=['cpu', 'gpu'],
                        default='gpu',
                        help="Which device to use.")
    parser.add_argument("--plot_samples", type=bool, default=True, help="Whether to plot samples.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num_encoder_samples", type=int, default=4,
                        help="Number of encoder samples in ELBO during training.")
    args = parser.parse_args()
    return args


def select_device(args):
    if args.device == 'gpu' and torch.cuda.is_available() and torch.cuda.device_count() >= 1:
        device = torch.device('cuda', index=0)
        logging.info(f"Running experiment on { torch.cuda.get_device_name(device) }")
    else:
        device = torch.device('cpu')
        logging.info(f"Running experiment on CPU")

    return device


def configure_logging(filename):
    logging.basicConfig(level=logging.INFO,
                        filename=filename,
                        filemode="w",
                        format="%(asctime)s [%(levelname)s]  %(message)s",
                        datefmt="%d.%m.%Y %H:%M:%S")


def main():

    configure_logging(filename="logs/train.log")
    args = parse_training_config()
    device = select_device(args)

    dataset = BrainDataset(args.dataset)
    model = VAE(input_size=(240, 240),
                kernel_size=args.conv_kernel_size,
                base_num_features=args.conv_filters)
    model.to(device)

    train(model, dataset, device,
          num_epochs=args.epochs,
          plot_samples=args.plot_samples,
          num_encoder_samples=args.num_encoder_samples,
          batch_size=args.batch_size)


if __name__ == '__main__':
    main()