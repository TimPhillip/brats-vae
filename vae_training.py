import os

import omegaconf
import torch
from torchvision.transforms import transforms
import torch.utils.data
import logging
from matplotlib import pyplot as plt
import math
import argparse
import numpy as np
import hydra
from omegaconf import DictConfig
import mlflow

from model.vae import VAE


def get_image_for_plotting(img):

    img = torch.squeeze(img)

    # rearange the image axis
    if len(img.shape) == 3:
        img = torch.permute(img, (1, 2, 0))

    # denormalize the image
    img = img * 0.5 + 0.5

    img = img.cpu().numpy()

    return img


def plot_vae_samples(samples):

    num_samples = len(samples)
    n_rows = int(math.sqrt(num_samples))
    assert n_rows * n_rows == num_samples, "Number of samples must be a square number."

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_rows)
    for ax, sample in zip(np.reshape(axs, -1), samples):
        img = get_image_for_plotting(sample)
        ax.imshow(img)


def plot_reconstruction(imgs, img_recons):

    num_examples = len(imgs)
    fig, axs = plt.subplots(nrows=2, ncols=num_examples, figsize=(8, 4))

    for i, (img, img_recon) in enumerate(zip(imgs, img_recons)):

        # image
        axs[0][i].imshow(get_image_for_plotting(img))
        axs[0][i].set_title('Original')

        # reconstruction
        axs[1][i].imshow(get_image_for_plotting(img_recon))
        axs[1][i].set_title('Reconstructed')

    fig.tight_layout()


def train(model, training_data, validation_data, device,
          num_epochs=100,
          batch_size=64,
          learning_rate=1e-3,
          num_encoder_samples=4,
          elbo_kl_weight=1.0,
          plot_samples=True,
          log_loss_every=1,
          plot_reconstruction_every=100):

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    training_data_loader = torch.utils.data.DataLoader(
        dataset=training_data,
        batch_size=batch_size,
        shuffle=True
    )

    validation_data_loader = torch.utils.data.DataLoader(
        dataset=validation_data,
        batch_size=batch_size,
        shuffle=True
    )

    logging.info(("-" * 30) + " Training " + ("-" * 30))

    for epoch in range(num_epochs):

        logging.info(("-" * 10) + f" Epoch { epoch} " + ("-" * 10))

        model.train()

        epoch_stats = {
            'elbo': 0.0,
            'rec_ll': 0.0,
            'kl': 0.0,
        }
        num_epoch_steps = 0

        for i, batch in enumerate(training_data_loader):
            batch = batch.to(device)

            optimizer.zero_grad()

            elbo = model.elbo(batch, num_encoder_samples=num_encoder_samples, kl_weight=elbo_kl_weight)
            (-elbo.elbo).backward()
            optimizer.step()

            if i % log_loss_every == 0:
                logging.info(f"Epoch { epoch }: {elbo}")

            num_epoch_steps += 1
            for metric in epoch_stats.keys():
                epoch_stats[metric] += getattr(elbo, metric)

        for metric in epoch_stats.keys():
            mlflow.log_metric(metric,
                              epoch_stats[metric] / num_epoch_steps,
                              step=epoch)

        # run evalutions
        with torch.no_grad():
            model.eval()

            eval_stats = {
                'elbo': 0.0,
                'rec_ll': 0.0,
                'kl': 0.0
            }
            num_evaluation_steps = 0

            for batch in validation_data_loader:
                batch = batch.to(device)

                elbo = model.elbo(batch, num_encoder_samples=num_encoder_samples, kl_weight=elbo_kl_weight)
                num_evaluation_steps += 1
                for metric in eval_stats.keys():
                    eval_stats[metric] += getattr(elbo, metric)

            try:
                # plot reconstructions with the last batch from evaluation
                img = batch[0:4]
                img_recon = model.reconstruction(img)
                plot_reconstruction(img, img_recon)
                recon_filename = f'recon/recon_epoch_{epoch}_{i}.png'
                plt.savefig(recon_filename)
                mlflow.log_artifact(recon_filename)
                plt.close()

                # plot samples
                samples = model.sample(num_samples=9)
                plot_vae_samples(samples)
                sample_filename = f'samples/samples_epoch_{epoch}.png'
                plt.savefig(sample_filename)
                mlflow.log_artifact(sample_filename)
                plt.close()

            except KeyboardInterrupt:
                raise KeyboardInterrupt()

            except Exception as e:
                logging.error(e)

            for metric in epoch_stats.keys():
                mlflow.log_metric(f"val_{metric}",
                                  epoch_stats[metric] / num_evaluation_steps,
                                  step=epoch)


def select_device(config):
    if config.device == 'gpu' and torch.cuda.is_available() and torch.cuda.device_count() >= 1:
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


@hydra.main(config_path="config",
            config_name="vae-training.yaml",
            version_base="1.3")
def main(config: DictConfig):

    # configure logging
    configure_logging(filename=config.logging.filename)
    logging.info(f"Started new experiment at { os.getcwd() }")

    # configure tracking
    tracking_uri = hydra.utils.to_absolute_path(config.tracking.uri)
    logging.info(f"MLFlow Tracking at {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)

    exp = mlflow.get_experiment_by_name(config.tracking.experiment)
    if exp is None:
        mlflow.create_experiment(config.tracking.experiment)
        logging.info(f"Created a new experiment with name {config.tracking.experiment} in MLFlow")

    mlflow.set_experiment(config.tracking.experiment)
    logging.info(f"Started a new run in MLFlow experiment {config.tracking.experiment}")

    with mlflow.start_run() as run:

        try:
            device = select_device(config)

            # transform the dataset to fit the model size
            image_size = config.model.image_size
            data_transforms = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)
            ])

            # create datasets
            training_data = hydra.utils.instantiate(config.dataset.splits.training, transform=data_transforms)
            validation_data = hydra.utils.instantiate(config.dataset.splits.validation, transform=data_transforms)

            # initialize the model
            model = VAE(input_size=(image_size, image_size),
                        input_channels=config.dataset.image_channels,
                        kernel_size=config.model.conv_kernel_size,
                        base_num_features=config.model.base_num_features,
                        final_conv_image_size=config.model.final_conv_image_size,
                        latent_size=config.model.latent_size)
            model.to(device)

            # finally run the training
            train(model, training_data, validation_data, device,
                  num_epochs=config.training.num_epochs,
                  plot_samples=True,
                  num_encoder_samples=config.training.num_encoder_samples,
                  elbo_kl_weight=config.training.elbo_kl_weight,
                  batch_size=config.training.batch_size,
                  log_loss_every=config.training.log_loss_every_steps,
                  plot_reconstruction_every=config.training.plot_reconstruction_every_steps,
                  learning_rate=config.training.learning_rate)

        except KeyboardInterrupt:
            logging.info(("-" * 30) + " End " + ("-" * 30))
            logging.warning("Experiment interrupted by keyboard interrupt.")

        except Exception as e:
            logging.error(e)
        finally:
            mlflow.log_artifact(config.logging.filename)


if __name__ == '__main__':
    main()