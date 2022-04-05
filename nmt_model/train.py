import argparse

import torch

from src.models import NMTTransformer
import src.schedulers
from src.trainers.trainer import Trainer
from src.utils import parse_config, set_seed, init_obj, prepare_dataloaders
from src.writers import WandbWriter


def main(args):
    training_config = parse_config(args.config_path)

    device = "cpu"
    if torch.cuda.is_available() and args.use_cuda:
        device = "cuda"
    print("Training on", device)

    set_seed(training_config["seed"])

    model = NMTTransformer(**training_config["model"]).to(device)
    optimizer = init_obj(torch.optim, training_config["optimizer"], params=model.parameters())
    scheduler = init_obj(
        [src.schedulers, torch.optim.lr_scheduler],
        training_config["scheduler"],
        optimizer=optimizer
    )

    criterion = init_obj(torch.nn, training_config["criterion"])

    dataloaders = prepare_dataloaders(training_config["data"])

    project_name = training_config.pop("logging_project_name")
    writer = WandbWriter(project_name, training_config)

    trainer = Trainer(model, criterion, optimizer, device, dataloaders["train"], dataloaders["val"], scheduler,
                      writer, **training_config["trainer"])

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an NMT model")
    parser.add_argument("--config_path", "-c", type=str, required=True)
    parser.add_argument("--checkpoint_folder", "-f", type=str, required=True)
    parser.add_argument("--not_use_cuda", dest="use_cuda", action="store_false")

    args = parser.parse_args()

    main(args)
