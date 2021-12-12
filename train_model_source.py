"""Train a source model using MNIST Data."""
import argparse
import numpy as np
import pathlib
import torch
import typing

from model.models import Net
from model.utils import GrayscaleToRgb
from model.make_data import get_datasets

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm

# Directory constants.
THIS_DIR = pathlib.Path(__file__).parent
DATA_DIR = THIS_DIR / "data"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_dataloaders(
    batch_size: int,
    num_workers: int = 1,
) -> typing.Tuple[DataLoader]::
    """Create training and validation data loaders.

    Args:
        batch_size: the size of a mini-batch.
        num_workers: the number workers on the local
            machine.
    Returns:
        A tuple of data loaders.
    """
    _, dataset = get_datasets(is_training=True)
    shuffled_indices = np.random.permutation(len(dataset))
    train_idx = shuffled_indices[:int(0.8*len(dataset))]
    val_idx = shuffled_indices[int(0.8*len(dataset)):]

    train_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True,
                              sampler=SubsetRandomSampler(train_idx),
                              num_workers=1, pin_memory=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, drop_last=False,
                            sampler=SubsetRandomSampler(val_idx),
                            num_workers=1, pin_memory=True)
    return train_loader, val_loader


def do_epoch(
    model: Net,
    dataloader: DataLoader,
    criterion: CrossEntropyLoss,
    optim=None,
) -> typing.Tuple[float]:
    """An epoch of training.

    Args:
        model: the model architecture.
        dataloader: the data loader.

    Returns:
        The mean loss and accuracy.
    """
    total_loss = 0
    total_accuracy = 0
    for x, y_true in tqdm(dataloader, leave=False):
        x, y_true = x.to(device), y_true.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y_true)

        if optim is not None:
            optim.zero_grad()
            loss.backward()
            optim.step()

        total_loss += loss.item()
        total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()
    mean_loss = total_loss / len(dataloader)
    mean_accuracy = total_accuracy / len(dataloader)

    return mean_loss, mean_accuracy


def main(args):
    train_loader, val_loader = create_dataloaders(args.batch_size)

    model = Net().to(device)
    optim = torch.optim.Adam(model.parameters())
    lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        patience=1,
        verbose=True,
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_accuracy = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss, train_accuracy = do_epoch(
            model,
            train_loader,
            criterion,
            optim=optim,
        )

        model.eval()
        with torch.no_grad():
            val_loss, val_accuracy = do_epoch(
                model,
                val_loader,
                criterion,
                optim=None,
            )

        tqdm.write(f'EPOCH {epoch:03d}: train_loss={train_loss:.4f}, '
                   f'train_accuracy={train_accuracy:.4f} '
                   f'val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}')

        if val_accuracy > best_accuracy:
            print('Saving model...')
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'trained_models/source.pt')

        lr_schedule.step(val_loss)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Train a network on MNIST')
    arg_parser.add_argument('--batch-size', type=int, default=32)
    arg_parser.add_argument('--epochs', type=int, default=5)
    args = arg_parser.parse_args()
    main(args)
