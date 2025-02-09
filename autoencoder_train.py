import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import argparse
import logging
from tqdm import tqdm
from datetime import datetime
from custom_models.unet_mobilenetv2 import UNetMobileNet
from transforms.transforms import get_train_transforms, get_autoencoder_transforms
from utils.logger import get_logger
from utils.data_utils import create_autoencoder_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Train a U-Net MobileNet Autoencoder")
    parser.add_argument('--data_dir', type=str, default='data', help='Path to dataset root directory')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    return parser.parse_args()


def create_run_directory(model_name, lr, epochs, batch_size):
    """
    Create run directory inside 'runs/' with the hyperparams in the folder name.

    :param model_name: model name
    :type model_name: str
    :param lr: learning rate
    :type lr: float
    :param epochs: epochs
    :type epochs: int
    :param batch_size: batch size
    :type batch_size: int
    :return: run_dir
    :rtype: str
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_folder_name = f"{model_name}_run_lr-{lr}_ep-{epochs}_bs-{batch_size}_{timestamp}"
    run_dir = os.path.join("runs", run_folder_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir



def train_autoencoder(
        model, 
        train_loader, 
        val_loader=None, 
        device='cuda', 
        epochs=30, 
        lr=1e-4, 
        logger=None, 
        run_dir=None
):
    """
    Train the U-Net MobileNetV2 Autoencoder and return the best model based on validation loss.

    :param model: U-Net MobileNetV2 model
    :type model: torch.nn.Module
    :param train_loader: Training dataloader
    :type train_loader: torch.utils.data.DataLoader
    :param val_loader: Validation dataloader, defaults to None
    :type val_loader: torch.utils.data.DataLoader, optional
    :param device: Device to train on, defaults to 'cuda'
    :type device: str, optional
    :param epochs: Number of epochs, defaults to 30
    :type epochs: int, optional
    :param lr: Learning rate, defaults to 1e-3
    :type lr: float, optional
    :param logger: Logger, defaults to None
    :type logger: logging.Logger, optional
    :param run_dir: Path to save logs and models, defaults to None
    :type run_dir: str, optional
    """

    criterion = nn.MSELoss()  
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    logger.info("Starting training with MSE Loss...")

    train_losses = []
    val_losses = [] if val_loader else None

    best_val_loss = float('inf') 
    best_model_wts = None
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0

        for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images = images.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        log_message = f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f}"

        # Validation Phase
        if val_loader:
            model.eval()
            running_val_loss = 0.0

            with torch.no_grad():
                for images, _ in val_loader:
                    images = images.to(device)

                    outputs = model(images)
                    val_loss = criterion(outputs, images)
                    running_val_loss += val_loss.item()

            avg_val_loss = running_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            log_message += f" | Val Loss: {avg_val_loss:.4f}"

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                best_model_wts = {k: v.cpu() for k, v in model.state_dict().items()}  # Save best model weights
                logger.info(f"New best model found at epoch {best_epoch} with Val Loss: {best_val_loss:.4f}")

        logger.info(log_message)

    logger.info("Training complete.")

    if best_model_wts is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_wts.items()})
        logger.info(f"Best model restored from epoch {best_epoch} with lowest Val Loss: {best_val_loss:.4f}")

    return model, train_losses, val_losses



def main():
    args = parse_args()
    
    run_dir = create_run_directory("unet_mobilenet_autoencoder", args.lr, args.epochs, args.batch_size)
    
    logger = get_logger(run_dir)
    logger.info(f"Starting Autoencoder Training with args: {args}")
    logger.info(f"Run directory created at: {run_dir}")

    # transforms
    transform = get_train_transforms()

    # load dataset
    train_loader, test_loader = create_autoencoder_dataloaders(
        base_dir=args.data_dir,
        transform_train=transform,
        transform_test=transform,
        batch_size=args.batch_size
    )

    # instantiate model
    autoencoder = UNetMobileNet().to(args.device)

    # train 
    trained_model, train_losses, val_losses = train_autoencoder(
        model=autoencoder, 
        train_loader=train_loader, 
        val_loader=test_loader,
        device=args.device, 
        epochs=args.epochs, 
        lr=args.lr,
        logger=logger,
        run_dir=run_dir
    )

    # save model
    encoder_path = os.path.join(run_dir, "mobilenetv2_encoder.pth")
    torch.save(trained_model.encoder.state_dict(), encoder_path)
    logging.info(f"Encoder model saved to {encoder_path}")

    # plot loss
    loss_plot_path = os.path.join(run_dir, "loss_plot.png")
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(loss_plot_path)
    plt.close()
    logger.info(f"Loss plot saved to: {loss_plot_path}")

if __name__ == "__main__":
    main()


# python autoencoder_train.py --data_dir data --epochs 30 --batch_size 64 --lr 1e-4 --device cuda