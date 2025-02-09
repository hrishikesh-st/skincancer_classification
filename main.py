import torch
import argparse
import os
from transforms.transforms import get_train_transforms, get_test_transforms
from utils.data_utils import create_dataloaders
from models.baseline import Baseline
from models.mobilenetv2 import SkinCancerMobileNetV2
from utils.train_utils import train_model
from utils.logger import get_logger
from datetime import datetime
import matplotlib.pyplot as plt

from utils.metrics import evaluate_on_test

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Skin Cancer Classifier with Val Split')
    parser.add_argument('--base_dir', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', type=float, default=1.0)
    parser.add_argument('--val_split', type=float, default=0.2, 
                        help='Fraction of training data used for validation')
    parser.add_argument('--seed', type=int, default=21, help='Random seed for splitting')
    parser.add_argument('--device', type=str, default='cuda', help='Choose device: cuda or cpu')
    
    parser.add_argument('--model_name', type=str, default='mobilenetv2',
                        choices=['baseline', 'mobilenetv2'],
                        help='Select which model architecture to train')

    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze the feature extraction layers in MobileNetV2')

    return parser.parse_args()


def create_run_directory(model_name, lr, epochs, batch_size, lr_decay):
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
    :param lr_decay: learning rate decay factor
    :type lr_decay: float
    :return: run_dir
    :rtype: str
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_folder_name = f"{model_name}_run_lr-{lr}_ep-{epochs}_bs-{batch_size}_decay-{lr_decay}_{timestamp}"
    run_dir = os.path.join("runs", run_folder_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def main():
    args = parse_args()

    # run dir
    run_dir = create_run_directory(args.model_name, args.lr, args.epochs, args.batch_size, args.lr_decay)

    # logger
    logger = get_logger(run_dir)
    logger.info(f"Run directory created at: {run_dir}")
    logger.info(f"Arguments: {args}")

    # set transforms
    transform_train = get_train_transforms()
    transform_test = get_test_transforms()
    
    # instantiate dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        base_dir=args.base_dir,
        transform_train=transform_train,
        transform_test=transform_test,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed
    )
    logger.info("Dataloaders created successfully.")

    # instantiate model
    if args.model_name == 'baseline':
        model = Baseline()
        model_save_name = 'baseline.pth'
    else:
        model = SkinCancerMobileNetV2(
            pretrained=True,
            freeze_backbone=args.freeze_backbone
        )
        model_save_name = 'mobilenetv2.pth'

    logger.info(f"Using model: {args.model_name}")

    # train and evaluate on test set
    trained_model, metrics_dict = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=None,  # We'll do final test evaluation below
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
        lr_decay=args.lr_decay,
        logger=logger
    )

    # training vs validation loss plot
    loss_plot_path = os.path.join(run_dir, "loss_plot.png")
    plt.figure()
    plt.plot(metrics_dict['train_losses'], label='Train Loss')
    plt.plot(metrics_dict['val_losses'], label='Val Loss')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(loss_plot_path)
    plt.close()
    logger.info(f"Loss plot saved to: {loss_plot_path}")

    # training vs validation accuracy plot
    acc_plot_path = os.path.join(run_dir, "accuracy_plot.png")
    plt.figure()
    plt.plot(metrics_dict['train_accuracies'], label='Train Accuracy')
    plt.plot(metrics_dict['val_accuracies'], label='Val Accuracy')
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.savefig(acc_plot_path)
    plt.close()
    logger.info(f"Accuracy plot saved to: {acc_plot_path}")

    # evaluate on the test set for final metrics
    logger.info("Evaluating on test set...")
    final_metrics = evaluate_on_test(trained_model, test_loader, device=args.device)

    logger.info("==== Final Test Metrics ====")
    logger.info(f"Accuracy:   {final_metrics['accuracy']:.2f}%")
    logger.info(f"Precision:  {final_metrics['precision']:.4f}")
    logger.info(f"Recall:     {final_metrics['recall']:.4f}")
    logger.info(f"Specificity:{final_metrics['specificity']:.4f}")
    logger.info(f"F1 Score:   {final_metrics['f1score']:.4f}")

    # Save confusion matrix
    cm_path = os.path.join(run_dir, "confusion_matrix.png")
    final_metrics['cm_fig'].savefig(cm_path)
    logger.info(f"Confusion matrix figure saved to: {cm_path}")

    # save the trained model
    model_path = os.path.join(run_dir, model_save_name)
    torch.save(trained_model.state_dict(), model_path)
    logger.info(f"Model saved successfully as {model_path}")

if __name__ == '__main__':
    main()