import torch
import torch.optim as optim
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import torchvision.models as models
from tqdm import tqdm

from transforms.transforms import get_train_transforms, get_test_transforms
from utils.data_utils import create_dataloaders
from utils.metrics import evaluate_on_test
from custom_models.encoder_classifier import EncoderClassifier
from utils.logger import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Skin Cancer Classifier with Val Split')
    parser.add_argument('--base_dir', type=str, default='data')
    parser.add_argument('--epochs1', type=int, default=10)
    parser.add_argument('--epochs2', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr1', type=float, default=1e-3)
    parser.add_argument('--lr2', type=float, default=5e-5)
    parser.add_argument('--lr_decay', type=float, default=1.0)
    parser.add_argument('--val_split', type=float, default=0.2, 
                        help='Fraction of training data used for validation')
    parser.add_argument('--seed', type=int, default=21, help='Random seed for splitting')
    parser.add_argument('--device', type=str, default='cuda', help='Choose device: cuda or cpu')
    
    parser.add_argument('--model_name', type=str, default='encoder_classifier')

    return parser.parse_args()


def create_run_directory(model_name, lr1, lr2, epochs1, epochs2, batch_size, lr_decay, seed):
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
    run_folder_name = f"{model_name}_run_lr1-{lr1}_lr2-{lr2}_ep1-{epochs1}_ep2-{epochs2}_bs-{batch_size}_decay-{lr_decay}_seed-{seed}_{timestamp}"
    run_dir = os.path.join("runs", run_folder_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def print_trainable_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n Total Parameters: {total_params:,}")
    print(f" Trainable Parameters: {trainable_params:,}\n")


def freeze_encoder(model, freeze=True):
    """
    Freeze encoder layers

    :param model: MOdel
    :type model: torch.nn.Module
    :param freeze: Freeze or unfreeze, defaults to True
    :type freeze: bool, optional
    """
    for i, (name, child) in enumerate(model.encoder.named_children()):
        if freeze:
           for param in child.parameters():
               param.requires_grad = False  # freeze
        else:
            for param in child.parameters():
                param.requires_grad = True   # unfreeze 


    print("\n Trainable Parameters After Freezing:")
    print_trainable_parameters(model)


def train_model(
    model, 
    train_loader, 
    val_loader, 
    test_loader=None,
    device='cuda', 
    epochs=10, 
    lr=1e-3,
    lr_decay=1.0,
    optimizer=None,
    logger=None
):
    """
    Model training and evaluation

    :param model: Model type
    :type model: torch.nn.Module
    :param train_loader: Train dataloader
    :type train_loader: torch.utils.data.DataLoader
    :param val_loader: Val dataloader
    :type val_loader: torch.utils.data.DataLoader
    :param test_loader: Test dataloader, defaults to None
    :type test_loader: torch.utils.data.DataLoader, optional
    :param device: Device to train, defaults to 'cuda'
    :type device: str, optional
    :param epochs: Number of epochs, defaults to 10
    :type epochs: int, optional
    :param lr: Learning rate, defaults to 1e-3
    :type lr: float, optional
    :param lr_decay: Learning rate decay factor, defaults to 1.0 
    :type lr_decay: float, optional
    :param logger: Logger, defaults to None
    :type logger: logging.Logger, optional
    :return: Trained model and metrics
    :rtype: torch.nn.Module, dict
    """
    
    # criterion = CustomLoss(alpha=0.5)
    criterion = nn.BCEWithLogitsLoss()  
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=lr_decay)
    
    model.to(device)
    logger.info("Starting training with CustomLoss...")

    train_losses = []
    val_losses   = []
    train_accs   = []
    val_accs     = []

    best_val_acc = 0.0
    best_model_wts = None
    best_epoch = 0

    # train
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train   = 0

        epoch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for images, labels in epoch_pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

            preds = torch.sigmoid(outputs) > 0.5  
            
            correct_train += (preds == labels).sum().item()
            total_train   += labels.size(0)

            epoch_pbar.set_postfix({"train_loss": f"{loss.item():.4f}"})
        
        avg_train_loss = running_train_loss / len(train_loader)
        train_accuracy = 100.0 * correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accs.append(train_accuracy)

        # val
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val   = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float()

                outputs = model(images).squeeze(1)
                val_loss = criterion(outputs, labels)
                running_val_loss += val_loss.item()

                preds = torch.sigmoid(outputs) > 0.5
                correct_val += (preds == labels).sum().item()
                total_val   += labels.size(0)

        avg_val_loss = running_val_loss / len(val_loader)
        val_accuracy = 100.0 * correct_val / total_val
        val_losses.append(avg_val_loss)
        val_accs.append(val_accuracy)

        logger.info(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%"
        )

        scheduler.step()

        # save the best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_epoch = epoch + 1
            best_model_wts = {k: v.cpu() for k, v in model.state_dict().items()}
            logger.info(
                f"New best validation accuracy: {best_val_acc:.2f}% "
                f"at epoch {best_epoch}"
            )

    logger.info("Training complete.")

    # restore the best model
    if best_model_wts is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_wts.items()})
        logger.info(f"Best model restored from epoch {best_epoch} with val acc={best_val_acc:.2f}%")

    # evaluate on the test set
    test_metrics = None
    if test_loader is not None:
        logger.info("Evaluating on test set using evaluate_on_test()...")
        test_metrics = evaluate_on_test(model, test_loader, device=device)
        logger.info("==== Final Test Metrics ====")
        logger.info(f"Accuracy:    {test_metrics['accuracy']:.2f}%")
        logger.info(f"Precision:   {test_metrics['precision']:.4f}")
        logger.info(f"Recall:      {test_metrics['recall']:.4f}")
        logger.info(f"Specificity: {test_metrics['specificity']:.4f}")
        logger.info(f"F1 Score:    {test_metrics['f1score']:.4f}")

    # best model + metrics
    return model, {
        "train_losses": train_losses,
        "val_losses":   val_losses,
        "train_accuracies": train_accs,
        "val_accuracies": val_accs,
        "test_metrics": test_metrics,
        "best_val_acc": best_val_acc,
        "best_epoch":   best_epoch
    }

def main():
    args = parse_args()
    
    run_dir = create_run_directory(args.model_name, args.lr1, args.lr2, args.epochs1, args.epochs2, args.batch_size, args.lr_decay, args.seed)
    
    logger = get_logger(run_dir)
    logger.info(f"Starting Training with args: {args}")
    logger.info(f"Run directory created at: {run_dir}")

    # set transforms
    transform_train = get_train_transforms()
    transform_test = get_test_transforms()

    # instantiate dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        base_dir=args.base_dir,
        transform_train=transform_train,
        transform_test=transform_test,
        batch_size=args.batch_size,
        val_split=0.2,
        seed=args.seed
    )
    logger.info("Dataloaders created successfully.")

    # load encoder
    finetuned_encoder_path = "runs/unet_mobilenet_autoencoder_run_lr-0.0001_ep-30_bs-64_2025-02-09_15-34-05_FINALRUN_Without_NORMALIZE/mobilenetv2_encoder.pth"
    encoder = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).features
    encoder.load_state_dict(torch.load(finetuned_encoder_path, map_location=args.device))
    encoder.to(args.device)
    encoder.eval()

    # instantiate model
    model = EncoderClassifier(encoder=encoder)
    model_save_name = "encoder_classifier.pth"

    # Train Dense Layers (Freeze Encoder)
    logger.info("==== Training Binary Classifier Stage 1 ====")
    logger.info("Training Dense Layers Only...")
    freeze_encoder(model, freeze=True)  # Fully freeze encoder
    print_trainable_parameters(model)

    optimizer = optim.Adam([
    {'params': model.fc1.parameters()},
    {'params': model.fc2.parameters()},
    {'params': model.fc3.parameters()}
    ], lr=args.lr1)

    # stage 1: Train Dense Layers
    trained_model, stage1_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=None,
        device=args.device,
        epochs=args.epochs1,
        optimizer=optimizer,
        lr_decay=args.lr_decay,
        logger=logger
    )

    # training/validation loss plot
    loss_plot_path = os.path.join(run_dir, "stage1_loss_plot.png")
    plt.figure()
    plt.plot(stage1_metrics['train_losses'], label='Train Loss')
    plt.plot(stage1_metrics['val_losses'], label='Val Loss')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(loss_plot_path)
    plt.close()
    logger.info(f"Loss plot saved to: {loss_plot_path}")

    # training/validation accuracy plot
    acc_plot_path = os.path.join(run_dir, "stage1_accuracy_plot.png")
    plt.figure()
    plt.plot(stage1_metrics['train_accuracies'], label='Train Accuracy')
    plt.plot(stage1_metrics['val_accuracies'], label='Val Accuracy')
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.savefig(acc_plot_path)
    plt.close()
    logger.info(f"Accuracy plot saved to: {acc_plot_path}")

    # evaluate on the test set for final metrics
    logger.info("Evaluating on test set...")
    stage1_final_metrics = evaluate_on_test(trained_model, test_loader, device=args.device)

    logger.info("==== Final Stage1 Metrics ====")
    logger.info(f"Accuracy:   {stage1_final_metrics['accuracy']:.2f}%")
    logger.info(f"Precision:  {stage1_final_metrics['precision']:.4f}")
    logger.info(f"Recall:     {stage1_final_metrics['recall']:.4f}")
    logger.info(f"Specificity:{stage1_final_metrics['specificity']:.4f}")
    logger.info(f"F1 Score:   {stage1_final_metrics['f1score']:.4f}")

    # save confusion matrix
    cm_path = os.path.join(run_dir, "confusion_matrix_stage1.png")
    stage1_final_metrics['cm_fig'].savefig(cm_path)
    logger.info(f"Confusion matrix figure saved to: {cm_path}")

    # stage 2: Fine-Tune Encoder
    # unfreeze Encoder 
    logger.info("==== Training Binary Classifier Stage 2 ====")
    logger.info("Fine-Tuning...")
    freeze_encoder(trained_model, freeze=False)  
    print_trainable_parameters(model)  
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr2)

    finetuned_model, stage2_metrics = train_model(
        model=trained_model,  
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=None,
        device=args.device,
        epochs=args.epochs2,
        optimizer=optimizer,
        lr_decay=args.lr_decay,
        logger=logger
    )

    # training/validation loss plot
    loss_plot_path = os.path.join(run_dir, "stage2_loss_plot.png")
    plt.figure()
    plt.plot(stage2_metrics['train_losses'], label='Train Loss')
    plt.plot(stage2_metrics['val_losses'], label='Val Loss')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(loss_plot_path)
    plt.close()
    logger.info(f"Loss plot saved to: {loss_plot_path}")

    # training/validation accuracy plot
    acc_plot_path = os.path.join(run_dir, "stage2_accuracy_plot.png")
    plt.figure()
    plt.plot(stage2_metrics['train_accuracies'], label='Train Accuracy')
    plt.plot(stage2_metrics['val_accuracies'], label='Val Accuracy')
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.savefig(acc_plot_path)
    plt.close()
    logger.info(f"Accuracy plot saved to: {acc_plot_path}")

    # evaluate on the test set for final metrics
    logger.info("Evaluating on test set...")
    stage2_final_metrics = evaluate_on_test(finetuned_model, test_loader, device=args.device)

    logger.info("==== Final Stage2 Metrics ====")
    logger.info(f"Accuracy:   {stage2_final_metrics['accuracy']:.2f}%")
    logger.info(f"Precision:  {stage2_final_metrics['precision']:.4f}")
    logger.info(f"Recall:     {stage2_final_metrics['recall']:.4f}")
    logger.info(f"Specificity:{stage2_final_metrics['specificity']:.4f}")
    logger.info(f"F1 Score:   {stage2_final_metrics['f1score']:.4f}")

    # save confusion matrix
    cm_path = os.path.join(run_dir, "confusion_matrix_stage2.png")
    stage2_final_metrics['cm_fig'].savefig(cm_path)
    logger.info(f"Confusion matrix figure saved to: {cm_path}")

    # save the trained model
    model_path = os.path.join(run_dir, model_save_name)
    torch.save(finetuned_model.state_dict(), model_path)
    logger.info(f"Model saved successfully as {model_path}")

if __name__ == '__main__':
    main()

# Best results:
# python ensemble_train_final.py --base_dir data --epochs1 25 --epochs2 40 --batch_size 64 --lr1 1e-3 --lr2 5e-5 --lr_decay 0.9 --val_split 0.2 --lr_decay 0.75 --model_name encoder_classifier --device cuda