import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils.metrics import evaluate_on_test

def train_model(
    model, 
    train_loader, 
    val_loader, 
    test_loader=None,
    device='cuda', 
    epochs=10, 
    lr=1e-3,
    lr_decay=1.0,
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
    
    criterion = nn.BCEWithLogitsLoss()  
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)
    
    model.to(device)
    logger.info("Starting training with BCEWithLogitsLoss...")

    # track training/val metrics each epoch
    train_losses = []
    val_losses   = []
    train_accs   = []
    val_accs     = []

    # track the best model
    best_val_acc = 0.0
    best_model_wts = None
    best_epoch = 0

    # training loop
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

        # validation
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
