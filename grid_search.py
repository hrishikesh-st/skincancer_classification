import os
import torch
import itertools
import matplotlib.pyplot as plt
from datetime import datetime

from utils.logger import get_logger
from utils.data_utils import create_dataloaders
from utils.train_utils import train_model
from utils.metrics import evaluate_on_test

from custom_models.mobilenetv2 import SkinCancerMobileNetV2
from transforms.transforms import get_train_transforms, get_test_transforms

# hyperparameter grid
param_grid = {
    "lr":        [1e-2, 5e-3, 1e-3, 5e-4, 1e-4],   # learning rates
    "epochs":    [10, 15, 20, 25, 30],             # epoch counters
    "batch_size":[16, 32, 64, 128, 256],           # batch sizes
    "lr_decay":  [1.0, 0.95, 0.9, 0.85, 0.8]       # decay factors
}

BASE_DIR    = "data"      
VAL_SPLIT   = 0.2         
SEED        = 21
DEVICE      = "cuda"      
NUM_WORKERS = 2           


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
    # setup master logger
    model_name = "mobilenetv2"
    master_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    master_log_dir   = os.path.join("runs", f"gridsearch_{master_timestamp}")
    os.makedirs(master_log_dir, exist_ok=True)
    
    master_logger = get_logger(master_log_dir)
    master_logger.info("Starting Grid Search for MobileNetV2 (constant model).")
    master_logger.info(f"Parameter Grid: {param_grid}")
    
    best_f1   = -1.0
    best_combo = None

    # set transforms    
    transform_train = get_train_transforms()
    transform_test  = get_test_transforms()

    # interate all hyperparameter combinations 
    for combo in itertools.product(
        param_grid["lr"],
        param_grid["epochs"],
        param_grid["batch_size"],
        param_grid["lr_decay"]
    ):
        lr_val, ep_val, bs_val, decay_val = combo
        
        # create run directory and logger
        run_dir = create_run_directory(model_name, lr_val, ep_val, bs_val, decay_val)
        logger = get_logger(run_dir)
        logger.info(f"Hyperparams => LR: {lr_val}, EPOCHS: {ep_val}, BS: {bs_val}, DECAY: {decay_val}")
        master_logger.info(f"=== START RUN: LR={lr_val}, EPOCHS={ep_val}, BS={bs_val}, DECAY={decay_val} ===")
        
        # instantiate dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            base_dir=BASE_DIR,
            transform_train=transform_train,
            transform_test=transform_test,
            batch_size=bs_val,
            val_split=VAL_SPLIT,
            seed=SEED,
            num_workers=NUM_WORKERS
        )
        
        # instantiate model
        model = SkinCancerMobileNetV2(pretrained=True, freeze_backbone=False)
        model_save_name = "mobilenetv2.pth"
        

        # train model
        trained_model, metrics_dict = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=None,  
            device=DEVICE,
            epochs=ep_val,
            lr=lr_val,
            logger=logger,
            lr_decay=decay_val  
        )
        
        # training vs validation loss plot
        loss_plot_path = os.path.join(run_dir, "loss_plot.png")
        plt.figure()
        plt.plot(metrics_dict['train_losses'], label='Train Loss')
        plt.plot(metrics_dict['val_losses'],   label='Val Loss')
        plt.title("Training vs. Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(loss_plot_path)
        plt.close()
        
        logger.info(f"Loss plot saved: {loss_plot_path}")
        
        # training vs validation accuracy plot
        acc_plot_path = os.path.join(run_dir, "accuracy_plot.png")
        plt.figure()
        plt.plot(metrics_dict['train_accuracies'], label='Train Accuracy')
        plt.plot(metrics_dict['val_accuracies'],   label='Val Accuracy')
        plt.title("Training vs. Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.savefig(acc_plot_path)
        plt.close()
        
        logger.info(f"Accuracy plot saved: {acc_plot_path}")
        

        # evaluate on test set
        logger.info("Evaluating on test set...")
        final_metrics = evaluate_on_test(trained_model, test_loader, device=DEVICE)
        
        # log results
        logger.info("==== Final Test Metrics ====")
        logger.info(f"Accuracy:   {final_metrics['accuracy']:.2f}%")
        logger.info(f"Precision:  {final_metrics['precision']:.4f}")
        logger.info(f"Recall:     {final_metrics['recall']:.4f}")
        logger.info(f"Specificity:{final_metrics['specificity']:.4f}")
        logger.info(f"F1 Score:   {final_metrics['f1score']:.4f}")
        
        # save confusion matrix
        cm_path = os.path.join(run_dir, "confusion_matrix.png")
        final_metrics['cm_fig'].savefig(cm_path)
        logger.info(f"Confusion Matrix saved: {cm_path}")
        
        # save model
        model_path = os.path.join(run_dir, model_save_name)
        torch.save(trained_model.state_dict(), model_path)
        logger.info(f"Model saved at {model_path}")
        
        # F1 score comparison to find best model
        if final_metrics["f1score"] > best_f1:
            best_f1 = final_metrics["f1score"]
            best_combo = (lr_val, ep_val, bs_val, decay_val)
            master_logger.info(f"New BEST F1 Score: {best_f1:.4f} with {best_combo}")
        else:
            master_logger.info(f"Finished run with F1= {final_metrics['f1score']:.4f} (NOT best).")
    
    master_logger.info("Grid Search Complete.")
    master_logger.info(f"Best F1 Score: {best_f1:.4f} with {best_combo}")

if __name__ == "__main__":
    main()
