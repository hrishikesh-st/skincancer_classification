import os
import torch
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, random_split
from datasets.skin_cancer_dataset import SkinCancerDataset

def create_dataloaders(
    base_dir='data',
    transform_train=None,
    transform_test=None,
    batch_size=32,
    num_workers=2,
    val_split=0.2,       
    seed=21
):
    """
    Create train, val, test dataloaders.

    :param base_dir: Path to data directory (train/ and test/ subfolders).
    :type base_dir: str, optional
    :param transform_train: Augmentations for train set
    :type transform_train: torchvision.transforms.Compose, optional
    :param transform_test: Augmentations for test set
    :type transform_test: torchvision.transforms.Compose, optional
    :param batch_size: batch size, defaults to 32
    :type batch_size: int, optional
    :param num_workers: dataloader workers, defaults to 2
    :type num_workers: int, optional
    :param val_split: validation split, defaults to 0.2
    :type val_split: float, optional
    :param seed: random seed, defaults to 42
    :type seed: int, optional
    :return: train_loader, val_loader, test_loader
    :rtype: torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader
    """

    # training dirs
    train_benign_dir = os.path.join(base_dir, 'train', 'Benign')
    train_malign_dir = os.path.join(base_dir, 'train', 'Malignant')
    
    # testing dirs
    test_benign_dir  = os.path.join(base_dir, 'test', 'Benign')
    test_malign_dir  = os.path.join(base_dir, 'test', 'Malignant')
    
    # instantiate train datasets
    ds_train_benign = SkinCancerDataset(train_benign_dir, transform=transform_train)
    ds_train_malignant = SkinCancerDataset(train_malign_dir, transform=transform_train)
    
    # instantiate test datasets
    ds_test_benign  = SkinCancerDataset(test_benign_dir,  transform=transform_test)
    ds_test_malignant  = SkinCancerDataset(test_malign_dir,  transform=transform_test)
    
    # combine benign and malignant datasets
    full_train_dataset = ConcatDataset([ds_train_benign, ds_train_malignant])
    test_dataset  = ConcatDataset([ds_test_benign, ds_test_malignant])
    
    # dataset split size setup
    dataset_size = len(full_train_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size

    # seed for reproducibility   
    generator = torch.Generator().manual_seed(seed)
    
    # train and val split
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=generator
    )
    
    # train dataloader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers
    )
    
    # val dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,    
        num_workers=num_workers
    )
    
    # test dataloader
    test_loader  = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def create_autoencoder_dataloaders(
    base_dir='data',
    transform_train=None,
    transform_test=None,
    batch_size=32,
    num_workers=2,     
):
    # training dirs
    train_benign_dir = os.path.join(base_dir, 'train', 'Benign')
    train_malign_dir = os.path.join(base_dir, 'train', 'Malignant')
    
    # testing dirs
    test_benign_dir  = os.path.join(base_dir, 'test', 'Benign')
    test_malign_dir  = os.path.join(base_dir, 'test', 'Malignant')

    # instantiate train datasets
    ds_train_benign = SkinCancerDataset(train_benign_dir, transform=transform_train)
    ds_train_malignant = SkinCancerDataset(train_malign_dir, transform=transform_train)
    
    # instantiate test datasets
    ds_test_benign  = SkinCancerDataset(test_benign_dir,  transform=transform_test)
    ds_test_malignant  = SkinCancerDataset(test_malign_dir,  transform=transform_test)

     # combine benign and malignant datasets
    full_train_dataset = ConcatDataset([ds_train_benign, ds_train_malignant])
    test_dataset  = ConcatDataset([ds_test_benign, ds_test_malignant])

    # train dataloader
    train_loader = DataLoader(
        full_train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers
    )

    # test dataloader
    test_loader  = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers
    )

    return train_loader, test_loader