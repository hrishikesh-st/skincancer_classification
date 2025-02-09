import torchvision.transforms as T

def get_train_transforms(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor()
    ])

def get_test_transforms(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor()
    ])

def get_autoencoder_transforms(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor()
    ])