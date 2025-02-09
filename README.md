# skincancer_classification

</div>
</figure>
<figure>
<div align="center">
<img src='media/teaser.png' width="80%">
<figcaption>
    <div>
        Classification Results
    </div>
</figcaption>
</div>

## Quick Links

- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
    - [Baseline CNN](#baseline-cnn)
    - [MobileNetV2](#mobilenetv2)
    - [Custom Encoder + Classifier](#custom-encoder--classifier)
  - [Results](#results)
    - [Baseline CNN Results](#baseline-cnn-results)
    - [MobileNetV2 Results](#mobilenetv2-results)
    - [Custom Encoder + Classifier Results](#custom-encoder--classifier-results)
---

## Installation

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/hrishikesh-st/skincancer_classification.git
   cd skincancer_classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
---

## Usage

### Training

Three models were trained:
1. Basline CNN
2. MobileNetV2
3. Custom Encoder + Classifier

#### Basline CNN

Train the baseline CNN model:
```bash
usage: baseline_MNetV2_train.py [-h] [--base_dir BASE_DIR] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--lr LR] [--lr_decay LR_DECAY] [--val_split VAL_SPLIT] [--seed SEED] [--device DEVICE] [--model_name {baseline,mobilenetv2}] [--freeze_backbone]

Train a Skin Cancer Classifier with Val Split

options:
  -h, --help            show this help message and exit
  --base_dir BASE_DIR
  --epochs EPOCHS
  --batch_size BATCH_SIZE
  --lr LR
  --lr_decay LR_DECAY
  --val_split VAL_SPLIT
                        Fraction of training data used for validation
  --seed SEED           Random seed for splitting
  --device DEVICE       Choose device: cuda or cpu
  --model_name {baseline,mobilenetv2}
                        Select which model architecture to train
  --freeze_backbone     Freeze the feature extraction layers in MobileNetV2
```

Example:
```bash
python baseline_MNetV2_train.py --base_dir data --epochs 21 --batch_size 64 --lr 1e-4 --val_split 0.2 --model_name baseline --device cuda
```

#### MobileNetV2

Train the MobileNetV2 model:
```bash
usage: baseline_MNetV2_train.py [-h] [--base_dir BASE_DIR] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--lr LR] [--lr_decay LR_DECAY] [--val_split VAL_SPLIT] [--seed SEED] [--device DEVICE] [--model_name {baseline,mobilenetv2}] [--freeze_backbone]

Train a Skin Cancer Classifier with Val Split

options:
  -h, --help            show this help message and exit
  --base_dir BASE_DIR
  --epochs EPOCHS
  --batch_size BATCH_SIZE
  --lr LR
  --lr_decay LR_DECAY
  --val_split VAL_SPLIT
                        Fraction of training data used for validation
  --seed SEED           Random seed for splitting
  --device DEVICE       Choose device: cuda or cpu
  --model_name {baseline,mobilenetv2}
                        Select which model architecture to train
  --freeze_backbone     Freeze the feature extraction layers in MobileNetV2
```

Example:
```bash
python baseline_MNetV2_train.py --base_dir data --epochs 21 --batch_size 64 --lr 1e-4 --val_split 0.2 --model_name mobilenetv2 --device cuda
```

#### Custom Encoder + Classifier

First train the autoencoder:
```bash
usage: autoencoder_train.py [-h] [--data_dir DATA_DIR] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--lr LR] [--device {cuda,cpu}]

Train a U-Net MobileNet Autoencoder

options:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Path to dataset root directory
  --epochs EPOCHS       Number of training epochs
  --batch_size BATCH_SIZE
                        Batch size for training
  --lr LR               Learning rate
  --device {cuda,cpu}   Device to use
```

Example:
```bash
python autoencoder_train.py --data_dir data --epochs 30 --batch_size 64 --lr 1e-4 --device cuda
```

Then train the custom encoder + classifier:
```bash
usage: encoder_classifier_train.py [-h] [--base_dir BASE_DIR] [--epochs1 EPOCHS1] [--epochs2 EPOCHS2] [--batch_size BATCH_SIZE] [--lr1 LR1] [--lr2 LR2] [--lr_decay LR_DECAY] [--val_split VAL_SPLIT] [--seed SEED] [--device DEVICE] [--model_name MODEL_NAME]

Train a Skin Cancer Classifier with Val Split

options:
  -h, --help            show this help message and exit
  --base_dir BASE_DIR
  --epochs1 EPOCHS1
  --epochs2 EPOCHS2
  --batch_size BATCH_SIZE
  --lr1 LR1
  --lr2 LR2
  --lr_decay LR_DECAY
  --val_split VAL_SPLIT
                        Fraction of training data used for validation
  --seed SEED           Random seed for splitting
  --device DEVICE       Choose device: cuda or cpu
  --model_name MODEL_NAME
```

Example:
```bash
python ensemble_train_final.py --base_dir data --epochs1 25 --epochs2 40 --batch_size 64 --lr1 1e-3 --lr2 5e-5 --lr_decay 0.9 --val_split 0.2 --lr_decay 0.75 --model_name encoder_classifier --device cuda
```
#### Real-time Camera Feed

For real-time camera feed, you can follow the same steps as above. Make sure to subscribe to the correct image topics based on your camera you are using.


### Results

The classification results for the three models are as follows:
 
#### Baseline CNN Results
<figure>
    <div align="center">
        <img src='media/baseline/baseline.png'>
        <figcaption>
            <div>
                Baseline CNN Results
            </div>
        </figcaption>
    </div>
</figure>

TestSet Performance:

| Metric                      |  Value   |
|-----------------------------|----------|
| Accuracy                    | 80.95 %  |
| Precision                   | 0.9317   |
| Recall                      | 0.6680   |
| F1 Score                    | 0.7781   |



#### MobileNetV2 Results
<figure>
    <div align="center">
        <img src='media/mobilenetv2/mobilenetv2.png'>
        <figcaption>
            <div>
                MobileNetV2 Results
            </div>
        </figcaption>
    </div>
</figure>

TestSet Performance:

| Metric                      |  Value   |
|-----------------------------|----------|
| Accuracy                    | 90.10 %  |
| Precision                   | 0.9465   |
| Recall                      | 0.8500   |
| F1 Score                    | 0.8957   |

#### Custom Encoder + Classifier Results

##### Encoder Results
<figure>
    <div align="left">
        <img src='media/autoencoder/loss_plot.png' width=35%>
        <figcaption>
            <div>
                Custom Encoder Results
            </div>
        </figcaption>
    </div>
</figure>

##### Classifier Results
<figure>
    <div align="center">
        <img src='media/encoder_classifier/encoder_classifier.png'>
        <figcaption>
            <div>
                Custom Encoder + Classifier Results
            </div>
        </figcaption>
    </div>
</figure>

TestSet Performance:

| Metric                      |  Value   |
|-----------------------------|----------|
| Accuracy                    | 91.50 %  |
| Precision                   | 0.9296   |
| Recall                      | 0.8980   |
| F1 Score                    | 0.9135   |