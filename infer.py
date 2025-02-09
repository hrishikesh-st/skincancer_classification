import os
import torch
import torchvision.transforms as transforms
import argparse
from PIL import Image, ImageDraw, ImageFont
from custom_models.baseline import Baseline
from custom_models.mobilenetv2 import SkinCancerMobileNetV2
from custom_models.encoder_classifier import EncoderClassifier
from utils.logger import get_logger  

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a folder of test images")
    parser.add_argument('--test_dir', type=str, required=True, help="Path to test dataset directory")
    parser.add_argument('--model_path', type=str, required=True, help="Path to trained model (.pth)")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to save annotated images")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help="Device to use for inference")
    parser.add_argument('--model_name', type=str, required=True, choices=['baseline', 'mobilenetv2', 'encoder_classifier'], 
                        help="Select the model architecture")
    return parser.parse_args()

def load_model(model_path, model_name, device):
    """
    Load the trained model for inference based on the architecture.
    Assumes all models are binary classifiers using BCEWithLogitsLoss.
    """
    if model_name == "baseline":
        model = Baseline()
    elif model_name == "mobilenetv2":
        model = SkinCancerMobileNetV2(pretrained=False)
    elif model_name == "encoder_classifier":
        model = EncoderClassifier()  
    
    # load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model

def visualize_predictions(image_path, output_dir, ground_truth, prediction):
    """
    Add black padding to the image and annotates ground truth and prediction labels.
    """
    image = Image.open(image_path).convert("RGB")
    img_width, img_height = image.size
    padding_size = 50
    new_width, new_height = img_width + 2 * padding_size, img_height + 2 * padding_size

    padded_image = Image.new("RGB", (new_width, new_height), "black")
    padded_image.paste(image, (padding_size, padding_size))

    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(padded_image)
    text_color = (255, 255, 255)  
    draw.text((10, 10), f"GT: {ground_truth}", fill=text_color, font=font)  
    draw.text((10, new_height - 30), f"PRED: {prediction}", fill=text_color, font=font)  

    # unique filename 
    img_name = os.path.basename(image_path)  
    output_filename = f"{ground_truth}_{img_name}"
    output_path = os.path.join(output_dir, output_filename)
    
    padded_image.save(output_path)

def run_inference(test_dir, model_path, model_name, output_dir, device, logger):
    """
    Runs inference on the test dataset and saves the annotated images.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Loading model from: {model_path}")

    model = load_model(model_path, model_name, device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    logger.info(f"Running inference on test dataset: {test_dir}")

    for class_folder in ["Benign", "Malignant"]:
        class_path = os.path.join(test_dir, class_folder)
        if not os.path.exists(class_path):
            logger.warning(f"Skipping missing folder: {class_path}")
            continue

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            image = Image.open(img_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor).squeeze(1)  
                pred_prob = torch.sigmoid(output).item()  
                pred_label = pred_prob > 0.5  

            predicted_class = "Malignant" if pred_label else "Benign"

            visualize_predictions(img_path, output_dir, class_folder, predicted_class)
            logger.info(f"Processed: {img_name} | GT: {class_folder} | PRED: {predicted_class} | Prob: {pred_prob:.4f}")

def main():
    args = parse_args()
    log_file = "inference.log"
    logger = get_logger(args.output_dir, log_file)
    logger.info(f"Starting inference with args: {args}")

    run_inference(args.test_dir, args.model_path, args.model_name, args.output_dir, args.device, logger)

if __name__ == "__main__":
    main()
