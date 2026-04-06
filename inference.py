import torch
import torch.nn as nn
import argparse
from model import MyCNN
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Compose, Normalize

def get_args():
    parser = argparse.ArgumentParser(description="Inference CNN for animal classification")
    parser.add_argument("-i", "--image-path", type=str, default="test_img/chicken.jpg")
    args = parser.parse_args()
    return args

def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    categories = [
        "butterfly", "cat", "chicken", "cow", "dog", 
        "elephant", "horse", "sheep", "spider", "squirrel"
    ]
    
    model = MyCNN(num_classes=len(categories))

    val_test_transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])

    checkpoint = torch.load("weights/best.pt", map_location=device) 
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    image = Image.open(args.image_path).convert("RGB")
    input_tensor = val_test_transform(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    softmax = nn.Softmax(dim=0) 
    
    with torch.inference_mode(): 
        output = model(input_batch)[0] 
        output = softmax(output)
        print("Xác suất dự đoán:", output)
        
        predicted_class = torch.argmax(output).item() 
        print("Kết quả:", categories[predicted_class])

if __name__ == '__main__':
    args = get_args()
    infer(args)