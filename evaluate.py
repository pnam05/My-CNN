import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import MyAnimalDataset
from model import MyCNN


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_test_transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])

    test_dataset = MyAnimalDataset(root="./data/Animal", split="test", transform=val_test_transform)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4
    )

    # Load model
    model = MyCNN(num_classes=len(test_dataset.categories))
    checkpoint = torch.load("weights/best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    label_ls = []
    prediction_ls = []

    with torch.no_grad():
        for images, labels in tqdm(test_dataloader, colour="green"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            label_ls.extend(labels.cpu())
            prediction_ls.extend(preds.cpu())

    acc = accuracy_score(label_ls, prediction_ls)
    cm = confusion_matrix(label_ls, prediction_ls)

    os.makedirs("results", exist_ok=True)

    save_confusion_matrix(
        cm,
        class_names=test_dataset.categories,
        save_path="results/confusion_matrix.png"
    )

    print(f"\n Test Accuracy: {acc:.4f}")


def save_confusion_matrix(cm, class_names, save_path="confusion_matrix.png"):
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    threshold = cm_norm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_norm[i, j] > threshold else "black"
            plt.text(j, i, f"{cm_norm[i, j]:.2f}",
                     horizontalalignment="center",
                     color=color)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    plt.savefig(save_path) 
    plt.close()

if __name__ == "__main__":
    evaluate()