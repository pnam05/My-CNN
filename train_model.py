import os
from dataset import MyAnimalDataset
from model import MyCNN
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Compose
from tqdm.autonotebook import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import argparse
import shutil
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, Resize, Compose, RandomHorizontalFlip, RandomRotation, ColorJitter, Normalize


def plot_confusion_matrix(writer, cm, class_names, epoch):
    figure = plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap="PRGn")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)

def get_args():
    parser = argparse.ArgumentParser(description="Train CNN for animal classification")
    parser.add_argument("-d", "--data-path", type=str, default="./data/Animal")
    parser.add_argument("-l", "--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("-m", "--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument("-n", "--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("-b", "--batch-size", type=int, default=16)
    parser.add_argument("-t", "--tensorboard", type=str, default="my_tensorboard")
    parser.add_argument("-r", "--resume", type=bool, default=False)
    args = parser.parse_args()

    return args

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = Compose([
        Resize((224, 224)),
        RandomHorizontalFlip(p=0.5),
        RandomRotation(15),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])

    train_dataset = MyAnimalDataset(root=args.data_path, split="train", transform=train_transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    val_dataset = MyAnimalDataset(root=args.data_path, split="valid", transform=val_test_transform)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    model = MyCNN(num_classes=len(train_dataset.categories))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    if args.resume:
        checkpoint = torch.load("weights/last.pt", map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint["best_acc"]
    else:
        start_epoch = 0
        best_acc = -1

    num_iter_per_epoch = len(train_dataloader)
    if os.path.isdir(args.tensorboard):
        shutil.rmtree(args.tensorboard)
    os.makedirs(args.tensorboard)
    writer = SummaryWriter(args.tensorboard)

    os.makedirs("weights", exist_ok=True)

    for epoch in range(start_epoch, args.num_epochs):
        # training
        model.train()
        progress_bar = tqdm(train_dataloader, colour="cyan")
        loss_ls = []
        for iter, (images, labels) in enumerate(progress_bar):
            # Forward
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            loss = criterion(predictions, labels)
            loss_ls.append(loss.item())
            avg_loss = np.mean(loss_ls)
            progress_bar.set_description("Epoch: {}/{}. Loss: {:.4f}".format(epoch+1, args.num_epochs, avg_loss))
            writer.add_scalar("Train/loss", avg_loss, global_step=epoch*num_iter_per_epoch+iter)
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        loss_ls = []
        label_ls = []
        prediction_ls = []
        progress_bar = tqdm(val_dataloader, colour="cyan")
        with torch.no_grad():
            for iter, (images, labels) in enumerate(progress_bar):
                # Forward
                images = images.to(device)
                labels = labels.to(device)

                predictions = model(images)
                loss = criterion(predictions, labels)
                loss_ls.append(loss.item())
                predicted_classes = torch.argmax(predictions, dim=1)
                label_ls.extend(labels.cpu())
                prediction_ls.extend(predicted_classes.cpu())
        loss_avg = np.mean(loss_ls)
        acc = accuracy_score(label_ls, prediction_ls)
        print("Validation. Epoch: {}/{}. Average loss: {}. Accuracy: {}".format(epoch+1, args.num_epochs, loss_avg, acc))
        writer.add_scalar("Val/loss", loss_avg, global_step=epoch)
        writer.add_scalar("Val/accuracy", acc, global_step=epoch)
        plot_confusion_matrix(writer, confusion_matrix(label_ls, prediction_ls), train_dataset.categories, epoch)

        # save checkpoint
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc
        }
        torch.save(checkpoint, "weights/last.pt")

        if acc > best_acc:
            torch.save(checkpoint, "weights/best.pt")
            best_acc = acc

if __name__ == '__main__':
    args = get_args()
    train(args)