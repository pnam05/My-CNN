import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Compose
from dataset import MyAnimalDataset


class MySimpleNetwork(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(in_features=3072, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x


class MyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = self.create_block(in_channels=3, out_channels=16)
        self.conv2 = self.create_block(in_channels=16, out_channels=32)
        self.conv3 = self.create_block(in_channels=32, out_channels=64)
        self.conv4 = self.create_block(in_channels=64, out_channels=128)
        self.conv5 = self.create_block(in_channels=128, out_channels=128)

        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=6272, out_features=1024),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU()
        )

        self.fc3 = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(x.shape[0], -1)  # ~ reshape in numpy

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

    def create_block(self, in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                      padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                      padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )


if __name__ == '__main__':

    transform = Compose([
        ToTensor(),
        Resize((224, 224))
    ])
    dataset = MyAnimalDataset(root="./data/Animal", transform=transform)
    model = MyCNN()

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    for images, labels in dataloader:
        images = images.float()
        output = model(images)
        print(output)
        break