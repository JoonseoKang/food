import glob
import os.path as osp
import random
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms


class ImageTransform:
    def __init__(self, resize, mean, std):
        self.data_transform = {
            "train": transforms.Compose(
                [
                    transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize(resize),
                    transforms.CenterCrop(resize),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
        }

    def __call__(self, img, phase="train"):
        return self.data_transform[phase](img)


def make_datapath_list(phase="train"):
    if phase == "train":
        target_path = osp.join("./**/*.jpg")
    else:
        target_path = osp.join("./**/**/*.jpg")

    print(target_path)
    path_list = [path for path in glob.glob(target_path)]

    return path_list


class CustomDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase="train"):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)

        img_transformed = self.transform(img, self.phase)

        label = osp.basename(osp.dirname(img_path))
        label = 0 if label == "italian" else 1

        return img_transformed, label


def create_data_loader(dataset, batch_size, shuffle):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def initialize_model(num_classes):
    use_pretrained = True
    net = models.vgg16(pretrained=use_pretrained)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)
    return net


def set_parameter_requires_grad(model, update_param_names, requires_grad=True):
    for name, param in model.named_parameters():
        if any(name.startswith(update_name) for update_name in update_param_names):
            param.requires_grad = requires_grad


def train(net, dataloaders, criterion, optimizer, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-------------")

        for phase in ["train", "val"]:
            is_training = phase == "train"
            if is_training:
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            if epoch == 0 and is_training:
                continue

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(is_training):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if is_training:
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloaders[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")


def main():
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    batch_size = 32
    num_epochs = 5

    transform = ImageTransform(size, mean, std)
    train_list = make_datapath_list(phase="train")
    val_list = make_datapath_list(phase="val")

    train_dataset = CustomDataset(file_list=train_list, transform=transform, phase="train")
    val_dataset = CustomDataset(file_list=val_list, transform=transform, phase="val")

    train_dataloader = create_data_loader(train_dataset, batch_size, shuffle=True)
    val_dataloader = create_data_loader(val_dataset, batch_size, shuffle=False)

    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    num_classes = 2
    net = initialize_model(num_classes)

    set_parameter_requires_grad(net, ["features"])
    set_parameter_requires_grad(
        net, ["classifier.0.weight", "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"]
    )
    set_parameter_requires_grad(net, ["classifier.6.weight", "classifier.6.bias"], requires_grad=True)

    optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    train(net, dataloaders_dict, criterion, optimizer, num_epochs)


if __name__ == "__main__":
    main()
