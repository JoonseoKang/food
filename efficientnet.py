import glob
import os.path as osp
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import time
import copy

# Load pretrained models
model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=2)
model2 = EfficientNet.from_pretrained("efficientnet-b7", num_classes=2)


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


size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transform = ImageTransform(size, mean, std)


def make_datapath_list(phase="train"):
    if phase == "train":
        target_path = osp.join("./**/*.jpg")
    else:
        target_path = osp.join("./**/**/*.jpg")

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


batch_size = 64
train_list = make_datapath_list(phase="train")
val_list = make_datapath_list(phase="val")

train_dataset = CustomDataset(file_list=train_list, transform=transform, phase="train")
val_dataset = CustomDataset(file_list=val_list, transform=transform, phase="val")

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

criterion = nn.CrossEntropyLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model2 = model2.to(device)

optimizer_ft = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
lmbda = lambda epoch: 0.98739
exp_lr_scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer_ft, lr_lambda=lmbda)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []

    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss, running_corrects, num_cnt = 0.0, 0, 0

            for inputs, labels in tqdm(dataloaders_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                num_cnt += len(labels)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / num_cnt
            epoch_acc = (running_corrects.double() / num_cnt).cpu() * 100

            if phase == "train":
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)

    time_elapsed = time.time() - since
    return model, train_loss, train_acc, valid_loss, valid_acc


model2, train_loss, train_acc, valid_loss, valid_acc = train_model(
    model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=11
)
