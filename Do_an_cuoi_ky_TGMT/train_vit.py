import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from transformers import ViTForImageClassification
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import numpy as np


def smooth(y, box_pts=5):
    box = np.ones(box_pts)/box_pts
    return np.convolve(y, box, mode='same')


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset_path = "dataset"

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(dataset_path, transform=transform)

    num_classes = len(dataset.classes)

    print("Classes:", dataset.classes)

    with open("classes.json","w") as f:
        json.dump(dataset.classes,f)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset,[train_size,val_size])


    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    torch.backends.cudnn.benchmark = True

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )

    model.to(device)


    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=5e-5
    )


    epochs = 50
    best_acc = 0


    train_losses = []
    val_losses = []

    train_accs = []
    val_accs = []


    for epoch in range(epochs):

        model.train()

        total_loss = 0
        correct = 0
        total = 0


        loop = tqdm(train_loader)

        for images, labels in loop:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images).logits

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()


            total_loss += loss.item()

            _, predicted = torch.max(outputs,1)

            correct += (predicted==labels).sum().item()

            total += labels.size(0)


        train_loss = total_loss/len(train_loader)

        train_acc = 100*correct/total


        model.eval()

        val_loss_total = 0

        val_correct = 0

        val_total = 0


        with torch.no_grad():

            for images,labels in val_loader:

                images = images.to(device)

                labels = labels.to(device)

                outputs = model(images).logits

                loss = criterion(outputs, labels)

                val_loss_total += loss.item()

                _,predicted = torch.max(outputs,1)

                val_correct += (predicted==labels).sum().item()

                val_total += labels.size(0)


        val_loss = val_loss_total/len(val_loader)

        val_acc = 100*val_correct/val_total


        train_losses.append(train_loss)

        val_losses.append(val_loss)

        train_accs.append(train_acc)

        val_accs.append(val_acc)


        print("train loss:",train_loss)

        print("val loss:",val_loss)

        print("train acc:",train_acc)

        print("val acc:",val_acc)


        if val_acc > best_acc:

            best_acc = val_acc

            torch.save(model.state_dict(),"vit_animals_best.pth")

            print("saved best model")


    print("Training complete")


    plt.figure(figsize=(10,8))


    plt.subplot(2,2,1)

    plt.plot(train_losses)

    plt.plot(smooth(train_losses),"--")

    plt.title("train/loss")


    plt.subplot(2,2,2)

    plt.plot(train_accs)

    plt.plot(smooth(train_accs),"--")

    plt.title("accuracy")


    plt.subplot(2,2,3)

    plt.plot(val_losses)

    plt.plot(smooth(val_losses),"--")

    plt.title("val/loss")


    plt.subplot(2,2,4)

    plt.plot(val_accs)

    plt.plot(smooth(val_accs),"--")

    plt.title("val accuracy")


    plt.tight_layout()

    plt.savefig("results.png")

    plt.show()



if __name__ == "__main__":

    main()