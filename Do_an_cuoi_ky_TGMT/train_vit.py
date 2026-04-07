import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from transformers import ViTForImageClassification, ViTConfig
from tqdm import tqdm
import json

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

    # ViT configuration (train from scratch)
    config = ViTConfig(
        image_size=224,
        patch_size=16,
        num_channels=3,
        num_labels=num_classes,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048
    )

    model = ViTForImageClassification(config)

    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    epochs = 20
    best_acc = 0

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

            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

        train_acc = 100*correct/total

        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images,labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images).logits

                _,predicted = torch.max(outputs,1)

                val_correct += (predicted==labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100*val_correct/val_total

        print("Train acc:",train_acc)
        print("Val acc:",val_acc)

        if val_acc > best_acc:

            best_acc = val_acc

            torch.save(model.state_dict(),"vit_animals_best.pth")

            print("Best model saved")

    print("Training complete")


if __name__ == "__main__":
    main()
