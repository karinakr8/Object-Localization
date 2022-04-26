# Antrasis GMM laboratorinis darbas

# Atliko:    Karina Krapaitė, 5 gr.
# LSP:       1911065
# Variantas: 6

import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import os
from data import ImageDataset
from model import BB_model
import training
import inference

BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.0001
# 56 %
NUM_WORKERS = 4

TRAIN_ANNOTATIONS_FILE = "E:/GMM datasets/dataset/OpenImages/OID/csv_folder/train-annotations-clean.csv"
TRAIN_IMAGE_DIR = "E:/GMM datasets/dataset/OpenImages/OID/Dataset/train/images"
VAL_ANNOTATIONS_FILE = "E:/GMM datasets/dataset/OpenImages/OID/csv_folder/validate-annotations-clean.csv"
VAL_IMAGE_DIR = "E:/GMM datasets/dataset/OpenImages/OID/Dataset/validate/images"

TEST_ANNOTATIONS_FILE = "E:/GMM datasets/dataset/test-data/csv_folder/test-annotations.csv"
TEST_IMAGE_DIR = "E:/GMM datasets/dataset/test-data/images"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# transformavimo funckijos

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    # transforms.Scale((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(  # norm
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    transforms.Normalize(  # inv_norm
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
])

train_dataset = ImageDataset(TRAIN_ANNOTATIONS_FILE, TRAIN_IMAGE_DIR, transform)
val_dataset = ImageDataset(VAL_ANNOTATIONS_FILE, VAL_IMAGE_DIR, transform)
test_dataset = ImageDataset(TEST_ANNOTATIONS_FILE, TEST_IMAGE_DIR, transform)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=3, shuffle=False)


def create_model():
    model = BB_model().to(device)

    loss_bb = nn.MSELoss()
    loss_class = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE,
                                                    steps_per_epoch=int(len(train_dataloader)),
                                                    epochs=EPOCHS,
                                                    anneal_strategy='linear')

    training.train(model, train_dataloader, loss_bb, loss_class, optimizer, device, EPOCHS, scheduler)

    torch.save(model.state_dict(), "E:/GMM datasets/dataset/model.pth")
    print("Trained model saved at E:/GMM datasets/dataset/model.pth")


def test_model(dataloader):
    model = BB_model().to(device)
    state_dict = torch.load("E:/GMM datasets/dataset/model.pth", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    inference.inference(model, dataloader, device)


def main():
    print(f"Using device: {device}.")
    print(f'Training dataset:   {len(train_dataset)}')
    print(f'Validating dataset: {len(val_dataset)}')
    print(f'Testing dataset:    {len(test_dataset)}')

    # create_model()
    # test_model(val_dataloader)
    test_model(test_dataloader)


if __name__ == '__main__':
    main()
