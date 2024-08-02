import torch
from torch import nn, save, load
from torchvision import models, transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
import matplotlib.pyplot as plt
import numpy as np

from ImageClassifier.Segmentation.SegmentationDataset import SegmentationDataset

# Config
num_classes = 2  # Foreground and background
img_path = "data/images"
mask_path = "data/masks"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, loader, loss_fn, opt, num_epochs=25):
    print("Training started")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device).long().squeeze(1)

            opt.zero_grad()
            outputs = model(images)['out']  # Extract the output from the model dictionary
            loss = loss_fn(outputs, masks)
            loss.backward()
            opt.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    return model


def render_predictions(model, loader):
    model.eval()
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device).long().squeeze(1)
        with torch.no_grad():
            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1)

        # Convert tensors to numpy arrays for visualization
        images_np = images.cpu().numpy().transpose(0, 2, 3, 1)
        masks_np = masks.cpu().numpy()
        preds_np = preds.cpu().numpy()

        # Normalize the predictions
        images_np = np.clip(images_np, 0, 1)

        # Plot the images, ground truth masks, and predictions
        for i in range(images_np.shape[0]):
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(images_np[i])
            ax[0].set_title("Image")
            ax[1].imshow(masks_np[i], cmap="gray")
            ax[1].set_title("Ground Truth")
            ax[2].imshow(preds_np[i], cmap="gray")
            ax[2].set_title("Prediction")
            plt.show()


# Define transformations
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
mask_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x > 0.5).long())  # Binary mask
])

# Define model
weights = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
model = models.segmentation.deeplabv3_resnet101(weights=weights)
model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1))
model = model.to(device)

# Define model parameters
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
dataset = SegmentationDataset(image_dir=img_path, mask_dir=mask_path, image_transform=image_transform, mask_transform=mask_transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Verify input dimensions
for images, masks in dataloader:
    print(f"Image batch shape: {images.shape}")
    print(f"Mask batch shape: {masks.shape}")
    break

# Train model
model = train_model(model, dataloader, criterion, optimizer, num_epochs=int(input("Enter training epochs: ")))

# Save model
with open("model.pt", "wb") as f:
    save(model.state_dict(), f)

# Render predictions
render_predictions(model, dataloader)
