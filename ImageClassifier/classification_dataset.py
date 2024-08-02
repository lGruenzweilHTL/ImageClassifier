import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from PIL import Image
import os


class ClassificationData(Dataset):
    def __init__(self, image_dir, labels, transform=None):
        self.image_dir = image_dir
        self.labels = labels
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

        assert len(self.image_files) == len(self.labels), "The number of images and labels must be the same."

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("L")
        image = Resize((28, 28))(image)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label)  # Convert label to tensor
        return image, label

# Example usage
# image_dir = 'path/to/your/images'
# labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Replace with actual labels
# custom_dataset = CustomDataset(image_dir, transform=ToTensor())
# data_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True)
