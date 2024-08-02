# Import dependencies
import torch.cuda
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, ToTensor
from PIL import Image

from classification_dataset import ClassificationData
from segmentor import segment_notes
from note_player import play_frequency

# train = datasets.MNIST(root="segmentation", download=True, train=True, transform=ToTensor())
labels = [2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0]  # 0 -> Quarter, 1 -> Half, 2 -> Full
train = ClassificationData("data/classification", labels, transform=ToTensor())
dataset = DataLoader(train, batch_size=32, shuffle=True)


# Image classifier nn
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 22 * 22, 10)
        )

    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        return self.model(x)


# Instance nn, loss, optimizer
device = ("cuda" if torch.cuda.is_available() else "cpu")
clf = ImageClassifier().to(device)
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()


# Training flow
def start_train():
    for epoch in range(int(input("Enter training epochs: "))):
        for batch in dataset:
            x, y = batch
            x, y = x.to(device), y.to(device)
            yhat = clf(x)
            loss = loss_fn(yhat, y)

            # Apply backprop
            opt.zero_grad()
            loss.backward()
            opt.step()

        # noinspection PyUnboundLocalVariable
        print(f"Epoch: {epoch}, loss: {loss.item()}")

    with open("model_state.pt", "wb") as f:
        save(clf.state_dict(), f)


def load_model_state(model, path):
    model.load_state_dict(load(path, weights_only=False))
    model.eval()


def translate_label(label):
    return ["Quarter", "Half", "Full"][label]


def run(image_path=None):
    img_path = image_path or input("Enter image path: ")
    img = Image.open(img_path)
    run_i(img)


def run_i(img):
    # Load model state
    model_path = "model_state.pt"
    load_model_state(clf, model_path)

    img = img.convert("L")
    img = Resize((28, 28))(img)
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)
    logits = clf(img_tensor)
    pred_prob = nn.Softmax(dim=1)(logits)
    y_pred = pred_prob.argmax(1)

    pred = translate_label(y_pred.item())
    print(f"Predicted: {pred}")
    return y_pred.item()


if __name__ == "__main__":
    inp = input("train or test or eval path? ").lower()
    if inp == "train":
        start_train()
    elif inp == "test":
        run()
    else:
        notes = segment_notes(inp, True)
        for n in notes:
            duration = 1 / [4, 2, 1][run_i(n)]
            play_frequency(440, duration)
