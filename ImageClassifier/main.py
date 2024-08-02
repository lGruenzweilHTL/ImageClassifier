# Import dependencies
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, ToTensor
from PIL import Image

from ImageClassifier.data import CustomDataset

# train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
labels = [2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0]  # 0 -> Quarter, 1 -> Half, 2 -> Full
train = CustomDataset("data/custom", labels, transform=ToTensor())
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
run_type = "cpu"  # 'cpu' -> run on cpu; 'cuda' -> run on gpu using cuda
clf = ImageClassifier().to(run_type)
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()


# Training flow
def start_train():
    for epoch in range(int(input("Enter training epochs: "))):
        for batch in dataset:
            x, y = batch
            x, y = x.to(run_type), y.to(run_type)
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


def run():
    # Load model state
    model_path = "model_state.pt"
    load_model_state(clf, model_path)

    # Evaluate image
    img_path = input("Enter image path: ")
    img = Image.open(img_path).convert("L")
    img = Resize((28, 28))(img)
    img_tensor = ToTensor()(img).unsqueeze(0).to(run_type)
    logits = clf(img_tensor)
    pred_prob = nn.Softmax(dim=1)(logits)
    y_pred = pred_prob.argmax(1)

    pred = translate_label(y_pred.item())
    print(f"Predicted: {pred}")


if __name__ == "__main__":
    inp = input("train or test? ").lower()
    if inp == "train":
        start_train()
    elif inp == "test":
        run()
