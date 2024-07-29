# Import dependencies
from torch import nn, save
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, 32)


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
            nn.Linear(64*(28-6)*(28-6), 10)
        )

    def forward(self, x):
        return self.model(x)


# Instance nn, loss, optimizer
run_type = "cpu"  # 'cpu' -> run on cpu; 'cuda' -> run on gpu using cuda
clf = ImageClassifier().to(run_type)
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training flow
if __name__ == "__main__":
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
