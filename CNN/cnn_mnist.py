import os
import struct
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

os.makedirs("output", exist_ok=True)

# 1. FUNCTIONS TO LOAD MNIST DATA

def load_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        num_images = struct.unpack('>I', f.read(4))[0]
        rows = struct.unpack('>I', f.read(4))[0]
        cols = struct.unpack('>I', f.read(4))[0]

        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)

    return images

def load_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        num_labels = struct.unpack('>I', f.read(4))[0]

        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels

# 2. LOAD TRAIN AND TEST DATA

train_images = load_mnist_images("dataset/train-images.idx3-ubyte")
train_labels = load_mnist_labels("dataset/train-labels.idx1-ubyte")

test_images = load_mnist_images("dataset/t10k-images.idx3-ubyte")
test_labels = load_mnist_labels("dataset/t10k-labels.idx1-ubyte")


# 3. PREPROCESS DATA

train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

train_images = np.expand_dims(train_images, axis=1)
test_images = np.expand_dims(test_images, axis=1)

X_train = torch.tensor(train_images, dtype=torch.float32)
y_train = torch.tensor(train_labels, dtype=torch.long)

X_test = torch.tensor(test_images, dtype=torch.float32)
y_test = torch.tensor(test_labels, dtype=torch.long)

# 4. CREATE DATA LOADERS

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 5. DEFINE BASIC CNN MODEL

class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        return x


# 6. SET DEVICE, MODEL, LOSS, OPTIMIZER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = BasicCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 7. TRAIN MODEL

epochs = 5
train_losses = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")


# 8. TEST ACCURACY

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\nTest Accuracy: {accuracy:.2f}%")


# 9. PLOT LOSS CONVERGENCE

plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), train_losses, marker='o')
plt.title("Training Loss Convergence")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("output/loss_convergence.png", dpi=300, bbox_inches="tight")
plt.show()

# 10. VISUALIZE FIRST CONV FILTERS


filters = model.conv1.weight.data.cpu().numpy()

plt.figure(figsize=(10, 5))
for i in range(filters.shape[0]):
    plt.subplot(2, 4, i + 1)
    plt.imshow(filters[i, 0], cmap='gray')
    plt.title(f"Filter {i+1}")
    plt.axis('off')

plt.tight_layout()
plt.savefig("output/cnn_filters.png", dpi=300, bbox_inches="tight")
plt.show()


# 11. SHOW SOME TEST PREDICTIONS

images, labels = next(iter(test_loader))
images = images.to(device)
labels = labels.to(device)

with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

images = images.cpu().numpy()
labels = labels.cpu().numpy()
predicted = predicted.cpu().numpy()

plt.figure(figsize=(10, 4))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(images[i][0], cmap='gray')
    plt.title(f"T:{labels[i]} P:{predicted[i]}")
    plt.axis('off')

plt.tight_layout()
plt.savefig("output/test_predictions.png", dpi=300, bbox_inches="tight")
plt.show()