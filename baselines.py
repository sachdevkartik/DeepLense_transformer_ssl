import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the contrastive loss function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2)
            + (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive


# Set hyperparameters
batch_size = 32
epochs = 10
learning_rate = 0.001
margin = 1.0

# Load dataset
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)
train_dataset = datasets.ImageFolder("path_to_train_dataset", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Load pre-trained ResNet18 model
model = resnet18(pretrained=True)
model.fc = nn.Identity()  # Remove the fully connected layer
model.to(device)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = ContrastiveLoss(margin)

# Training loop
for epoch in range(epochs):
    for batch_idx, (img1, img2, label) in enumerate(train_loader):
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)

        optimizer.zero_grad()
        output1 = model(img1)
        output2 = model(img2)
        loss = criterion(output1, output2, label)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item()}")

# Save the trained model
torch.save(model.state_dict(), "contrastive_model.pth")
