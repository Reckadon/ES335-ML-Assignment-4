import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models
from torchvision.models import VGG16_Weights

# Define a helper function for denormalizing and displaying images
def imshow(image, mean, std):
    image = image.permute(1, 2, 0).numpy()
    image = std * image + mean  # Denormalize
    image = np.clip(image, 0, 1)
    return image


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(224 * 224 * 3, 256)  # Adjusted input layer
        self.fc2 = nn.Linear(256, 256)           # Adjusted hidden layer 1
        self.fc3 = nn.Linear(256, 128)           # Adjusted hidden layer 2
        self.fc4 = nn.Linear(128, 32)            # Adjusted hidden layer 3
        self.fc5 = nn.Linear(32, 1)               # Output layer remains the same
        self.sigmoid = nn.Sigmoid()                  # Output layer

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.sigmoid(self.fc5(x))
        return x
    
class VGG1Block(nn.Module):
    def __init__(self):
        super(VGG1Block, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Convolutional layer
            nn.ReLU(inplace=True),  # ReLU activation
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 112 * 112, 64),  # Fully connected layer
            nn.ReLU(inplace=True),  # ReLU activation
            nn.Linear(64, 1),  # Output layer (binary classification)
            nn.Sigmoid()  # Sigmoid activation for binary classification
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the features
        x = self.classifier(x)
        return x
    
class VGG3Block(nn.Module):
    def __init__(self):
        super(VGG3Block, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 128),  # Adjust the size based on the output feature map size
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),  # Binary output (0 or 1)
            nn.Sigmoid()  # Sigmoid activation for binary classification
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.classifier(x)
        return x



# Initialize TensorBoard writer
writer = SummaryWriter("runs/VGG16")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset and split into train and test sets
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to (224, 224)
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomRotation(15),  # Random rotation between -15 to 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Random color jitter
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to (224, 224)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_data = datasets.ImageFolder(root='dataset', transform=transform_train)

train_size = int(0.8 * len(train_data))  # 80% for training
test_size = len(train_data) - train_size  # 20% for testing
train_dataset, test_dataset = random_split(train_data, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

final_test = datasets.ImageFolder(root='Test', transform=transform)
final_test_loader = DataLoader(final_test, batch_size=4, shuffle=False)

model = models.vgg16(pretrained=True)

# Modify the final layer to match the binary classification task
model.classifier[6] = nn.Sequential(
    nn.Linear(4096, 32),
    nn.ReLU(inplace=True),
    nn.Linear(32, 1),
    nn.Sigmoid()  # Sigmoid activation for binary classification
)

# Unfreeze all layers by setting requires_grad=True
for param in model.features.parameters():
    param.requires_grad = False  # Freeze Convulation layer
for param in model.classifier.parameters():
    param.requires_grad = True # Unfreeze FC layer

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up the loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training and logging loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.float().to(device)

        # Forward pass
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update running loss and accuracy
        running_loss += loss.item()
        preds = (outputs >= 0.5).float()
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)

        # Log training loss and accuracy per iteration
        writer.add_scalar("Training Loss", loss.item(), epoch * len(train_loader) + i)
        writer.add_scalar("Training Accuracy", correct_train / total_train, epoch * len(train_loader) + i)

    # Log average training loss and accuracy per epoch
    train_accuracy = correct_train / total_train
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.4f}")

    # Evaluate on the test set
    model.eval()
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.float().to(device)

            outputs = model(images).squeeze()
            preds = (outputs >= 0.5).float()
            correct_test += (preds == labels).sum().item()
            total_test += labels.size(0)

        test_accuracy = correct_test / total_test
        writer.add_scalar("Testing Accuracy", test_accuracy, epoch)
        print(f"Test Accuracy: {test_accuracy:.4f}")

    # Log test set images and predictions
    with torch.no_grad():
        images, labels = next(iter(test_loader))
        images, labels = images.to(device), labels.float().to(device)
        outputs = model(images).squeeze()
        preds = (outputs >= 0.5).float()

        fig = plt.figure(figsize=(12, 12))
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        num_images = min(32, len(images))

        for idx in range(num_images):
            ax = fig.add_subplot(4, 8, idx + 1, xticks=[], yticks=[])
            img = imshow(images[idx].cpu(), mean, std)
            ax.imshow(img)
            label = "Positive" if labels[idx].item() == 1 else "Negative"
            prediction = "Positive" if preds[idx].item() == 1 else "Negative"
            ax.set_title(f"L: {label}\nP: {prediction}", color=("green" if label == prediction else "red"))

        # Log the figure to TensorBoard
        writer.add_figure("Test Set Predictions", fig, global_step=epoch)

with torch.no_grad():
    all_images = []
    all_labels = []
    all_preds = []

    for images, labels in final_test_loader:
        images, labels = images.to(device), labels.float().to(device)
        outputs = model(images).squeeze()
        preds = (outputs >= 0.5).float()

        all_images.append(images.cpu())
        all_labels.append(labels.cpu())
        all_preds.append(preds.cpu())

    # Concatenate all images, labels, and predictions
    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)

    # Visualize a few samples
    fig = plt.figure(figsize=(12, 12))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    num_images = min(4, len(all_images))

    for idx in range(num_images):
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
        img = imshow(all_images[idx], mean, std)
        ax.imshow(img)
        label = "Positive" if all_labels[idx].item() == 1 else "Negative"
        prediction = "Positive" if all_preds[idx].item() == 1 else "Negative"
        ax.set_title(f"L: {label}\nP: {prediction}", color=("green" if label == prediction else "red"))

    # Log the figure to TensorBoard
    writer.add_figure("Final Test Set Predictions (All Data)", fig, global_step=epoch)

# Close the TensorBoard writer
writer.close()

# Save the trained model
#torch.save(model.state_dict(), "VGG1_state_dict.pth")

#total number of parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {num_params}")

print("Training completed and model saved.")
