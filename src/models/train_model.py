import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from src.data.dataset import AgeDataset
import os

# --- Settings ---
train_csv = "data/splits/train.csv"
val_csv = "data/splits/val.csv"
img_dir = "data/raw/crop_part1"  
batch_size = 32
epochs = 10
lr = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Transforms ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

# --- Datasets ---
train_dataset = AgeDataset(train_csv, img_dir, transform)
val_dataset = AgeDataset(val_csv, img_dir, transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# --- Model (bruk pre-trained resnet18) ---
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1)  # output = alder
model = model.to(device)

# --- Load saved model hvis den er der
if os.path.exists("models/age_model.pth"):
    model.load_state_dict(torch.load("models/age_model.pth"))
    print("great Loaded existing model weights to continue training")


# --- Loss & optimizer ---
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# --- Training loop ---
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # [batch,1]

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        mae = F.l1_loss(outputs, labels)
        running_mae += mae.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, MAE: {running_mae/len(train_loader):.4f}")

    # --- Save model ---
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/age_model.pth")
print("flottings Model saved to models/age_model.pth")

