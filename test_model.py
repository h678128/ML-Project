import torch
from torchvision import transforms, models
from PIL import Image
import argparse
import os

# --- Settings ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/age_model.pth"

# --- Argument parser for bilde ---
parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, required=True, help="Path to image file")  #endre i terminal til hvilke som helst bilde
args = parser.parse_args()

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# --- Load image ---
if not os.path.exists(args.image):
    raise FileNotFoundError(f"Image not found: {args.image}")

img = Image.open(args.image).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

# --- Load model ---
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.fc = torch.nn.Linear(model.fc.in_features, 1)
model = model.to(device)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print("flott Loaded trained model")
else:
    raise FileNotFoundError(f"Trained model not found: {model_path}")

model.eval()

# --- Predict ---
with torch.no_grad():
    output = model(img_tensor)
    predicted_age = output.item()
    print(f"Predicted age: {predicted_age:.1f} years")
