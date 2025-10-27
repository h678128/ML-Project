import torch
from torchvision import transforms, models
from PIL import Image
import gradio as gr
import os

# --- Device og modell ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/age_model.pth"

# --- Lag ResNet18 som i treningen ---
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.fc = torch.nn.Linear(model.fc.in_features, 1)  # output = alder
model = model.to(device)

# --- Last inn lagret modell ---
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print("flott Loaded trained model")
else:
    raise FileNotFoundError(f"Trained model not found: {model_path}")

model.eval()

# --- Transform for bilder ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# --- Funksjon for prediksjon ---
def predict_age(img):
    img = img.convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        predicted_age = output.item()
    return f"{predicted_age:.1f} years"

# --- Gradio Interface ---
iface = gr.Interface(
    fn=predict_age,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Age Predictor",
    description="Upload a face image and the AI will predict the age."
)

# --- Start nettsiden ---
iface.launch()
