from pathlib import Path
import torch
from torchvision import models, transforms
from PIL import Image


CLASS_NAMES = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "../models/vision/model_resnet_animals.pth"

# load model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def classify_animal(image_path: str):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        pred = model(img_tensor).argmax(1).item()
    return CLASS_NAMES[pred]

if __name__ == "__main__":
    print(classify_animal("test_images/dog/0003.jpeg"))
