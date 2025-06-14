import torch
from torchvision import models, transforms
from PIL import Image
import argparse
import os
import json

# Cargar clases
with open("model/idx_to_class.json", "r") as f:
    idx_to_class = json.load(f)

# Cargar recetas
with open("data/recipes.json", "r", encoding="utf-8") as f:
    recipes = json.load(f)

# Transformación de imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(image_path, model, idx_to_class, device):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    
    predicted_class = idx_to_class[str(predicted.item())]
    return predicted_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Ruta de la imagen a clasificar")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar modelo
    model = models.resnet18(weights=None)
    num_classes = len(idx_to_class)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("model/classifier.pth", map_location=device))
    model.to(device)

    # Realizar predicción
    predicted_class = predict(args.image_path, model, idx_to_class, device)
    print(f"\n🍽️ Alimento detectado: {predicted_class.replace('_', ' ').title()}")

    # Mostrar receta
    recipe = recipes.get(predicted_class)
    if recipe:
        print("\n📋 Ingredientes:")
        for ing in recipe['ingredients']:
            print(f"  - {ing.strip()}")

        print("\n📝 Instrucciones:")
        for i, step in enumerate(recipe['instructions'], 1):
            print(f"  {i}. {step.strip()}")
    else:
        print("\n❌ Receta no disponible para este platillo.")
