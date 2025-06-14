import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import json

# ----- Configuración general -----
st.set_page_config(page_title="Food Identifier", layout="centered")

# Sidebar
with st.sidebar:
    st.markdown("## 🍲 Food Identifier")
    st.write("Identifica platillos y obtén su receta completa al instante.")
    st.markdown("🔎 Sube una imagen o usa tu cámara 📷")
    st.markdown("---")
    st.write("👨‍💻 Proyecto desarrollado con PyTorch + Streamlit.")
    st.write("📚 Dataset: Food-101")

# ----- Cargar modelo y diccionarios -----
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 101)
    model.load_state_dict(torch.load("model/classifier.pth", map_location="cpu"))
    model.eval()
    return model

@st.cache_data
def load_class_dict():
    with open("model/idx_to_class.json", "r") as f:
        return json.load(f)

@st.cache_data
def load_recipes():
    with open("data/recipes.json", "r", encoding="utf-8") as f:
        return json.load(f)

model = load_model()
idx_to_class = load_class_dict()
recipes = load_recipes()

# ----- Transformación de la imagen -----
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# ----- Predicción -----
def predict(image):
    img_tensor = transform_image(image)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        class_name = idx_to_class[str(predicted.item())]
    return class_name

# ----- Interfaz principal -----
st.markdown("<h1 style='text-align: center;'>🍽️ Identificador de Platillos con Receta</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Descubre qué estás comiendo y aprende a prepararlo</p>", unsafe_allow_html=True)

with st.container():
    st.markdown("### 📤 Elegir imagen")
    option = st.radio("Selecciona un método de entrada:", ("📁 Subir imagen", "📷 Usar cámara"), horizontal=True)

    image = None
    if option == "📁 Subir imagen":
        uploaded_file = st.file_uploader("Sube una imagen del platillo", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
    elif option == "📷 Usar cámara":
        camera_input = st.camera_input("Toma una foto")
        if camera_input is not None:
            image = Image.open(camera_input).convert("RGB")

# Procesamiento y resultado
if image:
    st.image(image, caption="📸 Imagen seleccionada", use_container_width=True)

    with st.spinner("Analizando imagen..."):
        class_name = predict(image)

    st.success(f"🍛 Platillo identificado: **{class_name}**")

    if class_name in recipes:
        recipe_info = recipes[class_name]

        st.markdown("### 🧾 Receta Detallada")

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("#### 🥕 Ingredientes")
        with col2:
            for item in recipe_info["ingredients"]:
                st.markdown(f"✅ {item}")

        st.markdown("#### 👨‍🍳 Instrucciones")
        st.markdown(f"<div style='text-align: justify;'>{recipe_info['instructions']}</div>", unsafe_allow_html=True)
    else:
        st.warning("🚫 No se encontró receta para este platillo.")

