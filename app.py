import os
import requests
import zipfile
import streamlit as st
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Hàm tải tệp ZIP từ Hugging Face
def download_and_extract_zip(url, extract_to="images"):
    zip_path = "images.zip"
    response = requests.get(url, stream=True)
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_path)
    return extract_to

# Load CLIP model
@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return model, processor

# Load ALIGN model
@st.cache_resource
def load_align_model():
    model = AutoModel.from_pretrained("kakaobrain/align-base")
    processor = AutoProcessor.from_pretrained("kakaobrain/align-base")
    return model, processor

# Hàm tìm kiếm ảnh
def search_image(text_query, model, processor, text_embeddings, image_embeddings, image_paths, top_k=5):
    inputs = processor(text=[text_query], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs).numpy()

    similarities = cosine_similarity(text_features, image_embeddings)[0]
    top_k_indices = similarities.argsort()[-top_k:][::-1]

    result_images = [image_paths[i] for i in top_k_indices]
    return result_images

# Load embeddings và đường dẫn ảnh
@st.cache_data
def load_data(model_name, image_folder):
    if model_name == "CLIP":
        text_embeddings = np.load("text_embeddings.npy")
        image_embeddings = np.load("image_embeddings.npy")
    elif model_name == "ALIGN":
        text_embeddings = np.load("text_embeddings_align.npy")
        image_embeddings = np.load("image_embeddings_align.npy")

    image_filenames = np.load("image_filenames.npy", allow_pickle=True)
    image_paths = [os.path.join(image_folder, filename) for filename in image_filenames]

    return text_embeddings, image_embeddings, image_paths

# Giao diện Streamlit
st.set_page_config(page_title="Image Search", page_icon="📷", layout="wide")
st.title("📷 Image Search with CLIP and ALIGN")

# Tải và giải nén ảnh từ Hugging Face
zip_url = "https://huggingface.co/datasets/Lippovn04/images/resolve/main/images.zip"
image_folder = download_and_extract_zip(zip_url)

# Giao diện
model_name = st.selectbox("Chọn model:", ["CLIP", "ALIGN"])
text_query = st.text_input("Nhập mô tả văn bản:")
top_k = st.slider("Số lượng ảnh muốn tìm kiếm:", min_value=1, max_value=20, value=5)

if st.button("Tìm kiếm"):
    if text_query:
        if model_name == "CLIP":
            model, processor = load_clip_model()
        elif model_name == "ALIGN":
            model, processor = load_align_model()

        text_embeddings, image_embeddings, image_paths = load_data(model_name, image_folder)

        result_images = search_image(text_query, model, processor, text_embeddings, image_embeddings, image_paths, top_k=top_k)

        st.markdown("### 🎯 Kết quả tìm kiếm")
        cols = st.columns(3)
        for i, img_path in enumerate(result_images):
            col = cols[i % 3]
            try:
                with col:
                    st.image(Image.open(img_path), caption=os.path.basename(img_path), use_container_width=True)
            except FileNotFoundError:
                st.error(f"Không tìm thấy ảnh: {img_path}")
    else:
        st.warning("Vui lòng nhập nội dung!")
