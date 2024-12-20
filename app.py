import os
import requests
import zipfile
import streamlit as st
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Hàm tải tệp từ URL
def download_file(url, save_path):
    """Tải tệp từ URL và lưu vào đường dẫn cục bộ."""
    response = requests.get(url, stream=True)
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

# Hàm tải và giải nén tệp ZIP từng phần
def download_and_extract_zip(url, extract_to="images"):
    zip_path = "images.zip"
    response = requests.get(url, stream=True)
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Giải nén tệp theo từng phần thay vì toàn bộ một lúc
        for file in zip_ref.namelist():
            zip_ref.extract(file, extract_to)
            print(f"Đã giải nén: {file}")
    return extract_to

# Hàm tải embeddings từ Hugging Face
def load_embeddings_from_huggingface():
    hf_base_url = "https://huggingface.co/datasets/Lippovn04/Embeddings/resolve/main/"
    files = {
        "text_embeddings": "text_embeddings.npy",
        "text_embeddings_align": "text_embeddings_align.npy",
        "image_embeddings": "image_embeddings.npy",
        "image_embeddings_align": "image_embeddings_align.npy",
        "image_filenames": "image_filenames.npy",
    }
    for key, filename in files.items():
        if not os.path.exists(filename):
            download_file(hf_base_url + filename, filename)
            print(f"Tải xong: {filename}")
        else:
            print(f"File {filename} đã tồn tại, không cần tải lại.")

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

# Tải embeddings từ Hugging Face
load_embeddings_from_huggingface()
zip_url = "https://huggingface.co/datasets/Lippovn04/images/resolve/main/images.zip"
image_folder = download_and_extract_zip(zip_url)

# Giao diện Streamlit
st.set_page_config(page_title="Image Search", page_icon="📷", layout="wide")
st.title("📷 Image Search with CLIP and ALIGN")
st.write("Nhập mô tả văn bản và tìm kiếm ảnh phù hợp!")

# Lựa chọn model
model_name = st.selectbox("Chọn model:", ["CLIP", "ALIGN"])

# Nhập liệu
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
                    img = Image.open(img_path)
                    st.image(img, caption=os.path.basename(img_path), use_container_width=True)
            except FileNotFoundError:
                st.error(f"Không tìm thấy ảnh: {img_path}")
            except Exception as e:
                st.error(f"Lỗi khi tải ảnh: {str(e)}")
    else:
        st.warning("Vui lòng nhập nội dung!")
