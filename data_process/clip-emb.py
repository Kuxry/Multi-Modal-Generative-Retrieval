import os
import numpy as np
import torch
from PIL import Image
import clip
from tqdm import tqdm
import json

# 1. 设置路径
# 替换为你的 Flickr30k 图像目录路径
image_dir = "../data/Flickr30K/flickr30k-images"
# 替换为你的元数据文件路径（可选）
metadata_file = "../data/dataset_flickr30k.json"
# 输出嵌入文件的路径
output_path = "../data/Openflamingo_format/flicker/image_emb.npy"

# 2. 加载 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
# 使用 clip-ViT-L-14 模型
model, preprocess = clip.load("ViT-L/14", device=device)
print(f"Using device: {device}")

# 3. 获取图像文件列表
if os.path.exists(metadata_file):
    # 如果有元数据文件，从中提取图像文件名
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    # 假设元数据中有一个 'images' 字段，包含图像文件名
    image_files = [img['filename'] for img in metadata['images']]
else:
    # 如果没有元数据文件，直接从图像目录读取所有图像文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    print("No metadata file found, reading all images from directory.")

# 构建完整的图像路径列表
image_paths = [os.path.join(image_dir, img) for img in image_files]
print(f"Found {len(image_paths)} images to process.")

# 4. 生成图像嵌入
embeddings = []
for image_path in tqdm(image_paths, desc="Generating embeddings"):
    try:
        # 加载并预处理图像
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        # 生成嵌入
        with torch.no_grad():
            emb = model.encode_image(image)
        embeddings.append(emb.cpu().numpy())
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# 5. 保存嵌入为 .npy 文件
if embeddings:
    # 将所有嵌入拼接成一个 NumPy 数组
    embeddings = np.concatenate(embeddings, axis=0)
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # 保存为 .npy 文件
    np.save(output_path, embeddings)
    print(f"Embeddings saved to {output_path}")
    print(f"Shape of embeddings: {embeddings.shape}")
else:
    print("No embeddings were generated. Please check your image paths or data.")