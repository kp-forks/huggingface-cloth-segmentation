import os
import torch
import gradio as gr
from PIL import Image
from process import load_seg_model, get_palette, generate_mask

# Automatically select device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Ensure model directory exists
model_dir = 'model'
os.makedirs(model_dir, exist_ok=True)
checkpoint_path = os.path.join(model_dir, 'cloth_segm.pth')

# Download the model if not present
if not os.path.exists(checkpoint_path):
    import gdown
    url = 'https://drive.google.com/uc?id=1w0nZzH9g6n5l3xQ8Z8Z8Z8Z8Z8Z8Z8Z'  # Replace with actual URL
    gdown.download(url, checkpoint_path, quiet=False)

# Load model
net = load_seg_model(checkpoint_path, device=device)
palette = get_palette(4)

def run(img: Image.Image) -> Image.Image:
    try:
        cloth_seg = generate_mask(img, net=net, palette=palette, device=device)
        return cloth_seg
    except Exception as e:
        print(f"Error during processing: {e}")
        return None

# Define Gradio interface
title = "Demo for Cloth Segmentation"
description = "An app for Cloth Segmentation using U2NET."

iface = gr.Interface(
    fn=run,
    inputs=gr.Image(type="pil", label="Input Image"),
    outputs=gr.Image(type="pil", label="Cloth Segmentation"),
    title=title,
    description=description
)

if __name__ == "__main__":
    iface.launch(share=True)
