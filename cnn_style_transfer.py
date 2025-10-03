import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
#import matplotlib.pyplot as plt
from torchvision.models import vgg19 as vggm, VGG19_Weights
import os
from pathlib import Path
import gradio as gr
from tqdm.auto import tqdm

SIZE = (640, 480) #(H, W)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# Load the VGG19 model
vgg19 = vggm(weights=VGG19_Weights.IMAGENET1K_V1).to(device)
vgg19.eval()

mean = (0.485, 0.456, 0.406)  # ImageNet mean
std = (0.229, 0.224, 0.225)   # ImageNet std

#Preprocessing
transforms_preprocess = transforms.Compose([
    transforms.Resize(SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

#Image loading utility
def image_load(img, transform = transforms_preprocess, verbose= True):
    if isinstance(img, str):
        img = Image.open(img).convert('RGB')
    elif isinstance(img, Image.Image):
        img = img.convert('RGB')
    elif isinstance(img, torch.Tensor):
        # Assume it's already transformed, just ensure batch dimension
        if img.ndim == 3:  # C,H,W
            img = img.unsqueeze(0)
        elif img.ndim == 4:  # already batched
            pass
        else:
            raise ValueError(f"Unexpected tensor shape: {img.shape}")
        if verbose:
            print(img.shape)
        return img.to(device)
    else:
        raise TypeError(f"Unsupported input type: {type(img)}")

    # Only apply transform if we got a PIL image

    image = transform(img)
    image = torch.unsqueeze(image, dim= 0)
    if verbose:
        print(image.shape)
    return image.to(device)  


#Image paths
STYLE_IMAGE_PATH = './assets/style/'
CONTENT_IMAGE_PATH = './assets/target/'

style_files= os.listdir(STYLE_IMAGE_PATH)
content_files= os.listdir(CONTENT_IMAGE_PATH)

style_files = [Path(STYLE_IMAGE_PATH, f) for f in style_files]
content_files = [Path(CONTENT_IMAGE_PATH, f) for f in content_files]


# style_tr = image_load(Path(STYLE_IMAGE_PATH, style_files[-1]))
# content_tr = image_load(Path(CONTENT_IMAGE_PATH, content_files[-1]))


# Feature Extraction map : the first convolution layers in each blocks
LOSS_LAYERS = { '0': 'conv1_1', 
                '5': 'conv2_1',  
                '10': 'conv3_1', 
                '19': 'conv4_1', 
                '21': 'conv4_2', 
                '28': 'conv5_1'}


#Feature extraction using the model
def feature_extractor(x, model_features):
    extracted_features = {}
    for name, layer in model_features._modules.items():
        x = layer(x) #The image tensor should pass through all the layers
        if name in LOSS_LAYERS:
            extracted_features[LOSS_LAYERS[name]] = x
    return extracted_features



#Calculating Gram Matrix
def gram_matrix_calculator(feature_tensor):
    # feature_tensor: (N, C, H, W); assume N=1
    n, c, h, w = feature_tensor.size()
    feat = feature_tensor.squeeze(0)        # (C, H, W)
    feat = feat.view(c, h * w)              # (C, H*W)
    gram = torch.mm(feat, feat.t())         # (C, C)
    gram = gram.div(c * h * w)
    return gram

def training_loop(style_image, content_image, num_steps=300, style_weight=1e8, content_weight=1.0):
    style_tr = image_load(style_image)
    content_tr = image_load(content_image)

    # Freeze VGG parameters
    for p in vgg19.parameters():
        p.requires_grad = False

    # Extract features
    style_features = feature_extractor(style_tr, vgg19.features)
    content_features = feature_extractor(content_tr, vgg19.features)
    style_gram = {layer: gram_matrix_calculator(style_features[layer]) for layer in style_features}

    # Initialize target as a clone of content
    target = content_tr.clone().detach().requires_grad_(True)

    # LBFGS optimizer
    optimizer = torch.optim.LBFGS([target])

    # Layer weights
    weights = {'conv1_1': 1.0, 'conv2_1': 0.8, 'conv3_1': 0.6, 'conv4_1': 0.4, 'conv5_1': 0.2}
    loss_fn = nn.functional.mse_loss

    run = [0]  # mutable counter so closure can update it

    def closure():
        optimizer.zero_grad()

        target_features = feature_extractor(target, vgg19.features)

        # Content loss
        c_loss = loss_fn(target_features['conv4_2'], content_features['conv4_2'])

        # Style loss
        s_loss = 0.0
        for layer in weights:
            target_gram = gram_matrix_calculator(target_features[layer])
            style_gram_matrix = style_gram[layer]
            s_loss += loss_fn(target_gram, style_gram_matrix) * weights[layer]

        total_loss = style_weight * s_loss + content_weight * c_loss
        total_loss.backward()

        if run[0] % 50 == 0:
            print(f"Step {run[0]} | Total: {total_loss.item():.4f} | "
                  f"Style: {s_loss.item():.4f} | Content: {c_loss.item():.4f}")
        run[0] += 1

        return total_loss

    # Run optimization
    with tqdm(range(num_steps), desc="Optimizing") as pbar:
        for _ in pbar:
            optimizer.step(closure)

    return target.detach().cpu()

#Tensor to image
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

def tensor_to_image(image_tensor):
    # image_tensor: (1, C, H, W) normalized
    image = image_tensor.clone().detach().cpu().squeeze(0)  # (C, H, W)
    image = image.permute(1, 2, 0).numpy()                  # (H, W, C)
    image = image * np.array(std)[None, None, :] + np.array(mean)[None, None, :]  # de-normalize
    image = np.clip(image, 0.0, 1.0)
    image_uint8 = (image * 255).astype(np.uint8)
    return Image.fromarray(image_uint8)



def style_transfer(style_image, content_image):
    optimized = training_loop(style_image, content_image, num_steps=2)
    return tensor_to_image(optimized)

    
            
if __name__ == "__main__":
 
    gr.close_all()
    with gr.Blocks(theme=gr.themes.Glass()) as interface:
        with gr.Row():
            gr.Markdown("<h2 style='color: blue;'>Vanilla CNN Style Transfer</h2>")
        with gr.Row():
            with gr.Column():
                style_image=gr.Image(type="pil", label="Style Image",height=300,width=300)
            with gr.Column():
                content_image=gr.Image(type="pil", label="Content Image",height=300,width=300)
            with gr.Column():
                image_output=gr.Image(type="pil", label="Generated Image",height=300,width=300)
        
        with gr.Row():
            style_transfer_button=gr.Button("Style Transfer",variant= "primary")
            reset_button=gr.Button("Reset",variant= "secondary")

        with gr.Row():
            style_examples=gr.Examples(examples=[f for f in style_files], inputs=[style_image], label="Style Images")
            content_examples=gr.Examples(examples=[f for f in content_files], inputs=[content_image], label="Content Images")

        style_transfer_button.click(
            fn=style_transfer,
            inputs=[style_image, content_image],
            outputs=[image_output]
        )
        reset_button.click(
            fn=lambda: gr.update(value=None),
            outputs=[image_output]
        )
        interface.launch(share=False, server_port=7860)
        
