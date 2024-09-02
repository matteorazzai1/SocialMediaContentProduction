from diffusers import StableDiffusionPipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from io import BytesIO
import requests
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import clip
from transformers import CLIPProcessor, CLIPModel
import os
import numpy as np

def generate_image_example():
    image_url = "/static/images/example.jfif"
    return image_url


def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        try:
            with Image.open(img_path) as img:
                images.append(img.copy())  # Copy to avoid potential issues with lazy loading
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
    return images



def retrieve_init_image(caption):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Define the transform
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    image_folder_path = "../dataset/data_image"

    # Load and preprocess images
    images = load_images_from_folder(image_folder_path)


    text_inputs = processor(text=caption, return_tensors="pt").input_ids.to(device)

    # Preprocess the images
    image_inputs = torch.stack(
            [processor(images=image, return_tensors="pt").pixel_values.squeeze(0) for image in images]).to(device)

    with torch.no_grad():
        # Encode images and text
        image_features = model.get_image_features(pixel_values=image_inputs)
        text_features = model.get_text_features(input_ids=text_inputs)

        # Compute similarity
        image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
        text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
        similarity = torch.matmul(text_features, image_features.T)

        # Convert similarity to probabilities
        probs = similarity.softmax(dim=-1).cpu().numpy()

    images[np.argmax(probs[0])].show()
    return images[np.argmax(probs[0])]


def generate_image(caption, field, firm):

    init_image=retrieve_init_image(caption)

    # Load the reference image
    #init_image_path = "../generated_image/generated_image.png"
    #init_image = Image.open(init_image_path).convert("RGB")
    #init_image = init_image.resize((512, 512))

    # Load the pre-trained Stable Diffusion model
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")  # Use GPU if available, otherwise use "cpu"
    #self.update_progress_bar(60)

    # Define a prompt
    prompt = "An image related to this caption" + caption

    # Generate an image using the init image
    strength = 0.8  # Control how much the init image influences the final output (0-1 range)
    image = pipe(prompt=prompt, init_image=init_image, strength=strength).images[0]
    #self.update_progress_bar(80)

    # Save the generated image
    URL = "static/images/generated_image_" + firm + "_" + field + ".png"
    image.save(URL)

    return URL




def generate_image_adv(caption, field, firm, image_prompt):
    # Load the pre-trained Stable Diffusion model
    model_id = "CompVis/stable-diffusion-v1-4"  # or another model variant
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")  # Use GPU if available, otherwise use "cpu"

    # Define a prompt
    prompt = create_image_prompt(image_prompt)

    # Generate an image
    image = pipe(prompt).images[0]

    # Save the generated image
    image_url = "static/images/generated_image_" + firm + "_" + field + ".png"
    image.save()

    return image_url


def create_image_prompt(image_prompt):
    image_type = image_prompt.get('image_type', '')
    subject = image_prompt.get('subject', '')
    environment = image_prompt.get('environment', '')
    light = image_prompt.get('light', '')
    color = image_prompt.get('color', '')
    point_view = image_prompt.get('point_view', '')
    art_style = image_prompt.get('art_style', '')
    photo_type = image_prompt.get('photo_type', '')

    prompt_parts = []

    if image_type:
        prompt_parts.append(f"Image type: {image_type}.")
    if subject:
        prompt_parts.append(f"Subject description: {subject}.")
    if environment:
        prompt_parts.append(f"Environment: {environment}.")
    if light:
        prompt_parts.append(f"Lighting: {light}.")
    if color:
        prompt_parts.append(f"Color scheme: {color}.")
    if point_view:
        prompt_parts.append(f"Point of view: {point_view}.")
    if art_style:
        prompt_parts.append(f"Art style: {art_style}.")
    if photo_type:
        prompt_parts.append(f"Photo type: {photo_type}.")

    image_prompt_str = " ".join(prompt_parts)
    return image_prompt_str

