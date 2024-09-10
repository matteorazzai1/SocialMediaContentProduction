import base64
import random
import string

from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch
from torchvision.transforms import ToTensor
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
from urllib.parse import quote
import io
import base64


def generate_image_example():
    image_url = "/static/images/example.jfif"
    return image_url


def load_images_from_folder(folder_path):
    images = []
    paths = []
    for filename in os.listdir(folder_path):
        img_path = folder_path + "/" + filename

        try:
            with Image.open(img_path) as img:
                images.append(img.copy())
                paths.append(img_path)
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
    return images, paths


#to use an image as input to Stable Diffusion, the image has to be downloaded from github, so we upload the image on a github repo to allow the download subsequently
def upload_image(path_locale, field, firm):
    letters = string.ascii_letters + string.digits

    token = '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'  #token has expired when the repo was put to public, insert a valid token here
    owner = 'matteorazzai1'
    repo = 'photoHandling'
    path = 'image_' + firm + ''.join(random.choice(letters)) + '.jpg'
    branch = 'main'
    image_path = path_locale

    # Read and encode image
    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

    # Prepare request data
    data = {
        'message': 'Add image',
        'content': encoded_image
    }

    # Make the API request
    response = requests.put(
        f'https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}',
        headers={
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        },
        json=data
    )

    # Check response status
    if response.status_code == 201:
        print('Image uploaded successfully.')
        # Construct the URL
        url = f'https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}'
        print('Image URL:', url)
        return url
    else:
        print('Failed to upload image:', response.json())
        return None


def retrieve_init_image(caption):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image_folder_path = "../dataset/data_image"

    # Load and preprocess images
    images, paths = load_images_from_folder(image_folder_path)

    text_inputs = processor(text=caption[:77], return_tensors="pt").input_ids.to(device)

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

    #images[np.argmax(probs[0])].show()
    return paths[np.argmax(probs[0])]


def generate_image(caption, field, firm):
    init_image_path = retrieve_init_image(field + " " + firm)

    init_image_url = upload_image(init_image_path, field, firm)

    # Load the pre-trained Stable Diffusion model
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")  # Use GPU if available, otherwise use "cpu"

    # Define a prompt
    prompt = "An image related to this caption: " + caption

    response = requests.get(init_image_url)
    init_image = Image.open(BytesIO(response.content)).convert("RGB")
    init_image = init_image.resize((768, 512))

    images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images

    # Save the generated image
    URL = "static/images/generated_image_" + firm + "_" + field + ".png"
    images[0].save(URL)

    return URL


def create_image_prompt(caption, image_prompt):
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

    image_prompt_str = "An image related to this caption " + caption + "with the following characteristics:".join(
        prompt_parts)
    return image_prompt_str
