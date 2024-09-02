from diffusers import StableDiffusionPipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from io import BytesIO
import requests




def generate_image_example():
    image_url = "/static/images/example.jfif"
    return image_url


def generate_image(caption, field, firm):
    # Load the pre-trained Stable Diffusion model
    model_id = "CompVis/stable-diffusion-v1-4"  # or another model variant
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")  # Use GPU if available, otherwise use "cpu"
    #self.update_progress_bar(60)

    # Define a prompt
    prompt = "An image related to this caption" + caption

    # Generate an image
    image = pipe(prompt).images[0]
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

