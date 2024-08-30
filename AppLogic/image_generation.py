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

