# SocialMediaContentProduction

*Project work for the Business and Project Management Course, MSc in Artificial Intelligence and Data Engineering @ UNIPI, A.Y. 2023/2024*

## Overview
This project aims to develop a web app that allows the user to generate a social media post from a sequence of input fields. This tool is designed to be used from companies as a helpul method to create an engaging caption and an original image associated.

### Example
![Social media post generated using two input fields.](https://i.ibb.co/PW4jZQQ/gelato1.png)

*Figure 1: Social media post example using the 2 base case input fields, with the resulted image and caption.*

## Models used
The project has been built used the following models:
 - *Groq* (text generator)
 - *StableDiffusion* (image generator)

We developed some optimization solutions to improve the results, such as:

 - *RAG (Retrieval-Augmented Generation)* usage for the text generation
 - *Example-based technique* for the image generation

## Usage
The web app shows two basic input fields:

 - Company name
 - Main field

From these, the model starts to generate the post.

### Advanced Settings
It is available an "Advanced Settings Mode", that can be enabled selecting the apposite checkbox. It shows a longer list of input fields, designed after studying two papers about **Prompt Engineering**, as detailed in the main documentation.

## Requirements
The project has been deployed using Flask framework. It is essential to install:
- Python (version 3.6 or higher)
- pip (Python package installer)
- Flask (as python library)
- other libraries included in the Python files

## How to run the project
To run the Flask application: 

    python app.py

The application will be accessible from the address http://127.0.0.1:5000.
For an alternative method, other details about the project structure or how to run the project, please check the documentation.

## Important Notices
Notice that since Github blocks the API key every time it gets posted on a public repository, you may need to put your own key in the image_generator.py file (line 49) in order to make the code work (if you run it and you get an Authorization error, this is the case). 

Another important aspect to know about the project is that the StableDiffusion model, in order to run, needs a GPU compatible with CUDA, otherwise the application will not display the image realized by the custom model.
