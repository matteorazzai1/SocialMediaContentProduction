from flask import Flask, render_template, jsonify, request
# from caption_generation import generate_caption_example
from caption_generation import generate_caption, generate_caption_adv
# from image_generation import generate_image_example
from image_generation import generate_image, generate_image_adv

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/create_post', methods=['POST'])
def create_post():
    # Get elements from the interface fields
    company_name = request.json.get('company_name')
    main_field = request.json.get('main_field')
    setting_button = request.json.get('setting_button')
    prompt_values = request.json.get("prompt_values")

    image_prompt = prompt_values.get('image_prompt', {})
    text_prompt = prompt_values.get('text_prompt', {})

    # Generate caption and image
    # text = generate_caption(company_name, main_field)
    # image = generate_image(text, main_field, company_name)

    # Example code
    # text = generate_caption_example()
    # image = generate_image_example()

    # Generate caption and image (after prompt engineering changes)
    if setting_button:
        text = generate_caption_adv(company_name, main_field, text_prompt)
        image = generate_image_adv(text, main_field, company_name, image_prompt)
    else:
        print("CASO BASE")
        text = generate_caption(company_name, main_field)
        image = generate_image(text, main_field, company_name)

    new_post = {
        "text": text,
        "image": image
    }

    return jsonify(new_post)


if __name__ == '__main__':
    app.run(port=5002, debug=True)