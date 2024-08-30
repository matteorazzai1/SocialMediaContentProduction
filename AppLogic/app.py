from flask import Flask, render_template, jsonify, request
from caption_generation import generate_caption
#from caption_generation import generate_caption_example
from image_generation import generate_image
#from image_generation import generate_image_example

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/create_post', methods=['POST'])
def create_post():
    # Get elements from the interface fields
    company_name = request.json.get('company_name')
    main_field = request.json.get('main_field')

    # Generate caption and image
    text = generate_caption(company_name, main_field)
    image = generate_image(text, main_field, company_name)

    # Example code
    #text = generate_caption_example()
    #image = generate_image_example()

    new_post = {
        "text": text,
        "image": image
    }

    return jsonify(new_post)


if __name__ == '__main__':
    app.run(debug=True)
