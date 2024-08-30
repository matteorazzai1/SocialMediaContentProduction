from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/create_post', methods=['POST'])
def create_post():
    company_name = request.json.get('company_name')
    main_field = request.json.get('main_field')

    new_post = {
        "text": f"New post created for {company_name} in the field of {main_field}!",
        "image_url": "/static/images/example.jfif"
    }

    return jsonify(new_post)

if __name__ == '__main__':
    app.run(debug=True)