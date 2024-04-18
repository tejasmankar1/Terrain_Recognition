from flask import Flask, request, jsonify, render_template,redirect, url_for
from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from prediction.predict import Terrain
from utils.utils import decodeImage

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = Terrain(self.filename)
        
@app.route("/")
#@cross_origin()
def home():
    return render_template('index.html')

@app.route('/login.html', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Handle the login process here
        pass
    return render_template('login.html')
    
@app.route('/signup.html', methods=['GET', 'POST'])
def Signup():
    if request.method == 'POST':
        # Handle the login process here
        pass
    return render_template('signup.html')


@app.route("/input.html", methods=['GET'])
def Input():
    return render_template('input.html')


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the image to a file
    image_filename = "inputImage.jpg"
    image_file.save(image_filename)

    # Perform prediction
    classifier = Terrain(image_filename)
    prediction_result = classifier.predictionterrain()

    # Extract the predicted class name from the result
    predicted_class = prediction_result[0]['Class:']

    # Return prediction result as JSON
    return jsonify({'prediction': [predicted_class]})




if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=5500)

