from flask import Flask, request, jsonify
import numpy as np
import requests
import base64
import cv2

# Initialize Flask app
app = Flask(__name__)
# IBM Watson ML credentials
API_KEY = "hTWgNz0tDfdj5C3IFCPsipAExOSYBQCciJkPpnDnFyhm"

CLASS_NAMES = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 
               'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']

def get_token_header():
    """Get authentication token for IBM Watson ML API."""
    token_response = requests.post(
        'https://iam.cloud.ibm.com/identity/token',
        data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'}
    )
    token_response.raise_for_status()  # Raise an error for bad responses
    mltoken = token_response.json()["access_token"]
    return {'Content-Type': 'application/json', 'Authorization': f'Bearer {mltoken}'}

def preprocess_image(encoded_string):
    """Decodes base64 string and preprocesses the image using OpenCV."""
    try:
        image_bytes = base64.b64decode(encoded_string)
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)  # Convert to BGR format
        image = cv2.resize(image, (227, 227))  # Resize to model input size
        image = image.astype(np.float32) / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    except Exception as e:
        return None, f"Error during image preprocessing: {e}"

def predict_image(file):
    """Score the image using the deployed IBM Watson model."""
    processed_image = preprocess_image(file)
    if processed_image is None:
        return None, "Failed to preprocess image"

    payload_scoring = {"input_data": [{"values": processed_image.tolist()}]}
    
    try:
        response = requests.post(
            'https://jp-tok.ml.cloud.ibm.com/ml/v4/deployments/1a1b1ba9-87d3-4aea-ae58-311fe384fbdb/predictions?version=2021-05-01',
            json=payload_scoring, 
            headers=get_token_header()
        )
        response.raise_for_status()  # Check for HTTP errors
        predictions = response.json()['predictions'][0]['values'][0]
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = round(100 * np.max(predictions), 2)
        return predicted_class, confidence
    except Exception as e:
        return None, f"Prediction error: {e}"

@app.route('/api/python', methods=['GET'])
def hello():
    return "hello"

@app.route('/api/predict', methods=['POST'])
def process_image():
    """Process the incoming image and return the prediction."""
    data = request.get_json()
    if not data or 'data' not in data:
        return jsonify({'error': 'No image data provided'}), 400
    
    file = data.get("data")
    if not file:
        return jsonify({"error": "No data provided"}), 400

    predicted_class, confidence = predict_image(file)

    if predicted_class is None:
        return jsonify({"error": confidence}), 500

    return jsonify({
        "result": "This image most likely belongs to {} with a {:.2f}% confidence.".format(predicted_class, confidence)
    }), 200