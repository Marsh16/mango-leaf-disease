import tensorflow as tf
import numpy as np
import requests
import base64
from PIL import Image
import io
from ibm_watson_machine_learning import APIClient

# IBM Watson ML credentials
API_KEY =  "hTWgNz0tDfdj5C3IFCPsipAExOSYBQCciJkPpnDnFyhm"

CLASS_NAMES = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']

def get_token_header():
    token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
    API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
    mltoken = token_response.json()["access_token"]
    header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}
    return header

def predict_image(file):
    """Score the image using the deployed model."""
    processed_image = preprocess_image(file)
    payload_scoring = {"input_data": [{"values": processed_image.numpy().tolist()}]}
    try:
        response_scoring = requests.post('https://jp-tok.ml.cloud.ibm.com/ml/v4/deployments/1a1b1ba9-87d3-4aea-ae58-311fe384fbdb/predictions?version=2021-05-01', json=payload_scoring, headers=get_token_header())
        response = response_scoring.json()
        predictions = response['predictions'][0]['values']
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = 100 * np.max(predictions)
        return predicted_class, confidence
    except Exception as e:
        return None, str(e)

def preprocess_image(encoded_string):
    """Decodes base64 string and preprocesses the image."""
    try:
        image_bytes = base64.b64decode(encoded_string)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')  # Ensure 3 channels
        image = image.resize((227, 227))
        image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        return tf.convert_to_tensor(image_array)
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None
