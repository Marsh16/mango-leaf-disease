import tensorflow as tf
import numpy as np
import base64
from PIL import Image
import io
from ibm_watson_machine_learning import APIClient

# IBM Watson ML credentials
WML_CREDENTIALS = {
    "apikey": "hTWgNz0tDfdj5C3IFCPsipAExOSYBQCciJkPpnDnFyhm",
    "url": "https://jp-tok.ml.cloud.ibm.com"
}
SPACE_ID="094b021f-4f37-4ba4-8e3b-137f25ff2837"
CLASS_NAMES = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']

def get_wml_client():
    """Initialize and return IBM Watson ML API client."""
    return APIClient(WML_CREDENTIALS)

def predict_image(deployment_uid, file):
    """Score the image using the deployed model."""
    processed_image = preprocess_image(file)
    wml_client = get_wml_client()
    wml_client.set.default_space(SPACE_ID)
    payload = {"input_data": [{"values": processed_image.numpy().tolist()}]}

    try:
        result = wml_client.deployments.score(deployment_uid, payload)
        predictions = result['predictions'][0]['values']
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
