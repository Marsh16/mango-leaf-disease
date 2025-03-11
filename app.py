from flask import Flask, request, jsonify
from prediction_model.classifier import predict_image

# Initialize Flask app
app = Flask(__name__)

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

if __name__ == '__main__':
    app.run(debug=True,port=5006)