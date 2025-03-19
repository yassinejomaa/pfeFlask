from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) 

# Load the pre-trained model for zero-shot classification
classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli", tokenizer="xlm-roberta-large")


# Categories for classification
categories = ["Food", "Transport", "Entertainment", "Health", "Electronics", "Fashion", "Housing","Others"]

@app.route('/predict', methods=['POST'])
def predict():
    # Get the product name from the request
    data = request.get_json()
    product = data.get('product', '')
    
    if not product:
        return jsonify({'error': 'Product name is required'}), 400
    
    # Predict category for the product
    result = classifier(product, categories)
    predicted_category = result["labels"][0]
    
    return jsonify({
        'product': product,
        'predicted_category': predicted_category
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

