from flask import Flask, request, jsonify, Response
from transformers import pipeline
from flask_cors import CORS
from expense_advisor import ExpenseAdvisor 

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type", "Authorization"]}})

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


advisor = ExpenseAdvisor()

@app.route('/generate_advice', methods=['POST'])
def generate_advice():
    data = request.get_json()
    expenses = data.get('expenses')

    if not expenses or not isinstance(expenses, list):
        return jsonify({"error": "Please provide a list of expenses."}), 400

    return Response(advisor.generate_advice_stream(expenses), mimetype='text/plain')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

