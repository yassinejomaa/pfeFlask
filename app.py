from flask import Flask, request, jsonify, Response
from transformers import pipeline
from flask_cors import CORS
from expense_advisor import ExpenseAdvisor 
import torch
from sklearn.preprocessing import  StandardScaler
import pickle
import numpy as np

# Initialize Flask app
device = 0 if torch.cuda.is_available() else -1  # Utiliser 0 pour le GPU, -1 pour CPU
print(torch.cuda.is_available())
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

model = pickle.load(open("model_file.pkl", "rb"))

@app.route('/predict_expense/<float:income>/<int:bedrooms>/<int:vehicles>/<int:members>/<int:employed>', methods=['GET'])
def predict_expense(income, bedrooms, vehicles, members, employed):


    # Step 2: Reconstruct the StandardScaler used during training
    scaler = StandardScaler()
    
    # ⚠️ Replace these mean and scale values with actual values from your training scaler
    scaler.mean_ = np.array([3727.58003, 1.78491731, 0.0830595831, 4.62958768, 1.27152243])
    scaler.scale_ = np.array([4365.72591, 1.10275029, 0.350956128, 2.28743469, 1.14549119])
    scaler.var_ = np.array([19059562.7, 1.21605819, 0.123170204, 5.23235746, 1.31215006])
    scaler.n_features_in_ = 5


    # Step 3: Prepare input
    input_data = np.array([[income, bedrooms, vehicles, members, employed]])

    # Step 4: Scale input
    input_scaled = scaler.transform(input_data)

    # Step 5: Predict
    prediction = model.predict(input_scaled)

    # Step 6: Return response
    response = {
        'expense_prediction': prediction.tolist()
    }
    return jsonify(response)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

