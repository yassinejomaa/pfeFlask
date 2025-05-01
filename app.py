from flask import Flask, request, jsonify, Response
from transformers import pipeline
from flask_cors import CORS
from expense_advisor import ExpenseAdvisor 
import torch
from sklearn.preprocessing import MinMaxScaler
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

@app.route('/predict_expense/<int:input1>/<int:input2>/<int:input3>/<int:input4>', methods=['GET'])
def predict_expense(input1, input2, input3, input4):

    # Step 1: Prepare your input and min/max values
    input_values = [[input1, input2, input3, input4]]
    min_vals = [82.72, 0, 1, 0]
    max_vals = [86650.2,2, 26, 8]

    # Step 2: Fit MinMaxScaler using min/max as fake dataset
    scaler = MinMaxScaler()
    scaler.fit([min_vals, max_vals])  # Fit on boundary values only

    # Step 3: Transform the input
    normalized_input = scaler.transform(input_values)[0]

    # Convert the input to a NumPy array
    input_array = np.array([normalized_input])

    # Make the prediction
    prediction = model.predict(input_array)

    # Print and Return the response
    print("the predicted Expenses is:", prediction)
    response = {
        'prediction': prediction.tolist()
    }
    return jsonify(response)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

