from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib  # I
import os
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow requests from any origin
# Load the machine learning model from the .pkl file
model = joblib.load("svm_model.pkl")

@app.route('/predict/<value>', methods=['POST'])  # Define route with parameter
def predict(value):
    ddata = request.get_json()
    print("Data received from frontend:", ddata)
      
    # Extract features from the received data (you may need to preprocess the data)
    features = [
        ddata['age'],
        ddata['gender'],
        ddata['education'],
        ddata['personal_computer'],
        ddata['employed'],
        ddata['disabled'],
        ddata['internet_access'],
        ddata['parents'],
        ddata['gap_in_resume'],
        ddata['unemployed'],
        ddata['reads_outside_work_school'],
        ddata['difficulty_concentrating'],
        ddata['anxiety'],
        ddata['depression'],
        ddata['obsessive_thoughts'],
        ddata['fluctuating_mood'],
        ddata['panic_episodes'],
        ddata['compulsive_habits'],
        ddata['fatigue']
    ]

    # Make predictions using the loaded model
    predictions = model.predict([features]).tolist()  # Convert predictions to a list
    
    # Return predictions as a response
    return jsonify({'prediction': predictions})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)