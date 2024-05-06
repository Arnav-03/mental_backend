from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib  # Import joblib to load the .pkl file

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow requests from any origin

# Load the machine learning model from the .pkl file
model = joblib.load("your_model.pkl")

@app.route('/', methods=['POST'])
def predict():
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
    predictions = model.predict([features])  # Assuming your model accepts a list of features
    
    # Return predictions as a response
    return jsonify({'prediction': predictions})

if __name__ == '__main__':
    app.run()
