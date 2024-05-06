from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

app = Flask(__name__)

# Load the trained SVM model
classifier = joblib.load('svm_model.pkl')

# Function to preprocess user input
def preprocess_input(age, gender, education, personal_computer, employed, disabled, internet_access, parents, gap_in_resume, unemployed, reads_outside_work_school, difficulty_concentrating, anxiety, depression, obsessive_thoughts, fluctuating_mood, panic_episodes, compulsive_habits, fatigue):
    # Define mapping dictionaries for age and education levels
    age_mapping = {'18-29': 1, '30-44': 2, '45-60': 3, '> 60': 4}
    education_mapping = {'Some highschool': 1, 'High School or GED': 2, 'Some Undergraduate': 3,
                         'Completed Undergraduate': 4, 'Some Masters': 5, 'Completed Masters': 6,
                         'Some Phd': 7, 'Completed Phd': 8}
    
    # Convert user input to numerical values using the mappings
    age_numeric = age_mapping.get(age, -1)
    gender_numeric = 1 if gender.lower() == 'male' else 0
    education_numeric = education_mapping.get(education, -1)
    
    # Map "Yes" and "No" values to 1 and 0 respectively
    personal_computer_numeric = 1 if personal_computer.lower() == 'yes' else 0
    employed_numeric = 1 if employed.lower() == 'yes' else 0
    disabled_numeric = 1 if disabled.lower() == 'yes' else 0
    internet_access_numeric = 1 if internet_access.lower() == 'yes' else 0
    parents_numeric = 1 if parents.lower() == 'yes' else 0
    gap_in_resume_numeric = 1 if gap_in_resume.lower() == 'yes' else 0
    unemployed_numeric = 1 if unemployed.lower() == 'yes' else 0
    reads_outside_work_school_numeric = 1 if reads_outside_work_school.lower() == 'yes' else 0
    difficulty_concentrating_numeric = 1 if difficulty_concentrating.lower() == 'yes' else 0
    anxiety_numeric = 1 if anxiety.lower() == 'yes' else 0
    depression_numeric = 1 if depression.lower() == 'yes' else 0
    obsessive_thoughts_numeric = 1 if obsessive_thoughts.lower() == 'yes' else 0
    fluctuating_mood_numeric = 1 if fluctuating_mood.lower() == 'yes' else 0
    panic_episodes_numeric = 1 if panic_episodes.lower() == 'yes' else 0
    compulsive_habits_numeric = 1 if compulsive_habits.lower() == 'yes' else 0
    fatigue_numeric = 1 if fatigue.lower() == 'yes' else 0
    
    # Return a list of preprocessed values
    return [age_numeric, gender_numeric, education_numeric, personal_computer_numeric, employed_numeric,
            disabled_numeric, internet_access_numeric, parents_numeric, gap_in_resume_numeric, unemployed_numeric,
            reads_outside_work_school_numeric, difficulty_concentrating_numeric, anxiety_numeric, depression_numeric,
            obsessive_thoughts_numeric, fluctuating_mood_numeric, panic_episodes_numeric, compulsive_habits_numeric,
            fatigue_numeric]

# Function to make predictions based on user input
def predict_mental_illness(input_values):
    # Preprocess user input
    input_data = preprocess_input(*input_values)
    
    # Make prediction
    prediction = classifier.predict([input_data])
    
    # Return prediction
    return prediction[0]

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.get_json()
    
    # Extract input values
    input_values = (data['age'], data['gender'], data['education'], data['personal_computer'], data['employed'],
                    data['disabled'], data['internet_access'], data['parents'], data['gap_in_resume'],
                    data['unemployed'], data['reads_outside_work_school'], data['difficulty_concentrating'],
                    data['anxiety'], data['depression'], data['obsessive_thoughts'], data['fluctuating_mood'],
                    data['panic_episodes'], data['compulsive_habits'], data['fatigue'])
    
    # Make prediction
    prediction = predict_mental_illness(input_values)
    
    # Return prediction as JSON response
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
