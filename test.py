import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

# Load the trained SVM model
classifier = joblib.load('svm_model.pkl')

# Function to preprocess user input


def preprocess_input(age, gender, education, personal_computer, employed, disabled, internet_access, parents, gap_in_resume, unemployed, reads_outside_work_school, difficulty_concentrating, anxiety, depression, obsessive_thoughts, fluctuating_mood, panic_episodes, compulsive_habits, fatigue):
    # Define mapping dictionaries for age and education levels
    age_mapping = {'18': 1, '30': 2, '45': 3, '60': 4}
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

    # Display prediction to the user
    if prediction[0] == 0:
        print("Based on the provided information, the model predicts that the individual does not identify as having a mental illness.")
    else:
        print("Based on the provided information, the model predicts that the individual identifies as having a mental illness.")


# Ask the user for input
print("Please provide the following information:")
age = input("Age (18-29, 30-44, 45-60, > 60): ")
gender = input("Gender (Male/Female): ")
education = input(
    "Education Level (Some highschool, High School or GED, ..., Completed Phd): ")
personal_computer = input("Owns a Personal Computer (Yes/No): ")
employed = input("Is Currently Employed at least Part-Time (Yes/No): ")
disabled = input("Is Legally Disabled (Yes/No): ")
internet_access = input("Has Regular Access to the Internet (Yes/No): ")
parents = input("Lives with Parents (Yes/No): ")
gap_in_resume = input("Has a Gap in Resume (Yes/No): ")
unemployed = input("Is Unemployed (Yes/No): ")
reads_outside_work_school = input(
    "Reads Outside of Work and School (Yes/No): ")
difficulty_concentrating = input("Difficulty Concentrating (Yes/No): ")
anxiety = input("Feelings of Anxiety (Yes/No): ")
depression = input("Symptoms of Depression (Yes/No): ")
obsessive_thoughts = input("Obsessive Thoughts (Yes/No): ")
fluctuating_mood = input("Fluctuating Mood (Yes/No): ")
panic_episodes = input("Episodes of Panic (Yes/No): ")
compulsive_habits = input("Compulsive Habits (Yes/No): ")
fatigue = input("Fatigue (Yes/No): ")

# Make predictions based on user input
input_values = (age, gender, education, personal_computer, employed, disabled, internet_access, parents, gap_in_resume,
                unemployed, reads_outside_work_school, difficulty_concentrating, anxiety, depression,
                obsessive_thoughts, fluctuating_mood, panic_episodes, compulsive_habits, fatigue)
predict_mental_illness(input_values)
