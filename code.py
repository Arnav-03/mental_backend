import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
#load dataset
df = pd.read_csv(r"H:\mental illness ml model\website\backend\mental_illness_survey_1 -hehe.csv")
#make a copy of dataframe for transformation
df1 = df.copy()
#drop unrequired columns
df1.drop(['Respondent ID','Collector ID', 'Start Date', 'End Date','I receive food stamps','I am on section 8 housing','How many times were you hospitalized for your mental illness','Household Income','Region', 'IP Address','Annual income from social welfare programs',
       'Email Address', 'First Name', 'Last Name', 'Custom Data 1', 'Device Type','Annual income (including any social welfare programs) in USD','How many days were you hospitalized for your mental illness','I have been hospitalized before for my mental illness','Total length of any gaps in my resume in months'],
        axis=1, inplace=True)
# Define a dictionary mapping the old column names to the new rephrased names
new_column_names = {
    'Lack of concentration': 'Difficulty Concentrating',
    'Anxiety': 'Feelings of Anxiety',
    'Depression': 'Symptoms of Depression',
    'Obsessive thinking': 'Obsessive Thoughts',
    'Mood swings': 'Fluctuating Mood',
    'Panic attacks': 'Episodes of Panic',
    'Compulsive behavior': 'Compulsive Habits',
    'Tiredness': 'Fatigue'
}

# Use the rename() function to replace the column names
df1.rename(columns=new_column_names, inplace=True)
# Get the first column and drop it from the DataFrame
mental_illness_col = df1.pop('I identify as having a mental illness')

# Add the column to the end of the DataFrame
df1['I identify as having a mental illness'] = mental_illness_col
# Define a mapping of column names to their edited versions
column_name_mapping = {
    'Education': 'Education Level',
    'I have my own computer separate from a smart phone': 'Owns a Personal Computer',
    'I am currently employed at least part-time': 'Is Currently Employed at least Part-Time',
    'I am legally disabled': 'Is Legally Disabled',
    'I have my regular access to the internet': 'Has Regular Access to the Internet',
    'I live with my parents': 'Lives with Parents',
    'I have a gap in my resume': 'Has a Gap in Resume',
    'I am unemployed': 'Is Unemployed',
    'I read outside of work and school': 'Reads Outside of Work and School',
    'Difficulty Concentrating': 'Difficulty Concentrating',
    'Feelings of Anxiety': 'Feelings of Anxiety',
    'Symptoms of Depression': 'Symptoms of Depression',
    'Obsessive Thoughts': 'Obsessive Thoughts',
    'Fluctuating Mood': 'Fluctuating Mood',
    'Episodes of Panic': 'Episodes of Panic',
    'Compulsive Habits': 'Compulsive Habits',
    'Fatigue': 'Fatigue',
    'Age': 'Age',
    'Gender': 'Gender',
    'I identify as having a mental illness': 'Identifies as having a mental illness'
}

# Rename the columns using the mapping
df1.rename(columns=column_name_mapping, inplace=True)
# List of columns with null values
cols_with_null = ['Compulsive Habits', 'Fluctuating Mood', 'Obsessive Thoughts', 
                  'Episodes of Panic', 'Difficulty Concentrating', 
                  'Symptoms of Depression', 'Feelings of Anxiety', 'Fatigue']

# Fill null values with 1 if not null, else 0
for col in cols_with_null:
    df1[col] = df1[col].notnull().astype(int)
# Replace values in 'Gender' column using the mapping dictionary
df1['Gender'].replace({'Male': 1, 'Female': 0}, inplace=True)
# Define mapping dictionary for age ranges
age_mapping = {
    '18-29': 1,
    '30-44': 2,
    '45-60': 3,
    '> 60': 4
}

# Map the 'Age' column using the mapping dictionary
df1['Age'] = df1['Age'].map(age_mapping)


# Define mapping dictionary for education levels
education_mapping = {
    'Some highschool': 1,
    'High School or GED': 2,
    'Some Undergraduate': 3,
    'Completed Undergraduate': 4,
    'Some Masters': 5,
    'Completed Masters': 6,
    'Some Phd': 7,
    'Completed Phd': 8
}

# Map the 'Education' column using the mapping dictionary
df1['Education Level'] = df1['Education Level'].map(education_mapping)

# Define a list of columns containing "Yes" and "No" values
yes_no_columns = ['Is Currently Employed at least Part-Time', 'Is Legally Disabled', 
                  'Has Regular Access to the Internet', 'Lives with Parents', 
                  'Has a Gap in Resume', 'Is Unemployed', 'Reads Outside of Work and School',
                  'Identifies as having a mental illness','Owns a Personal Computer']

# Map "Yes" and "No" values to 1 and 0 respectively for each column
for col in yes_no_columns:
    df1[col] = df1[col].map({'Yes': 1, 'No': 0})

df1.head(10)
X=df1.iloc[:,:-1].values
y=df1.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 5)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)

classes= df1['Identifies as having a mental illness'].unique()
from sklearn.metrics import ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot()
plt.show()
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
print(report)


import joblib

# Save the trained SVM model to a file
joblib.dump(classifier, 'svm_model.pkl')

