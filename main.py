import streamlit as st
import pickle
import pandas as pd

# Load the saved machine learning model
model = pickle.load(open('model.pkl', 'rb'))
col_map_dict = pickle.load(open('col_map_dict.pkl', 'rb'))

# drop HeartDisease from col_map_dict
col_map_dict.pop('HeartDisease')

# Define the categorical column options
smoking_options = ['Yes', 'No']
alcohol_drinking_options = ['No', 'Yes']
stroke_options = ['No', 'Yes']
diff_walking_options = ['No', 'Yes']
sex_options = ['Female', 'Male']
age_category_options = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older']
race_options = ['White', 'Black', 'Asian', 'American Indian/Alaskan Native', 'Other', 'Hispanic']
diabetic_options = ['No', 'Yes', 'No, borderline diabetes', 'Yes (during pregnancy)']
physical_activity_options = ['No', 'Yes']
gen_health_options = ['Poor', 'Fair', 'Good', 'Very good', 'Excellent']
asthma_options = ['No', 'Yes']
kidney_disease_options = ['No', 'Yes']
skin_cancer_options = ['No', 'Yes']

# label encode function
def label_encode(df_sub):
    for col in df_sub.columns:
        data = df_sub[col].values[0]
        if col in col_map_dict.keys():
            df_sub[col] = col_map_dict[col][data]
    return df_sub

# Define a function to preprocess the input data
def preprocess_input(data):
    data = label_encode(data)
    return data

# Define the input form
st.title('Heart Disease Prediction System')
st.subheader('Enter the following patient information:')

bmi = st.number_input('Body Mass Index (BMI)', min_value=0.0, max_value=100.0, value=0.0)
physical_health = st.number_input('Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good? (0-30 days)', min_value=0.0, max_value=100.0, value=0.0)
mental_health = st.number_input('Thinking about your mental health, for how many days during the past 30 days was your mental health not good? (0-30 days)', min_value=0.0, max_value=100.0, value=0.0)
sleep_time = st.number_input('On average, how many hours of sleep do you get in a 24-hour period?', min_value=0.0, max_value=24.0, value=0.0)
smoking = st.selectbox('Have you smoked at least 100 cigarettes in your entire life? ', options=smoking_options)
alcohol_drinking = st.selectbox('Alcohol Drinking', options=alcohol_drinking_options)
stroke = st.selectbox('(Ever told) (you had) a stroke?', options=stroke_options)
diff_walking = st.selectbox('Do you have serious difficulty walking or climbing stairs?', options=diff_walking_options)
sex = st.selectbox('Sex', options=sex_options)
age_category = st.selectbox('Age Category', options=age_category_options)
race = st.selectbox('Race', options=race_options)
diabetic = st.selectbox('(Ever told) (you had) diabetes?', options=diabetic_options)
physical_activity = st.selectbox('Adults who reported doing physical activity or exercise during the past 30 days other than their regular job', options=physical_activity_options)
gen_health = st.selectbox('General Health', options=gen_health_options)
asthma = st.selectbox('(Ever told) (you had) asthma?', options=asthma_options)
kidney_disease = st.selectbox('Not including kidney stones, bladder infection or incontinence, were you ever told you had kidney disease?', options=kidney_disease_options)
skin_cancer = st.selectbox('Skin Cancer', options=skin_cancer_options)

pred_click = st.button('Predict')

if pred_click:


    # Convert the input data into a Pandas dataframe
    data = pd.DataFrame({
        'BMI': bmi,
        'Smoking': smoking,
        'AlcoholDrinking': alcohol_drinking,
        'Stroke': stroke,
        'PhysicalHealth': physical_health,
        'MentalHealth': mental_health,
        'DiffWalking': diff_walking,
        'Sex': sex,
        'AgeCategory': age_category,
        'Race': race,
        'Diabetic': diabetic,
        'PhysicalActivity': physical_activity,
        'GenHealth': gen_health,
        'SleepTime': sleep_time,
        'Asthma': asthma,
        'KidneyDisease': kidney_disease,
        'SkinCancer': skin_cancer
        }, index=[0])

    # Preprocess the input data
    data = preprocess_input(data)

    # Make the prediction
    prediction = model.predict_proba(data)[:, 1][0]

    # Display the prediction
    st.write(f'The predicted probability of heart disease is: {prediction:.2%}')
