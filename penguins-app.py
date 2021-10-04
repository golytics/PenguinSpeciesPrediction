import pandas as pd
import numpy as np
import pickle
import streamlit as st
list
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Predicting The Penguin Species
This app predicts the **Penguin** species!

Data Source: [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.
""")

st.subheader('How to use the model?')
''' 
You can use the model by modifying the User Input Parameters on the left. The parameters will be passed to the classification 
model and the model will run each time you modify the parameters. 

1- You will see the values of the parameters in the **'User Input Parameters'** section.

2- You will see the classification result under the **'Prediction'** section.

3- You will see the **'prediction propability'** (: the propability that the user input parameters is in one of the 3 classes ==> {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}) 
in the last section
'''

st.sidebar.header('User Input Features')

# st.sidebar.markdown("""
# [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
# """)

# Here, we create a custom function to accept all of the input parameters from the sidebar to create a dictionary that will be
# passed to a Pandas dataframe and show it in the User Input part of the screen
def user_input_features():
    island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
    sex = st.sidebar.selectbox('Sex',('male','female'))
    bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
    bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
    flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
    body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
    data = {'island': island,
            'bill_length_mm': bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex': sex}
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

# Here, we will read the data from the CSV file. Then, drop the target 'species' column from the dataframe. Then, combine the user input dataframe with the
# resulting dataframe (without the target column) in a dataframe
# This will be useful for the encoding phase
penguins_raw = pd.read_csv('penguins_cleaned.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df,penguins],axis=0)

# Here, we get each ordinal feature/ variable in the 'col' list and encode it to 1 and 0. Then, we add it, after encoding to
# the dataframe and delete the original column
encode = ['sex','island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

# Selects only the first row (the user input data)
df = df[:1]

# Displays the user input features
st.subheader('User Input features')

st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

# st.subheader('Class Labels and Their Corresponding Index Numbers are Adelie= 0,Chinstrap= 1, Gentoo= 2}')
# st.write(set(penguins_raw['species']))


st.subheader('Prediction')
penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(penguins_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)