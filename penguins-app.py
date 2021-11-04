import pandas as pd
import numpy as np
import pickle
import streamlit as st
#list
from sklearn.ensemble import RandomForestClassifier


#source: https://pypi.org/project/streamlit-analytics/
import streamlit_analytics

# We use streamlit_analytics to track the site like in Google Analytics
streamlit_analytics.start_tracking()

# configuring the page and the logo
st.set_page_config(page_title='Mohamed Gabr - House Price Prediction', page_icon ='logo.png', layout = 'wide', initial_sidebar_state = 'auto')


import os
import base64

# the functions to prepare the image to be a hyperlink
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache(allow_output_mutation=True)
def get_img_with_href(local_img_path, target_url):
    img_format = os.path.splitext(local_img_path)[-1].replace('.', '')
    bin_str = get_base64_of_bin_file(local_img_path)
    html_code = f'''
        <a href="{target_url}">
            <img src="data:image/{img_format};base64,{bin_str}" />
        </a>'''
    return html_code


# preparing the layout for the top section of the app
# dividing the layout vertically (dividing the first row)
row1_1, row1_2, row1_3 = st.columns((1, 5, 4))

# first row first column
with row1_1:
    gif_html = get_img_with_href('logo.png', 'https://golytics.github.io/')
    st.markdown(gif_html, unsafe_allow_html=True)

with row1_2:
    # st.image('logo.png')
    st.title("Predicting The Penguin Species Using Artificial Intelligence")
    st.markdown("<h2>A Famous Machine Learning Project (Practical Project for Students)</h2>", unsafe_allow_html=True)

# first row second column
with row1_3:
    st.info(
        """
        ##
        This data product has been prepared to be used as a practical project in the training courses provided by Dr. Mohamed Gabr. Developing the final model required
        many steps following the CRISP-DM methodology. After building the model we used it to predict the Penguin Speciesr type in this application.
        """)








st.write("""
This app predicts the **Penguin** species!
""")


st.write("""We have 3 types as shown in the below image""")

images_list=['Adelie.jpg', 'Chinstrap.jpg', 'Gentoo.jpg']
indices_on_page=['Adelie', 'Chinstrap', 'Gentoo']
st.image(images_list, width=200, caption=indices_on_page)

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
# st.write(penguins_species[prediction])
#
#
predicted_class=penguins_species[prediction]

html_str = f"""
<h3 style="color:lightgreen;">{predicted_class[0]}</h3>
"""

st.markdown(html_str, unsafe_allow_html=True)
st.image(predicted_class[0]+'.jpg', width=200)


st.subheader('Prediction Probability')
st.write(prediction_proba)


st.info("""**Note: ** [The data source is palmerpenguins library]: ** (https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst. the following steps have been applied till we reached the model:

        1- Data Acquisition/ Data Collection (reading data, adding headers)

        2- Data Cleaning / Data Wrangling / Data Pre-processing (handling missing values, correcting data fromat/ data standardization 
        or transformation/ data normalization/ data binning/ Preparing Indicator or binary or dummy variables for Regression Analysis/ 
        Saving the dataframe as ".csv" after Data Cleaning & Wrangling)

        3- Exploratory Data Analysis (Analyzing Individual Feature Patterns using Visualizations/ Descriptive statistical Analysis/ 
        Basics of Grouping/ Correlation for continuous numerical variables/ Analysis of Variance-ANOVA for ctaegorical or nominal or 
        ordinal variables/ What are the important variables that will be used in the model?)

        4- Model Development (Single Linear Regression and Multiple Linear Regression Models/ Model Evaluation using Visualization)

        5- Polynomial Regression Using Pipelines (one-dimensional polynomial regession/ multi-dimensional or multivariate polynomial 
        regession/ Pipeline : Simplifying the code and the steps)

        6- Evaluating the model numerically: Measures for in-sample evaluation (Model 1: Simple Linear Regression/ 
        Model 2: Multiple Linear Regression/ Model 3: Polynomial Fit)

        7- Predicting and Decision Making (Prediction/ Decision Making: Determining a Good Model Fit)

        8- Model Evaluation and Refinement (Model Evaluation/ cross-validation score/ over-fitting, under-fitting and model selection)

""")

with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

footer = """<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Published By: <a href="https://golytics.github.io/" target="_blank">Dr. Mohamed Gabr</a></p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)

streamlit_analytics.stop_tracking(unsafe_password="forward1")