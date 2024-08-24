from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
import joblib
import os
import shutil
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# Define hyperparameters for each model for grid search
from sklearn.preprocessing import StandardScaler

crops = ['apple', 'arecanut', 'ashgourd', 'banana', 'barley', 'beetroot',
       'bittergourd', 'blackgram', 'blackpepper', 'bottlegourd',
       'brinjal', 'cabbage', 'cardamom', 'carrot', 'cashewnuts',
       'cauliflower', 'coffee', 'coriander', 'cotton', 'cucumber',
       'drumstick', 'garlic', 'ginger', 'grapes', 'horsegram',
       'jackfruit', 'jowar', 'jute', 'ladyfinger', 'maize', 'mango',
       'moong', 'onion', 'orange', 'papaya', 'pineapple', 'pomegranate',
       'potato', 'pumpkin', 'radish', 'ragi', 'rapeseed', 'rice',
       'ridgegourd', 'sesamum', 'soyabean', 'sunflower', 'sweetpotato',
       'tapioca', 'tomato', 'turmeric', 'watermelon', 'wheat']
states = ['andaman and nicobar islands', 'andhra pradesh',
       'arunachal pradesh', 'assam', 'bihar', 'chandigarh',
       'chhattisgarh', 'dadra and nagar haveli', 'goa', 'gujarat',
       'haryana', 'himachal pradesh', 'jammu and kashmir', 'jharkhand',
       'karnataka', 'kerala', 'madhya pradesh', 'maharashtra', 'manipur',
       'meghalaya', 'mizoram', 'nagaland', 'odisha', 'puducherry',
       'punjab', 'rajasthan', 'sikkim', 'tamil nadu', 'telangana',
       'tripura', 'uttar pradesh', 'uttarakhand', 'west bengal']
new_scaler = joblib.load('C:/Users/RAMU GOPI/AA-Major Project/all models/ML models/ml_saved_models/scalerV1.0.pkl')
param_grid = {
    'LogisticRegression': {'max_iter': [1000]},
    'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'DecisionTreeClassifier': {'max_depth': [None, 5, 10, 15]},
    'KNeighborsClassifier': {'n_neighbors': [3, 5, 7]},
    'GaussianNB': {},  # No hyperparameters for GaussianNB
    'RandomForestClassifier': {'n_estimators': [50, 100, 200]},
    'VotingClassifier': {},  # Hyperparameters are set inside VotingClassifier definition
    'BaggingClassifier': {'n_estimators': [10, 50, 100]},
    'AdaBoostClassifier': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]},
    'GradientBoostingClassifier': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]}
}

# List of models with their respective names
models = [
    ('LogisticRegression', LogisticRegression()),
    ('SVM', SVC()),
    ('DecisionTreeClassifier', DecisionTreeClassifier()),
    ('KNeighborsClassifier', KNeighborsClassifier()),
    ('GaussianNB', GaussianNB()),
    ('RandomForestClassifier', RandomForestClassifier()),
    ('VotingClassifier', VotingClassifier([('Random Forest', RandomForestClassifier()),
                                           ('SVM', SVC()),
                                           ('Logistic Regression', LogisticRegression(max_iter=1000))])),
    ('BaggingClassifier', BaggingClassifier()),
    ('AdaBoostClassifier', AdaBoostClassifier()),
    ('GradientBoostingClassifier', GradientBoostingClassifier())
]
# List to store models supporting predict_proba
models_with_proba = []

import tensorflow as tf


def data_for_cnn(new_data1):
    #new_data1[0] = states.index(new_data1[0])
    new_scaler = joblib.load('C:/Users/RAMU GOPI/AA-Major Project/all models/ML models/ml_saved_models/scalerV1.0.pkl')
    new_data1 = new_scaler.transform([new_data1])
    return new_data1
def data_for_ann(new_data1):
    #new_data1[0] = states.index(new_data1[0])
    new_scaler = joblib.load('C:/Users/RAMU GOPI/AA-Major Project/all models/ML models/ml_saved_models/scalerV1.0.pkl')
    new_data1 = new_scaler.transform([new_data1])
    return new_data1
def data_for_LSTM(new_data1):
    #new_data1[0] = states.index(new_data1[0])
    new_scaler = joblib.load('C:/Users/RAMU GOPI/AA-Major Project/all models/ML models/ml_saved_models/scalerV1.0.pkl')
    new_data1 = np.array(new_scaler.transform([new_data1]))
    new_data1 = new_data1.reshape(new_data1.shape[0], 1,new_data1.shape[1])
    return new_data1
def data_for_voting_dl(new_data1):
    #new_data1[0] = states.index(new_data1[0])
    new_scaler = joblib.load('C:/Users/RAMU GOPI/AA-Major Project/all models/ML models/ml_saved_models/scalerV1.0.pkl')
    new_data1 = new_scaler.transform([new_data1])
    new_data1 = [new_data1,new_data1]
    return new_data1
def prediction_from_ann(new_data):
    ann_model = tf.keras.models.load_model("C:/Users/RAMU GOPI/AA-Major Project/all models/DL models/ANN_V10.keras")
    pred = ann_model.predict(new_data)
    return crops[np.argmax(pred,axis = 1)[0]],pred[0]
def prediction_from_cnn(new_data):
    cnn_model = tf.keras.models.load_model("C:/Users/RAMU GOPI/AA-Major Project/all models/DL models/CNN_V10.keras")
    pred = cnn_model.predict(new_data)
    return crops[np.argmax(pred,axis = 1)[0]],pred[0]
def prediction_from_lstm(new_data):
    lstm_model = tf.keras.models.load_model("C:/Users/RAMU GOPI/AA-Major Project/all models/DL models/LSTM_V11.keras")
    pred = lstm_model.predict(new_data)
    return crops[np.argmax(pred,axis = 1)[0]],pred[0]


import joblib
def create_lstm_model():
    LSTM_V10 = Sequential()
    LSTM_V10.add(LSTM(256, input_shape=(1, X_train_reshaped.shape[2]), activation='relu'))
    #model.add(Dense(128,activation = 'relu'))
    LSTM_V10.add(Dense(53, activation='softmax'))  # Adjust output units and activation for multiclass
    # Compile the model
    LSTM_V10.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return LSTM_V10



# Function to create GRU model
def create_gru_model():
    GRU_V10 = Sequential()
    GRU_V10.add(GRU(256, input_shape=(1, X_train_reshaped.shape[2]), kernel_initializer=he_normal(), activation=LeakyReLU(alpha=0.03), return_sequences=True))
    GRU_V10.add(GRU(128, activation=LeakyReLU(alpha=0.03)))  # Additional GRU layer
    #model2.add(Dense(64, activation=LeakyReLU(alpha=0.03)))
    GRU_V10.add(Dense(53, activation='sigmoid'))
    GRU_V10.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.03), metrics=['accuracy'])
    return GRU_V10
def create_ann_model():
    ANN_V10 = Sequential()
    # Add layers
    ANN_V10.add(Dense(128, input_shape=(1, X_train_reshaped.shape[2]), activation='relu'))
    ANN_V10.add(Dense(64, activation='relu'))
    ANN_V10.add(Dense(32, activation='relu'))
    ANN_V10.add(Dense(53, activation='softmax'))  # Softmax for multi-class classification
    # Compile the model
    ANN_V10.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return ANN_V10
voting_classifier = joblib.load('C:/Users/RAMU GOPI/AA-Major Project/all models/DL models/voting_classifier_100epV12.pkl')
def prediction_from_voting_dl(new_data):
    pred = voting_classifier.predict_proba(new_data)
    return crops[np.argmax(pred)],pred[0]


# Check and add models supporting predict_proba to the list
for name, model in models:
    if hasattr(model, 'predict_proba'):
        models_with_proba.append((name, model))

# Access models_with_proba for further use
# print(models_with_proba)
dl_models_with_proba = [(data_for_ann,prediction_from_ann),(data_for_cnn,prediction_from_cnn),
                        (data_for_LSTM,prediction_from_lstm),(data_for_voting_dl,prediction_from_voting_dl)]
def ml_predict(new_data):
    #new_data[0]  =states.index(list(new_data)[0])
    probs = []
    new_data[0] = states.index(new_data[0])
    new_data = new_scaler.transform([new_data])
    new_data = new_data
    # Loop through each model and accumulate prediction probabilities
    for name, model_ in models_with_proba:
        #new_data = ['andaman and nicobar islands',100,40,140,5.86,1925.68,27.0]
        model = joblib.load(f'C:/Users/RAMU GOPI/AA-Major Project/all models/ML models/ml_saved_models/hyper/{name}_hyper_22.pkl')
        probabilities = model.predict_proba(new_data)[0]
        probs.append(probabilities)
    probs = (sum(probs)/len(models_with_proba))*100
    good = {}
    optional = {}
    for i in range(5):
        if i ==0:
            print("Most Probable Crop:", crops[np.argmax(probs)])
            good[crops[np.argmax(probs)]]= max(probs)
        else:
            print("optional crop ",i," :",crops[np.argmax(probs)])
            optional[crops[np.argmax(probs)]] = max(probs)
        print("accuracy : ", max(probs))
        i = np.argmax(probs)
        probs[i] = 0.0
    return good, optional

def dl_predict(new_data_new):
    probs = []
    new_data_new[0] = states.index(new_data_new[0])
    for create_data,model in dl_models_with_proba:
        data = create_data(new_data_new)
        probabilities = model(data)
        probs.append(list(probabilities[1]))
    probs3 = (sum(np.array(probs))/5)*100
    good = {}
    optional = {}
    for i in range(5):
        if i ==0:
            print("Most Probable Crop:", crops[np.argmax(probs3)]) 
            good[crops[np.argmax(probs3)]]= max(probs3)
        else:
            print("optional crop ",i," :",crops[np.argmax(probs3)])
            optional[crops[np.argmax(probs3)]] = max(probs3)
        print("accuracy : ", max(probs3))
        i = np.argmax(probs3)
        probs3[i] = 0
    return good, optional
# new_data = ["assam",120,60,65,6.12,2169.32,23.736364]
# dl_predict(new_data)


def ml_dl_predict(new_data):
    #new_data[0]  =states.index(list(new_data)[0])
    probs = []
    new_data_new = new_data
    new_data[0] = states.index(new_data[0])
    new_data_new = new_data
    new_data = new_scaler.transform([new_data])
    new_data = new_data
    # Loop through each model and accumulate prediction probabilities
    for name, model_ in models_with_proba:
        #new_data = ['andaman and nicobar islands',100,40,140,5.86,1925.68,27.0]
        model = joblib.load(f'C:/Users/RAMU GOPI/AA-Major Project/all models/ML models/ml_saved_models/hyper/{name}_hyper_22.pkl')
        probabilities = model.predict_proba(new_data)[0]
        probs.append(probabilities)
#     new_data_new[0] = states.index(new_data_new[0])
    for create_data,model in dl_models_with_proba:
        data = create_data(new_data_new)
        probabilities = model(data)
        probs.append(list(probabilities[1]))
    probs3 = (sum(np.array(probs))/(len(models_with_proba)+len(dl_models_with_proba)))*100
    good = {}
    optional = {}
    for i in range(5):
        if i ==0:
            print("Most Probable Crop:", crops[np.argmax(probs3)])
            good[crops[np.argmax(probs3)]]= max(probs3)
        else:
            print("optional crop ",i," :",crops[np.argmax(probs3)])
            optional[crops[np.argmax(probs3)]] = max(probs3)
        print("accuracy : ", max(probs3))
        i = np.argmax(probs3)
        probs3[i] = 0
    return good, optional
# new_data = ["jammu and kashmir",60,30,30,6.11,293.36,14.700000]
# ml_dl_predict(new_data)

import streamlit as st
import requests

# Function to fetch weather data
def fetch_weather(city_name):
    API_KEY = '0146e9f5467b6ada89e5092a83f0d7fb'  
    url = f'http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={API_KEY}&units=metric'

    response = requests.get(url)
    data = response.json()

    if data['cod'] == 200:
        temperature = data['main']['temp']
        return temperature
    else:
        return None


# Streamlit UI


import streamlit as st

def recommendation_page(states):
    st.title('Crop Recommendation')
    placeholder1 = st.empty()

    # DL predictions
    placeholder1.markdown("&nbsp;")
    # Sidebar for inputs
    st.sidebar.subheader('Select Inputs:')
    states = states  # Placeholder for state dropdown
    #state_input = "jammu and kashmir"
    state_input = st.sidebar.selectbox('Select State', states)
    param_1 = st.sidebar.text_input('N', value='60')
    param_2 = st.sidebar.text_input('P', value='30')
    param_3 = st.sidebar.text_input('K', value='30')
    param_4 = st.sidebar.text_input('Ph', value='6.11')
    param_5 = st.sidebar.text_input('rainfall', value='293.36')
    
    temperature = np.abs(fetch_weather(state_input))

    if temperature is not None:
        # Pop-up message for success
        st.success(f"Current temperature in {state_input} is {temperature}°C")
        param_6 = st.sidebar.text_input('Temperature', value=f'{temperature}')
    else:
        # Pop-up message for failure
        st.error(f"Failed to fetch weather data for {state_input}")
        param_6 = st.sidebar.text_input('Temperature', value='{14.7}')
    # Convert text inputs to numeric values for predictions
    param_1 = float(param_1)
    param_2 = float(param_2)
    param_3 = float(param_3)
    param_4 = float(param_4)
    param_5 = float(param_5)
    param_6 = float(param_6)
    tech = st.selectbox('Select Technique', ['All','ML-DL',"ML","DL"])
    # Button to trigger predictions
    if st.sidebar.button('Predict'):
        st.subheader('Predictions')
        
        # Creating three columns layout for predictions of three models
        col1, col2, col3 = st.columns(3)
        # ML-DL predictions in the third column
        if tech in ['All','ML-DL']:
            with col1:
                st.subheader('ML-DL Predictions')
                # Perform predictions based on user input
                good, optional = ml_dl_predict([state_input, param_1, param_2, param_3, param_4, param_5, param_6])

                # Display ML-DL predictions
                st.write(f"Most Probable Crop:<H7><font color='green'>{list(good.keys())[0]}</font></H7>", unsafe_allow_html=True)#'Most Probable Crop:', list(good.keys())[0])
                st.write('Accuracy:', round(list(good.values())[0],2),"%")
                st.write(" ")
                st.write('Optional Crops:')
                for k, v in optional.items():
                    st.write(f"{k}, Accuracy: {round(v,2)}","%")
        # ML predictions in the first column
        if tech in ['All','ML']:
            with col2:
                st.subheader('ML Predictions')
                # Perform predictions based on user input
                good, optional = ml_predict([state_input, param_1, param_2, param_3, param_4, param_5, param_6])

                # Display ML predictions
                st.write(f"Most Probable Crop:<H7><font color='green'>{list(good.keys())[0]}</font></H7>", unsafe_allow_html=True)
                st.write('Accuracy:', round(list(good.values())[0],2),"%")
                st.write(" ")
                st.write('Optional Crops:')
                for k, v in optional.items():
                    st.write(f"{k}, Accuracy: {round(v,2)}","%")

        # DL predictions in the second column
        if tech in ['All','DL']:
            with col3:
                st.subheader('DL Predictions')
                # Perform predictions based on user input
                good, optional = dl_predict([state_input, param_1, param_2, param_3, param_4, param_5, param_6])

                # Display DL predictions
                st.write(f"Most Probable Crop:<H7><font color='green'>{list(good.keys())[0]}</font></H7>", unsafe_allow_html=True)#list(good.keys())[0])
                st.write('Accuracy:', round(list(good.values())[0],2),"%")
                st.write(" ")
                st.write('Optional Crops:')
                for k, v in optional.items():
                    st.write(f"{k}, Accuracy: {round(v)}","%")
#Disease detect
diseases = ['Pepper_bell_Bacterial_spot',
 'Pepper_bell_healthy',
 'Potato_Early_blight',
 'Potato_Late_blight',
 'Potato_healthy',
 'Tomato_Late_blight',
 'Tomato_Tomato_mosaic_virus',
 'Tomato Leaf Mold',
 'Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato_Target_Spot',
 'Tomato Tomato_YellowLeaf Curl Virus',
 'Tomato_Septoria_leaf_spot',
'Tomato_healthy']
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import streamlit as st
from keras.preprocessing import image
import numpy as np
import pandas as pd


# Load the trained model and pesticide data
load_model2 = tf.keras.models.load_model('C:/Users/RAMU GOPI/AA-Major Project/Crop protectio/disease_detect_V10')
load_model2.load_weights('C:/Users/RAMU GOPI/AA-Major Project/Crop protectio/disease_detect_weights_V10')
pesti = pd.read_csv("C:/Users/RAMU GOPI/AA-Major Project/Crop protectio/pesticides.csv")


def predict_disease(image_file,load_model2):
    # Load and preprocess the image for prediction
    img = image.load_img(image_file, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale to match the training data preprocessing

    prediction = load_model2.predict(img_array)

    # Decode the prediction
    predicted_class = np.argmax(prediction)
    predicted_class_name = diseases[predicted_class]

    # Get pesticide details based on predicted class
    pred = dict(pesti[pesti['Disease'] == predicted_class_name])
    
    return predicted_class_name, pred

def protection_page(load_model2):
    st.title('Crop Protection Management')

    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.sidebar.button('Predict')
        st.sidebar.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        st.sidebar.image(uploaded_file, caption='Sample Image', use_column_width=True)
        predicted_class, pred = predict_disease(uploaded_file,load_model2)

        st.write(f"**<H4>Predicted Disease:** <font color='green'>{predicted_class}</font></H4>", unsafe_allow_html=True)
        st.write(f"**<H5>Description:**</H5>\n{pred['Description'].values[0]}", unsafe_allow_html=True)
        st.markdown(f"**<H5>Symptoms:</H5>**\n{pred['Symptoms'].values[0]}", unsafe_allow_html=True)
        st.markdown(f"**<H5>Pest Management (Organic/Non-Organic):</H5>**\n{pred['Management (Organic/Non-Organic)'].values[0]}", unsafe_allow_html=True)
        st.markdown(f"**<H5>Refer here:</H5>**\n{pred['Website Links'].values[0]}", unsafe_allow_html=True)

    else:
        new_image_path = 'C:/Users/RAMU GOPI/AA-Major Project/Crop protectio/PlantVillage1/Tomato__Tomato_YellowLeaf__Curl_Virus/0a1e2ed0-619c-43da-8c47-f8000a252954___UF.GRC_YLCV_Lab 03060.jpg'
        img = image.load_img(new_image_path, target_size=(256, 256))#Crop protectio\PlantVillage1\
        st.sidebar.image(img, caption='Sample Image', use_column_width=True)
        if st.sidebar.button('Predict'):
            st.write("No image uploaded. Showing default sample image <H7><font color='red'>Tomato__Tomato_YellowLeaf__Curl_Virus</font></H7>  prediction.", unsafe_allow_html=True)

            
            

            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Rescale to match the training data preprocessing

            prediction = load_model2.predict(img_array)

            predicted_class = np.argmax(prediction)
            predicted_class_name = diseases[predicted_class]

            pred = dict(pesti[pesti['Disease'] == predicted_class_name])

            st.write(f"**<H4>Predicted Disease:** <font color='green'>{predicted_class_name}</font></H4>", unsafe_allow_html=True)
            st.markdown(f"**<H5>Description:</H5>**\n{pred['Description'].values[0]}", unsafe_allow_html=True)
            st.markdown(f"**<H5>Symptoms:</H5>**\n{pred['Symptoms'].values[0]}", unsafe_allow_html=True)
            st.markdown(f"**<H5>Pest Management (Organic/Non-Organic):</H5>**\n{pred['Management (Organic/Non-Organic)'].values[0]}", unsafe_allow_html=True)
            st.markdown(f"**<H5>Refer here:</H5>**\n{pred['Website Links'].values[0]}", unsafe_allow_html=True)


def main(states):
    page = st.sidebar.selectbox("Choose a page", ("Recommendation", "Protection"))

    if page == "Recommendation":
        recommendation_page(states)
    else:
        protection_page(load_model2)

if __name__ == '__main__':
    main(states)
