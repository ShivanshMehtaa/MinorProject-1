import numpy as np
from PIL import Image,ImageOps
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from keras.models import load_model
import requests
from streamlit_lottie import st_lottie
import webbrowser

lottie_animation1 = 'https://lottie.host/7ae40c73-ab8c-47d1-870a-c94f085098f5/aYp9l9poEg.json'
lottie_animation2='https://lottie.host/ec59b0d4-75eb-4606-bef9-0449a32746e6/84FAuYAtmP.json'

def lottie_loader(url):
    r=requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# loading the saved models
# C:\Users\Shivansh\OneDrive\Desktop\preddisease\saved models\diabetes_model.sav
diabetes_model = pickle.load(open('C:/Users/Shivansh/OneDrive/Desktop/preddisease/saved models/diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open('C:/Users/Shivansh/OneDrive/Desktop/preddisease/saved models/heart_disease_model.sav','rb'))

parkinsons_model = pickle.load(open('C:/Users/Shivansh/OneDrive/Desktop/preddisease/saved models/parkinsons_model.sav', 'rb'))


def classify(image, model, class_names):
    """
    This function takes an image, a model, and a list of class names and returns the predicted class and confidence
    score of the image.

    Parameters:
        image (PIL.Image.Image): An image to be classified.
        model (tensorflow.keras.Model): A trained machine learning model for image classification.
        class_names (list): A list of class names corresponding to the classes that the model can predict.

    Returns:
        A tuple of the predicted class name and the confidence score for that prediction.
    """
    # convert image to (224, 224)
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # convert image to numpy array
    image_array = np.asarray(image)

    # normalize image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # set model input
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # make prediction
    prediction = model.predict(data)
    # index = np.argmax(prediction)
    index = 0 if prediction[0][0] > 0.95 else 1
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score

st.set_page_config(page_title='MediBuddy', page_icon=':health_worker:',layout='wide')

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style/style.css")

# sidebar for navigation
with st.sidebar:
    st.info('Here is the list of the services we provide')
    selected = option_menu('Services',
                           ['Home',
                          'Pneumonia Prediction',
                          'Brain Tumor Detection',
                           'Heart Disease Prediction',
                           'Parkinsons Prediction',
                           'Diabetes Prediction'],
                          icons=['home','','activity','heart','person'])
    
    
# Diabetes Prediction Page
if(selected=='Home'):
    st.title(':health_worker: Medibuddy')
    st.subheader('Medibuddy  is a disease prediction software that takes the report values from the user as input and determines the result based on the top-class DL technologies')

    with st.container():
        left_column, right_column = st.columns(2)
        with left_column:
            st.write("---")
            st.header(':gear: Services we provide')
            st.write(
                """
                Here is the list of all the services our application provides as of now
                - Pneumonia prediction system
                - Brain Tumor detection system
                - Diabetes prediction system
                - Heart Disease prediction system
                - Parkinsons disease prediction system
                """
            )
        with right_column:
            st_lottie(lottie_animation1,height='200',key='medi')
    
    with st.container():
        st.write('---')
        st.header(''':mailbox: Having issues with our results ?
                  Inform us here''')
        contact_form="""
        <form action="https://formsubmit.co/shivanshmehta0000@gmail.com" method="POST">
            <input type='hidden' name="_captcha" value='false'>
            <input type="text" name="name" placeholder="Enter your name here" required>
            <input type="email" name="email" placeholder="Enter your mail address here" required>
            <textarea name='message' placeholder="Please type in your issue here"></textarea>
            <button type="submit">Send</button>
        </form>
    """
        st.markdown(contact_form, unsafe_allow_html=True)

    
if(selected == 'Brain Tumor Detection'):
    st.title('Brain TUmor Detection System')
    st.subheader("""
                Click the button down to go to the desired page for Brain TUmor Detection Service
                 """)
    st.info('this button redirects you to an external website which is fully assured by us')
    if st.button('Brain Tumor Detection'):
        webbrowser.open_new_tab('http://127.0.0.1:5000/')

    with st.container():
        left_two, right_two = st.columns(2)
        with left_two:
            st.write('---')   
            st.subheader(
                """
                :hospital: Here is the List of Hospitals that expretise in the cure of Brain Tumors
                - Shalby Hospital
                - Marble City Hospital
                - Jabalpur Hospital
                """
            )
        with right_two:
            st_lottie(lottie_animation2,height='200',key='medi')

if (selected == 'Diabetes Prediction'):
    
    # page title
    st.title('Diabetes Prediction using ML')
    
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    
    with col2:
        Insulin = st.text_input('Insulin Level')
    
    with col3:
        BMI = st.text_input('BMI value')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if (diab_prediction[0] == 1):
          diab_diagnosis = 'The person is diabetic, please take necessary precautions immediately'
        else:
          diab_diagnosis = 'The person is not diabetic'
        
    st.success(diab_diagnosis)

    with st.container():
        left_two, right_two = st.columns(2)
        with left_two:
            st.write('---')   
            st.subheader(
                """
                :hospital: Here is the List of Hospitals that expretise in the cure of Diabetes 
                - Shalby Hospital
                - Marble City Hospital
                - Jabalpur Hospital
                """
            )
        with right_two:
            st_lottie(lottie_animation2,height='200',key='medi')




# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Sex')
        
    with col3:
        cp = st.text_input('Chest Pain types')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        
     
     
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease, consult a doctor immediately'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)

    with st.container():
        left_two, right_two = st.columns(2)
        with left_two:
            st.write('---')   
            st.subheader(
                """
                :hospital: Here is the List of Hospitals that expretise in the cure of Heart Diseases
                - Shalby Hospital
                - Marble City Hospital
                - Jabalpur Hospital
                """
            )
        with right_two:
            st_lottie(lottie_animation2,height='200',key='medi')
        
    
    

# Parkinson's Prediction Page
if (selected == "Parkinsons Prediction"):
    
    # page title
    st.title("Parkinson's Disease Prediction using ML")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
        
    with col4:
        DDA = st.text_input('Shimmer:DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
        
    
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        
        if (parkinsons_prediction[0] == 1):
          parkinsons_diagnosis = "The person has Parkinson's disease, please take necessary precautions immediately"
        else:
          parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
    st.success(parkinsons_diagnosis)

    with st.container():
        left_two, right_two = st.columns(2)
        with left_two:
            st.write('---')   
            st.subheader(
                """
                :hospital: Here is the List of Hospitals that expretise in the cure of Parkinsons Disease
                - Shalby Hospital
                - Marble City Hospital
                - Jabalpur Hospital
                """
            )
        with right_two:
            st_lottie(lottie_animation2,height='200',key='medi')

if (selected=='Pneumonia Prediction'):
    st.title('Pneumonia Prediction')
    st.header('Please upload a chest X-ray image')

    # upload file
    file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

    # load classifier
    model = load_model('./model/pneumonia_classifier.h5')

    # load class names
    with open('./model/labels.txt', 'r') as f:
        class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
        f.close()

    # display image
    if file is not None:
        image = Image.open(file).convert('RGB')
        st.image(image, use_column_width=True)

        # classify image
        class_name, conf_score = classify(image, model, class_names)

        # write classification
        st.write("## {}".format(class_name))
        # st.write("### score: {}%".format(int(conf_score * 1000) / 10))

        with st.container():
            left_two, right_two = st.columns(2)
            with left_two:
                st.write('---')   
                st.subheader(
                    """
                    :hospital: Here is the List of Hospitals that expretise in the cure of Pneumonia
                    - Shalby Hospital
                    - Marble City Hospital
                    - Jabalpur Hospital
                    """
                )
            with right_two:
                st_lottie(lottie_animation2,height='200',key='medi')





