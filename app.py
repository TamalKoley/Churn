import streamlit as st;
import numpy as np;
import tensorflow as tf;
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder;
import pandas as pd;
import pickle;

model=tf.keras.models.load_model('model.h5');
with open('label_encoder_gender.pkl','rb') as file:
    le_gen=pickle.load(file);
with open('label_encoder_geography.pkl','rb') as file:
    ohe_geo=pickle.load(file);
with open('scalar.pkl','rb') as file:
    scalar=pickle.load(file);

st.title('Customer Churn Prediction')

geography=st.selectbox('Geography',ohe_geo.categories_[0]);
gender=st.selectbox('Gender',le_gen.classes_);
age=st.slider('Age',18,92);
balance=st.number_input('Balance');
credit_score=st.number_input('Credit Score');
estimated_salary=st.number_input('Estimated Salary');
tenure=st.slider('Tenure',0,10);
num_of_products=st.slider('Num Of Products',1,4);
has_cr_card=st.selectbox('Has Credit Card',[0,1]);
is_active_member=st.selectbox('Is Active Member',[0,1]);


input_data={
    'CreditScore': credit_score,
    'Geography' : geography,
    'Gender' : gender,
    'Age' : age,
    'Tenure' : tenure,
    'Balance' : balance,
    'NumOfProducts' : num_of_products,
    'HasCrCard' : has_cr_card,
    'IsActiveMember' : is_active_member,
    'EstimatedSalary' : estimated_salary
}

input_data=pd.DataFrame([input_data]);
input_data['Gender']=le_gen.transform(input_data['Gender']);
geo_encoded=ohe_geo.transform([input_data['Geography']]).toarray();
input_data_df=pd.concat([input_data,pd.DataFrame(geo_encoded,columns=ohe_geo.get_feature_names_out(['Geography']))],axis=1);
input_data_df.drop(columns=['Geography'],inplace=True);
input_data_df_scaled=scalar.transform(input_data_df);
prediction=model.predict(input_data_df_scaled);
prediction_proba=prediction[0][0]

if prediction_proba > 0.5:
    st.write("The Customer is likely going to Churn" ,prediction_proba);
else:
    st.write("The Customer is likely not going to Churn",prediction_proba);






