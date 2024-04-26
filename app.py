import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

data = pd.read_csv('creditcard.csv')

legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# train logistic regression model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# Streamlit app
st.title("Credit Card Fraud Detection Model")
st.write("Enter the features of dataset to check if the transaction is legitimate or fraudulent:")

input_df = st.text_input('Input All features')
input_df_lst = input_df.split(',')
submit = st.button("Submit")

if submit:
    # get input feature values
    features = np.array(input_df_lst, dtype=np.float64)
    # make prediction
    prediction = model.predict(features.reshape(1,-1))
    # display result
    if prediction[0] == 0:
        st.write("Legitimate transaction")
    else:
        st.write("Fraudulent transaction")
