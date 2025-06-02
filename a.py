import pickle
import streamlit as st
import numpy as np

with open('model.pickle', 'rb') as f:
  model = pickle.load(f)
st.title('Sales Prediction')
c1,c2 = st.columns(2)
X = []
with c1:
    X.append(st.number_input('TV'))
with c2:
    X.append(st.number_input('Radio'))
e = st.button('Predict', use_container_width = True)
if e and X == [0,0]:
    st.error('Please input TV or Radio ads')
elif e:
    X = np.array(X).reshape(1,-1)
    y_pred = model.predict(X)
    st.success(f"Sale Prediction: {y_pred}", icon = "💵")