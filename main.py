import streamlit as st
import pandas as pd
from os import path
import pickle

df = pd.read_csv(path.join("data", "auto-mpg.csv"))

st.write("auto-mpg.csv dataset:")
st.write(df)

st.line_chart(df[["mpg", "weight"]])

st.scatter_chart(df[["mpg", "weight"]])

# st.map(14.83165, 143.47531)

x = st.slider("x", min_value=0, max_value=100, value=25)
st.title("{} squared is {}".format(x, x * x))

st.title("Flower Classification")
model = pickle.load(open(path.join("model", "lr_model.pkl"), "rb"))
st.write(model.predict([[1,2,3,4]]))