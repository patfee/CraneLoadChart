import streamlit as st
import pandas as pd

st.title("Crane Load Chart Viewer")

uploaded_file = st.file_uploader("Upload your CraneData.xlsx")
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("Data Preview:", df.head())
    st.line_chart(df.set_index(df.columns[0]))
