import os
import streamlit as st
import config as config

def sidebar():
  st.sidebar.page_link('app.py', label='Homepage')
  st.sidebar.page_link('pages/Analyse_Data.py', label='Data Analysis')

st.title(f'Homepage')
st.markdown('Vũ Lê Đức Thiện - CSA08')

sidebar()
