import streamlit as st
from mediapipeapp import SquatCounter

st.set_page_config()
st.title("Squatcounter")
st.button(on_click=SquatCounter(), label='start cam')
