"""Web app for plant pathology model.

Based on Vladimir Iglovikov's example code, here:
https://ternaus.blog/tutorial/2020/08/28/Trained-model-what-is-next.html#vii-20-min-create-webapp
"""

import numpy as np
import streamlit as st
from PIL import Image
from plant_pathology.pretrained_models import get_model


st.set_option("deprecation.showfileUploaderEncoding", False)


@st.cache
def cached_model():
    return get_model("resnet18_2021-04-07")


model = cached_model()

st.title("Classify leaf diseases")

file = st.file_uploader("Upload an image...")

if file is not None:
    st.image(file)
    st.write("Classifying disease...")
    preds = model.predict(file)
    st.write(preds)
