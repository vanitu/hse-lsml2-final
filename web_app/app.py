import PIL.Image
import streamlit as st
# import pandas as pd
# import numpy as np

import mlflow
from ultralytics import YOLO
import PIL
from tempfile import NamedTemporaryFile

MLFLOW_SERVER_URL = 'http://mlflow:5001/'
experiment_name = 'damage_detection_yolo'
client = mlflow.tracking.MlflowClient(MLFLOW_SERVER_URL)

mlflow.set_tracking_uri(MLFLOW_SERVER_URL)
experiment = mlflow.set_experiment(experiment_name)

@st.cache_resource
def load_model(experiment_name):
    version = client.get_model_version_by_alias(experiment_name, "production")
    model = YOLO(mlflow.artifacts.download_artifacts(version.source))
    return version,model

@st.dialog("Model Data")
def model_data(model):
    st.write(model)

st.set_page_config(layout="wide")

version, model = load_model(experiment_name)

st.title("Damage Detection")
with st.sidebar:
    st.subheader("Model Version")
    ("Model:",version.name,f' v{version.version}')
    ("Stage:",version.current_stage)
    ("mAP50:",version.tags["mAP50"])
    if st.button("Model"):
         model_data(version)

image_input_container = st.empty()
image_file = image_input_container.file_uploader("Upload Damaged Vehicle modle", type=['png', 'jpg'])

if image_file is not None:
    image_input_container.empty()
    original, predicted = st.columns(2)
    with original:
        with st.expander("Original Image", expanded=True):
            st.image(image_file, width=512)
        # with st.container(border = True):
        #     st.subheader("Original Image")
            
    with predicted:
        with st.expander("Detected Damages", expanded=True):
            with st.spinner("Working....please wait."):
                with NamedTemporaryFile(suffix='.jpg') as temp:
                    temp.write(image_file.read())
                    temp.seek(0)
                    predicted = model(temp.name)[0]
                    plotted_BGR  = predicted.plot(conf=True)
                    im_rgb = PIL.Image.fromarray(plotted_BGR[..., ::-1])
                    st.image(im_rgb, width=512)