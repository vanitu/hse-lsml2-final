# Vehicles Damages Detection

**HSE MDS LSML2 Final Project**

## Description

Service allow to detect damages of the vehicle using CV Yolo finetuned model.
Preptrained Yolo11n from Ultralitics was finetuned on damages dataset for several epochs.
Best model was validated and deployed as web application service

**Dataset**

Car Damages Dataset from Roboflow
https://universe.roboflow.com/claimoo-52a5r/car-damage-detection-20na7/dataset/6

**MLOps**

MLflow was used to store and operate models and experiments.
Yolo11n model from Ultralitics was trained for 10,20,30,40 epoch.
Best model was autoselected by mAP50 metric, tested and marked as production

**WebApp**

Streamlit Application developed for demonstraction purposes. 
Webapp upon startup loads latest production model from MLFlow server, cache it and use for inference

**Deployment**

Full solution deployed using Docker-Compose. 
It will run MLFlow container and WebApp container

**Pocess and code**

Whole process may be found at project.ipynb file

## Project Structure

```{code}
root
│   README.md
│   requirements.txt    
│   project.ipynb # code
|   docker-compose.yaml 
|   damage_dataset.yaml # Yolo Dataset description
|
└───mlflow
|   │   Dockerfile # MLFlow server 
|   |   requirements.txt
│   │   data # Mounted as volume inside docker to store MLFlow data 
│   
└───web_app
|   │   app.py # Streamlit web application
|   |   Dockerfile
|   |   requirements.txt
|
└───screenshots # Screenshots 
```
## MLFlow

**MlFlow runs**
![runs](screenshots\mlflow_runs.png)

**MlFlow runs**
![metrics](screenshots\mlflow_run_mterics_charts.png)

**MlFlow Pr-Recall**
![pr-recall](screenshots\pr-rec-curve.png)

**MlFlow models**

![Models](screenshots\mlflow_models.png)

**MlFlow training**

![Training](screenshots\training_process.png)

## WebApplication

**Landing Page**
![WebAppLanding](screenshots\webapp_landing.png)

**Processing Page**
![WebAppLanding](screenshots\webapp_working.png)

**Inference Result**
![WebAppLanding](screenshots\webapp_results.png)