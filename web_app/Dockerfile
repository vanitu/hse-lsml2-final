# FROM pytorch/pytorch:latest
#FROM python:3.9-slim
FROM ultralytics/ultralytics:latest

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    libgl1

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY app.py app.py
# RUN pip install ultralytics

EXPOSE 8501


HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
