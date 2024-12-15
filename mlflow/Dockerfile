FROM ghcr.io/mlflow/mlflow:v2.19.0
WORKDIR /mldata
# RUN mkdir -p /mldata
# RUN mkdir -p /mldata/artifacts
# RUN mkdir -p /mldata/registry
# ENV MLFLOW_BACKEND_STORE_URI=/mldata/mlruns
# ENV MLFLOW_REGISTRY_STORE_URI=/mldata/registry
# ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=/mldata/artifacts
ENTRYPOINT ["mlflow", "server","-h 0.0.0.0", "-p 5001","--dev"]

