FROM python:3.10-slim

WORKDIR /mlflow

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir mlflow[extras]

ENV MLFLOW_ENABLE_JOB_SCHEDULER=false

EXPOSE 5000

CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri "sqlite:///mlflow.db" \
    --default-artifact-root "/mlflow/artifacts" \
    --allowed-hosts "*"
