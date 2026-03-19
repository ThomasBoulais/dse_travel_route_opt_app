FROM python:3.10-slim

WORKDIR /app

COPY requirements/ui.txt .
RUN pip install --no-cache-dir -r ui.txt

COPY src ./src
COPY configs ./configs

ENV PYTHONPATH="/app"

EXPOSE 8501

CMD ["streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
