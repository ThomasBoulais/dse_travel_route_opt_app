FROM python:3.10-slim

WORKDIR /app

COPY requirements/inference.txt .
RUN pip install --no-cache-dir -r inference.txt

COPY src/inference ./inference
COPY models ./models

CMD ["python", "inference/generate_itinerary.py"]
