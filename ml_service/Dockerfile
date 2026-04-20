FROM python:3.11-slim

RUN apt-get update && apt-get install -y postgresql-client && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Train models at build time, then run inference loop at runtime
RUN python train_model.py

ENV PYTHONUNBUFFERED=1
CMD ["python", "inference.py"]
