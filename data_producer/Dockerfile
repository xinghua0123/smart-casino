FROM python:3.11-slim
WORKDIR /app
COPY data_producer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY data_producer .
COPY casino /app/casino
ENV PYTHONUNBUFFERED=1
CMD ["python", "floor_simulator.py"]
