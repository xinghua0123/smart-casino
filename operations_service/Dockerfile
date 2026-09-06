FROM python:3.11-slim
WORKDIR /app
COPY operations_service/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY casino /app/casino
COPY operations_service /app/operations_service
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "operations_service.server"]
