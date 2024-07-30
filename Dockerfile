FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["sh", "-c", "pytest --cov=main --cov-report=term -v && uvicorn main:app --host 0.0.0.0 --port 8000"]
