# 1. Utiliser une image Python officielle
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

ENV PORT=5000

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]