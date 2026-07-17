FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app/Model-Deployment

EXPOSE 8080

CMD ["python", "app.py"]