FROM python:3.10-slim

# System deps for dlib / face-recognition
RUN apt-get update && apt-get install -y cmake libboost-all-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]
