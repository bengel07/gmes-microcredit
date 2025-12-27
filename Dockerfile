FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

COPY . .

# Port par défaut si $PORT n'est pas défini
ENV PORT=10000
EXPOSE $PORT

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]
