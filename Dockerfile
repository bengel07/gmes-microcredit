FROM python:3.11-slim

# Pour Debian 12 (trixie), utilisez ces packages
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier d'abord requirements.txt pour mieux utiliser le cache Docker
COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel

# Installez d'abord les packages de base
RUN pip install numpy==1.26.4
RUN pip install opencv-python-headless==4.10.0.84  # Headless pour Render

# Pour deepface, installez sans dlib ou avec une version spécifique
RUN pip install deepface==0.0.79 --no-deps || \
    echo "Continuing without some deepface dependencies..."

# Puis installez le reste
RUN pip install -r requirements.txt

COPY . .

# Port pour Render
ENV PORT=10000
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT"]
