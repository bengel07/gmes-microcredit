FROM python:3.11-slim

# Pour Render, installez uniquement les dépendances essentielles
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier d'abord requirements.txt pour mieux utiliser le cache Docker
COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel

# Installez les packages un par un pour éviter les problèmes
RUN pip install numpy==1.26.4
RUN pip install opencv-python-headless==4.10.0.84  # Utilisez headless pour Render
RUN pip install tensorflow==2.13.0

# Pour deepface, essayez sans dlib ou avec une version alternative
RUN pip install deepface==0.0.79 --no-deps || \
    echo "Skipping deepface full installation..."

# Puis installez le reste
RUN pip install -r requirements.txt

COPY . .

# Port par défaut pour Render
CMD ["gunicorn", "--workers=2", "--bind=0.0.0.0:$PORT", "app:app"]
