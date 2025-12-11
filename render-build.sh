#!/bin/bash

# Mettre à jour le système
sudo apt-get update

# Installer les dépendances système nécessaires pour dlib
sudo apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    python3-dev \
    python3-pip \
    python3-venv

# Créer un environnement virtuel Python (optionnel mais recommandé)
python3 -m venv .venv
source .venv/bin/activate

# Mettre à jour pip
pip install --upgrade pip setuptools wheel

# Installer dlib depuis les sources
pip install dlib==19.24.1

# Installer face-recognition et autres dépendances
pip install face-recognition
pip install -r requirements.txt
