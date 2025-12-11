from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from utils.notifications import notification_manager
import os
from dotenv import load_dotenv
from routes.auth import auth_bp  # ✅ Import corrigé
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename

from PIL import Image
import io
import base64

from deepface import DeepFace
import mediapipe as mp
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
import warnings
from pkg_resources import PkgResourcesDeprecationWarning
warnings.filterwarnings("ignore", category=PkgResourcesDeprecationWarning)
from models import Groupe  # Import direct du modèle
load_dotenv()  # Charge les variables d'environnement

# Configuration de l'application
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'gmes-microcredit-2024')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///gmes.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# Configuration des uploads
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
# Enregistre le Blueprint
app.register_blueprint(auth_bp, url_prefix="/auth")

# Initialisation des extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'connexion'


mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)
mp_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

def detect_face(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = mp_face.process(rgb)
    return result.detections

def is_fake_image(img):
    """Détection anti-fake : photo d’écran / imprimée / IA"""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = mp_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return True

    blur = cv2.Laplacian(img, cv2.CV_64F).var()
    brightness = np.mean(img)

    if blur < 35:  # trop flou → photocopie ou écran
        return True

    if brightness < 30:  # trop sombre → suspect
        return True

    return False


# ==================== MODÈLES COMPLETS ====================

class User(UserMixin, db.Model):
    __tablename__ = 'users'



    id = db.Column(db.Integer, primary_key=True)

    # Champs communs à tous les utilisateurs
    username = db.Column(db.String(80), unique=True, nullable=True)
    email = db.Column(db.String(120), unique=True)
    password_hash = db.Column(db.String(255))
    fonction = db.Column(db.String(50))  # caissier, conseiller, etc.
    role = db.Column(db.String(20), default='client')  # client, employe, admin, superviseur
    statut = db.Column(db.String(20), default='actif')  # 'actif', 'en_attente', 'inactif'
    approuve_par = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)  # Admin qui a approuvé
    date_approbation = db.Column(db.DateTime, nullable=True)
    permissions = db.Column(db.Text)  # Stocke les permissions en JSON
    nom = db.Column(db.String(100))
    prenom = db.Column(db.String(100))
    telephone = db.Column(db.String(20))
    date_creation = db.Column(db.DateTime, default=datetime.utcnow)
    groupe_id = db.Column(db.Integer, db.ForeignKey('groupes.id'), nullable=True)

    # Champs spécifiques aux clients
    code_client = db.Column(db.String(20), unique=True, nullable=True)
    adresse = db.Column(db.Text)
    cin = db.Column(db.String(50), unique=True, nullable=True)
    date_naissance = db.Column(db.DateTime, nullable=True)
    profession = db.Column(db.String(100))
    revenu_mensuel = db.Column(db.Float, default=0)
    date_inscription = db.Column(db.DateTime, default=datetime.utcnow)
    groupe_id = db.Column(db.Integer, db.ForeignKey('groupes.id'), nullable=True)

    # Nouveaux champs
    depenses_mensuelles = db.Column(db.Float, default=0)
    capacite_remboursement = db.Column(db.Float, default=0)
    photo_id = db.Column(db.String(255))
    photo_selfie = db.Column(db.String(255))
    verification_faciale = db.Column(db.Boolean, default=False)
    score_verification = db.Column(db.Float, default=0)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


    def has_permission(self, permission_name):
        """Méthode d'instance pour has_permission - Version corrigée"""
        if not self:
            return False

        # Admin a tous les accès
        if self.role == 'admin':
            return True

        # Superviseur a accès à tous les dashboards employés
        elif self.role == 'superviseur':
            return permission_name in ['caissier', 'conseiller', 'analyste_credit', 'gestionnaire_groupe', 'rapports']

        # Employé vérifie ses permissions spécifiques
        elif self.role == 'employe':
            if self.permissions:
                try:
                    import json
                    permissions_list = json.loads(self.permissions)
                    return permission_name in permissions_list
                except:
                    # Fallback: vérification par fonction
                    return getattr(self, 'fonction', None) == permission_name
            # Fallback: vérification par fonction
            return getattr(self, 'fonction', None) == permission_name

        # Client n'a pas de permissions spéciales
        return False


    @property
    def est_client(self):
        return self.role == 'client'

    @property
    def est_employe(self):
        return self.role == 'employe'

    @property
    def est_admin(self):
        return self.role == 'admin'

    @property
    def nom_complet(self):
        return f"{self.prenom} {self.nom}".strip()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def compare_faces(id_image_path, selfie_image_path):
    """Compare les visages entre photo ID et selfie"""
    try:
        # Charger les images
        id_image = face_recognition.load_image_file(id_image_path)
        selfie_image = face_recognition.load_image_file(selfie_image_path)

        # Détecter les visages
        id_face_encodings = face_recognition.face_encodings(id_image)
        selfie_face_encodings = face_recognition.face_encodings(selfie_image)

        if not id_face_encodings or not selfie_face_encodings:
            return False, "Aucun visage détecté dans une des images"

        # Comparer les visages
        match = face_recognition.compare_faces([id_face_encodings[0]], selfie_face_encodings[0])
        distance = face_recognition.face_distance([id_face_encodings[0]], selfie_face_encodings[0])

        return match[0], f"Similarité: {round((1 - distance[0]) * 100, 2)}%"

    except Exception as e:
        return False, f"Erreur de comparaison: {str(e)}"


templates = Jinja2Templates(directory="templates")

@app.get("/face-check")
def face_check(request: Request):
    return templates.TemplateResponse("face_check.html", {"request": request})

@app.post("/verify_face")
async def verify_face(id_image: UploadFile = File(...), selfie: UploadFile = File(...)):

    id_bytes = await id_image.read()
    selfie_bytes = await selfie.read()

    id_img = cv2.imdecode(np.frombuffer(id_bytes, np.uint8), cv2.IMREAD_COLOR)
    selfie_img = cv2.imdecode(np.frombuffer(selfie_bytes, np.uint8), cv2.IMREAD_COLOR)

    # 1. Visage détecté ?
    if not detect_face(id_img):
        return {"success": False, "error": "Aucun visage détecté sur la pièce d'identité."}

    if not detect_face(selfie_img):
        return {"success": False, "error": "Aucun visage détecté sur le selfie."}

    # 2. Anti-fake
    if is_fake_image(selfie_img):
        return {"success": False, "error": "Selfie suspect — possible photo d'écran ou impression."}

    # 3. Comparaison IA (Facenet512)
    result = DeepFace.verify(
        img1_path=id_bytes,
        img2_path=selfie_bytes,
        model_name="Facenet512",
        distance_metric="cosine",
        enforce_detection=False
    )

    score = round((1 - result["distance"]) * 100, 2)

    return {"success": result["verified"], "score": score}


@app.route('/conseiller/creer-dossier', methods=['GET', 'POST'])
@login_required
def creer_dossier():
    """Créer un nouveau dossier client avec reconnaissance faciale"""
    if current_user.role != 'employe' or not current_user.has_permission('conseiller'):
        return redirect(url_for('employe_dashboard'))

    if request.method == 'POST':
        try:
            # Données de base
            nom = request.form.get('nom')
            prenom = request.form.get('prenom')
            email = request.form.get('email')
            telephone = request.form.get('telephone')
            cin = request.form.get('cin')
            adresse = request.form.get('adresse')
            profession = request.form.get('profession')
            revenu_mensuel = float(request.form.get('revenu_mensuel', 0))
            depenses_mensuelles = float(request.form.get('depenses_mensuelles', 0))

            # Gérer les photos (upload OU caméra)
            photo_id_file = request.files.get('photo_id')
            photo_selfie_file = request.files.get('photo_selfie')
            photo_id_data = request.form.get('photo_id_data')  # Base64 de la caméra
            photo_selfie_data = request.form.get('photo_selfie_data')

            # Photo ID
            if photo_id_data:  # Priorité à la caméra
                id_filename = f"id_{cin}_camera.jpg"
                id_path = os.path.join(app.config['UPLOAD_FOLDER'], id_filename)
                save_base64_image(photo_id_data, id_path)
            elif photo_id_file and allowed_file(photo_id_file.filename):
                id_filename = f"id_{cin}_{secure_filename(photo_id_file.filename)}"
                id_path = os.path.join(app.config['UPLOAD_FOLDER'], id_filename)
                photo_id_file.save(id_path)
            else:
                return render_template('creer_dossier.html', error="Photo ID requise")

            # Selfie
            if photo_selfie_data:  # Priorité à la caméra
                selfie_filename = f"selfie_{cin}_camera.jpg"
                selfie_path = os.path.join(app.config['UPLOAD_FOLDER'], selfie_filename)
                save_base64_image(photo_selfie_data, selfie_path)
            elif photo_selfie_file and allowed_file(photo_selfie_file.filename):
                selfie_filename = f"selfie_{cin}_{secure_filename(photo_selfie_file.filename)}"
                selfie_path = os.path.join(app.config['UPLOAD_FOLDER'], selfie_filename)
                photo_selfie_file.save(selfie_path)
            else:
                return render_template('creer_dossier.html', error="Selfie requis")

            # Vérification faciale
            match, message = compare_faces(id_path, selfie_path)

            if not match:
                # Supprimer les images si échec
                os.remove(id_path)
                os.remove(selfie_path)
                return render_template('creer_dossier.html',
                                       error=f"Échec vérification faciale: {message}")

            # Calcul capacité d'emprunt
            capacite_remboursement = revenu_mensuel - depenses_mensuelles
            score_capacite = min(100, (capacite_remboursement / revenu_mensuel * 100)) if revenu_mensuel > 0 else 0

            # Créer le client
            nouveau_client = User(
                nom=nom,
                prenom=prenom,
                email=email,
                telephone=telephone,
                cin=cin,
                adresse=adresse,
                profession=profession,
                revenu_mensuel=revenu_mensuel,
                depenses_mensuelles=depenses_mensuelles,
                capacite_remboursement=capacite_remboursement,
                photo_id=id_filename,
                photo_selfie=selfie_filename,
                verification_faciale=True,
                score_verification=float(message.replace("Similarité:", "").replace("%", "").strip()),
                role='client',
                statut='actif'
            )

            # Mot de passe temporaire
            password_temp = "Temp123!"
            nouveau_client.set_password(password_temp)

            db.session.add(nouveau_client)
            db.session.commit()

            return render_template('creer_dossier_success.html',
                                   client=nouveau_client,
                                   message=message,
                                   password_temp=password_temp)

        except Exception as e:
            return render_template('creer_dossier.html',
                                   error=f"Erreur lors de la création: {str(e)}")

    return render_template('creer_dossier.html')


class Groupe(db.Model):
    __tablename__ = 'groupes'

    id = db.Column(db.Integer, primary_key=True)
    nom = db.Column(db.String(100))
    code_groupe = db.Column(db.String(20), unique=True)
    zone = db.Column(db.String(100))
    date_creation = db.Column(db.DateTime, default=datetime.utcnow)
    statut = db.Column(db.String(20), default='actif')


class Pret(db.Model):
    __tablename__ = 'prets'

    id = db.Column(db.Integer, primary_key=True)
    client_id = db.Column(db.Integer)
    groupe_id = db.Column(db.Integer)
    montant = db.Column(db.Float)
    taux_interet = db.Column(db.Float, default=12.0)
    duree_mois = db.Column(db.Integer)
    date_demande = db.Column(db.DateTime, default=datetime.utcnow)
    date_approbation = db.Column(db.DateTime)
    date_decaissement = db.Column(db.DateTime)
    statut = db.Column(db.String(20), default='en_attente')
    motif = db.Column(db.String(100))
    montant_interet = db.Column(db.Float)
    montant_total = db.Column(db.Float)
    mensualite = db.Column(db.Float)


class Remboursement(db.Model):
    __tablename__ = 'remboursements'

    id = db.Column(db.Integer, primary_key=True)
    pret_id = db.Column(db.Integer)
    client_id = db.Column(db.Integer)
    montant = db.Column(db.Float)
    date_remboursement = db.Column(db.DateTime, default=datetime.utcnow)
    date_echeance = db.Column(db.DateTime)
    statut = db.Column(db.String(20), default='paye')
    type_paiement = db.Column(db.String(20))
    reference = db.Column(db.String(100))


class Notification(db.Model):
    __tablename__ = 'notifications'

    id = db.Column(db.Integer, primary_key=True)
    utilisateur_id = db.Column(db.Integer)
    titre = db.Column(db.String(200))
    message = db.Column(db.Text)
    type_notification = db.Column(db.String(50))
    lue = db.Column(db.Boolean, default=False)
    date_creation = db.Column(db.DateTime, default=datetime.utcnow)
    lien = db.Column(db.String(500))


class NotificationManager:
    def __init__(self):
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', 587))
        self.smtp_username = os.getenv('SMTP_USERNAME')
        self.smtp_password = os.getenv('SMTP_PASSWORD')
        self.sms_api_key = os.getenv('SMS_API_KEY')
        self.sms_api_secret = os.getenv('SMS_API_SECRET')

    def notifier_remboursement_reussi(self, user, remboursement):
        """Notification pour remboursement réussi"""
        pret = Pret.query.get(remboursement.pret_id)

        message = f"""
        ✅ Remboursement confirmé !

        Cher(e) {user.prenom} {user.nom},

        Votre remboursement de {remboursement.montant:.2f} € a été enregistré avec succès.

        📋 Détails :
        - Prêt : {pret.motif if pret else 'N/A'}
        - Référence : {remboursement.reference}
        - Date : {remboursement.date_remboursement.strftime('%d/%m/%Y %H:%M')}
        - Méthode : {remboursement.type_paiement}

        Merci pour votre ponctualité !
        L'équipe GMES Microcrédit
        """

        print(f"📧 Notification: {user.nom_complet} a effectué un remboursement de {remboursement.montant} €")

        # Envoyer les notifications
        self._envoyer_notification_db(user, "Remboursement confirmé", message, 'success')
        self._envoyer_email(user, "✅ Remboursement confirmé - GMES", message)
        self._envoyer_sms(user, f"GMES: Remboursement de {remboursement.montant} € confirmé. Merci!")

    def notifier_approbation_pret(self, user, pret):
        """Notification pour approbation de prêt"""
        message = f"""
        🎉 Félicitations ! Votre prêt est approuvé.

        Cher(e) {user.prenom} {user.nom},

        Votre demande de prêt a été approuvée.

        📋 Détails du prêt :
        - Montant : {pret.montant:.2f} €
        - Durée : {pret.duree_mois} mois
        - Mensualité : {pret.mensualite:.2f} €
        - Motif : {pret.motif}

        Les fonds seront disponibles sous 24-48h.

        L'équipe GMES Microcrédit
        """

        self._envoyer_notification_db(user, "Prêt approuvé", message, 'success')
        self._envoyer_email(user, "🎉 Prêt approuvé - GMES", message)
        self._envoyer_sms(user, f"GMES: Prêt de {pret.montant} € approuvé!")

    def notifier_rejet_pret(self, user, pret, motif):
        """Notification pour rejet de prêt"""
        message = f"""
        ❌ Statut de votre demande de prêt

        Cher(e) {user.prenom} {user.nom},

        Votre demande de prêt a été rejetée.

        📋 Détails :
        - Motif : {motif}
        - Montant demandé : {pret.montant:.2f} €

        Nous vous encourageons à :
        • Améliorer votre score de crédit
        • Revoir votre capacité de remboursement
        • Soumettre une nouvelle demande ultérieurement

        L'équipe GMES Microcrédit
        """

        self._envoyer_notification_db(user, "Prêt rejeté", message, 'warning')
        self._envoyer_email(user, "❌ Statut de votre prêt - GMES", message)
        self._envoyer_sms(user, f"GMES: Prêt rejeté. Consultez votre email pour plus de détails.")

    def notifier_rappel_remboursement(self, user, pret, jours_restants):
        """Rappel de remboursement"""
        message = f"""
        ⏰ Rappel de remboursement

        Cher(e) {user.prenom} {user.nom},

        Votre prochaine échéance de remboursement approche !

        📋 Détails :
        - Prêt : {pret.motif}
        - Mensualité : {pret.mensualite:.2f} €
        - Jours restants : {jours_restants}

        Pensez à effectuer votre paiement à temps pour éviter les pénalités.

        L'équipe GMES Microcrédit
        """

        self._envoyer_notification_db(user, f"Rappel: {jours_restants} jours", message, 'info')
        self._envoyer_email(user, f"⏰ Rappel de remboursement - {jours_restants} jours", message)
        if jours_restants <= 2:  # SMS seulement si très proche
            self._envoyer_sms(user, f"RAPPEL GMES: {pret.mensualite:.2f} € dans {jours_restants} jour(s)")

    def notifier_nouveau_groupe(self, user, groupe):
        """Notification pour nouveau groupe"""
        message = f"""
        👥 Bienvenue dans votre nouveau groupe !

        Cher(e) {user.prenom} {user.nom},

        Vous avez rejoint le groupe : {groupe.nom}

        📋 Informations du groupe :
        - Code : {groupe.code_groupe}
        - Zone : {groupe.zone}
        - Coordinateur : {groupe.coordinateur.nom_complet if groupe.coordinateur else 'À désigner'}

        Participez activement aux réunions et bénéficiez de la solidarité du groupe !

        L'équipe GMES Microcrédit
        """

        self._envoyer_notification_db(user, "Nouveau groupe", message, 'info')
        self._envoyer_email(user, "👥 Bienvenue dans votre groupe - GMES", message)

    def _envoyer_notification_db(self, user, titre, message, type_notif):
        """Enregistre la notification en base de données"""
        try:
            notification = Notification(
                utilisateur_id=user.id,
                titre=titre,
                message=message,
                type_notification=type_notif,
                lien='/notifications'
            )
            db.session.add(notification)
            db.session.commit()
        except Exception as e:
            print(f"Erreur notification DB: {e}")

    def _envoyer_email(self, user, sujet, message):
        """Envoie un email (simulé pour l'instant)"""
        try:
            if self.smtp_username and self.smtp_password:
                # Ici vous intégreriez votre service d'email
                print(f"📧 Email envoyé à {user.email}: {sujet}")
            else:
                print(f"📧 [SIMULATION] Email à {user.email}: {sujet}")
        except Exception as e:
            print(f"Erreur email: {e}")

    def _envoyer_sms(self, user, message):
        """Envoie un SMS (simulé pour l'instant)"""
        try:
            if self.sms_api_key and user.telephone:
                # Ici vous intégreriez votre service SMS
                print(f"📱 SMS envoyé à {user.telephone}: {message}")
            else:
                print(f"📱 [SIMULATION] SMS à {user.telephone}: {message}")
        except Exception as e:
            print(f"Erreur SMS: {e}")

    def notifier_retard_remboursement(self, user, pret, jours_retard):
        """Notification pour retard de remboursement"""
        message = f"""
        ⚠️ Retard de remboursement

        Cher(e) {user.prenom} {user.nom},

        Votre remboursement est en retard de {jours_retard} jour(s).

        📋 Détails :
        - Prêt : {pret.motif}
        - Mensualité : {pret.mensualite:.2f} €
        - Jours de retard : {jours_retard}

        Veuillez régulariser votre situation au plus vite pour éviter :
        • Des pénalités de retard
        • Une affectation de votre score de crédit
        • Des restrictions futures

        L'équipe GMES Microcrédit
        """

        self._envoyer_notification_db(user, f"Retard: {jours_retard} jour(s)", message, 'warning')
        self._envoyer_email(user, f"⚠️ Retard de remboursement - {jours_retard} jour(s)", message)
        self._envoyer_sms(user, f"RETARD GMES: {jours_retard} jour(s). {pret.mensualite:.2f} €")


notification_manager = NotificationManager()




def obtenir_actions_utilisateur(user_id):
    """Récupère les actions d'un utilisateur pour la gamification"""
    # Simulation - à remplacer par votre logique réelle
    actions = [
        {'type': 'remboursement_ponctuel', 'description': 'Remboursement à temps'},
        {'type': 'pret_rembourse', 'description': 'Prêt complètement remboursé'},
        {'type': 'participation_groupe', 'description': 'Participation active au groupe'}
    ]
    return actions

def calculer_historique_client(client_id):
    """Calcule l'historique d'un client pour le scoring"""
    # Simulation - à remplacer par votre logique réelle
    return {
        'nombre_prets': Pret.query.filter_by(client_id=client_id).count(),
        'prets_rembourses': Pret.query.filter_by(client_id=client_id, statut='termine').count(),
        'taux_remboursement': 85,  # À calculer dynamiquement
        'jours_retard_moyen': 2,
        'incidents_paiement': 0
    }


def calculer_statistiques_utilisateur(user):
    """Calcule les stats pour le tableau de bord"""
    from utils.ai_scoring import ai_scorer
    from utils.gamification import gamification

    stats = {}

    # Vérifier si c'est un Client (avec groupe_id) ou User (admin/employé)
    if hasattr(user, 'groupe_id'):  # C'est un Client
        # Score de crédit IA seulement pour les clients
        client_data = {
            'revenu_mensuel': getattr(user, 'revenu_mensuel', 0),
            'anciennete_client': (datetime.utcnow() - user.date_inscription).days // 30,
            'profession': getattr(user, 'profession', 'Non spécifié')
        }
        historique = calculer_historique_client(user.id)
        score = ai_scorer.calculate_credit_score(client_data, {}, historique)

        stats['score_credit'] = score
        stats['score_categorie'] = 'excellent' if score >= 750 else 'good' if score >= 650 else 'fair'
        stats['score_label'] = ai_scorer.explain_score(client_data, {}, historique)

        # Gamification seulement pour les clients
        user_actions = obtenir_actions_utilisateur(user.id)
        points = gamification.calculate_points(user_actions)
        niveau_info = gamification.get_level_progress(points)

        stats.update({
            'niveau': niveau_info['current_level'],
            'points': points,
            'progression': niveau_info['progress'],
            'badge': niveau_info['current_badge']
        })

        # Groupe seulement pour les clients
        if user.groupe_id:
            groupe = Groupe.query.get(user.groupe_id)
            stats.update({
                'groupe_nom': groupe.nom if groupe else None,
                'groupe_membres': User.query.filter_by(groupe_id=user.groupe_id).count() if user.groupe_id else 0,
            })

    # Statistiques communes à tous les utilisateurs - CORRECTIONS ICI
    stats.update({
        'prets_actifs': Pret.query.filter(
            Pret.client_id == user.id,
            Pret.statut == 'approuve'
        ).count() if hasattr(user, 'groupe_id') else 0,

        'montant_actifs': db.session.query(db.func.sum(Pret.montant)).filter(
            Pret.client_id == user.id,
            Pret.statut == 'approuve'
        ).scalar() or 0 if hasattr(user, 'groupe_id') else 0,

        'notifications_non_lues': Notification.query.filter_by(utilisateur_id=user.id, lue=False).count() if hasattr(
            Notification, 'query') else 0
    })

    return stats


# Dans app.py ou routes.py
@app.route('/')
def index():
    # Récupérer les vraies valeurs depuis votre base de données
    from models import db, Client, Pret, Remboursement
    from models import get_statistiques

    # Récupérer les statistiques réelles
    stats = get_statistiques()

    # Calculer les statistiques réelles
    total_clients = db.session.query(Client).count()

    total_prets = db.session.query(db.func.sum(Pret.montant)).scalar() or 0

    # Calcul du taux de remboursement
    total_due = db.session.query(db.func.sum(Pret.montant)).scalar() or 1
    total_paid = db.session.query(db.func.sum(Remboursement.montant)).scalar() or 0
    taux_remboursement = int((total_paid / total_due) * 100) if total_due > 0 else 0

    # Passer les variables au template
    return render_template('index.html',
                           clients_actifs=total_clients,
                           prets_accordes=total_prets,
                           taux_remboursement=taux_remboursement,
                           communautes_desservies=5)  # À adapter



# === AJOUTEZ CE FILTRE JINJA2 ===

@app.template_filter('format_htg')
def format_htg(value):
    """Formate les montants en HTG"""
    value = float(value)
    if value >= 1000000:
        return f"{(value/1000000):.1f}M"
    elif value >= 1000:
        return f"{(value/1000):.0f}K"
    return f"{value:,.0f}"


# Ajoutez cette fonction utilitaire
def save_base64_image(base64_string, output_path):
    """Convertit et sauvegarde une image base64"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]

        image_data = base64.b64decode(base64_string)
        with open(output_path, 'wb') as f:
            f.write(image_data)
        return True
    except Exception as e:
        print(f"Erreur sauvegarde image base64: {e}")
        return False

# ==================== CONFIGURATION USER LOADER ====================

# @login_manager.user_loader
# def load_user(user_id):
#     return User.query.get(int(user_id))  # ✅ Plus simple !
#

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))  # ← Version moderne au lieu de User.query.get()

# @app.route("/")
# def index():
#     return {"message": "✅ API GMES en ligne et fonctionnelle"}

@app.route("/")
def index():
    return render_template('accueil.html')

# ==================== INITIALISATION ====================

def initialiser_donnees():
    """Initialise la base de données avec des données de test"""
    try:
        print("🗃️ Création des tables...")
        db.drop_all()  # ⚠️ SUPPRIME les tables existantes
        db.create_all()

        print("👤 Création des comptes...")

        # Créer l'admin
        if not User.query.filter_by(username='admin').first():
            admin = User(
                username='admin',
                email='admin@gmes.com',
                role='admin',  # ← Type défini ici
                nom='Admin',
                prenom='System',
                telephone='+50900000000'
            )
            admin.set_password('admin123')
            db.session.add(admin)
            print("✅ Admin créé: admin / admin123")

        # Créer un employé
        if not User.query.filter_by(username='employe').first():
            employe = User(
                username='employe',
                email='employe@gmes.com',
                role='employe',  # ← Type défini ici
                nom='Pierre',
                prenom='Jean',
                telephone='+50912345678'
            )
            employe.set_password('employe123')
            db.session.add(employe)
            print("✅ Employé créé: employe / employe123")

        # Créer un groupe
        if not Groupe.query.first():
            groupe = Groupe(
                nom="Femmes Entrepreneures",
                code_groupe="GRP001",
                zone="Port-au-Prince"
            )
            db.session.add(groupe)
            print("✅ Groupe créé")

        # Créer un client
        if not User.query.filter_by(email='client@example.com').first():
            client = User(
                code_client="CLT001",
                nom="Dupont",
                prenom="Marie",
                telephone="+50912345670",
                email="client@example.com",
                adresse="Port-au-Prince",
                cin="1234567890",
                date_naissance=datetime(1985, 5, 15),
                profession="Commerçante",
                revenu_mensuel=15000,
                groupe_id=1
            )
            client.set_password("client123")
            db.session.add(client)
            print("✅ Client créé: client@example.com / client123")

        db.session.commit()
        print("🎉 Données initialisées avec succès!")
        return True

    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation: {e}")
        import traceback
        traceback.print_exc()
        return False
# Initialiser les données au démarrage
with app.app_context():
    initialiser_donnees()


# ==================== ROUTES ====================

@app.route('/')
def accueil():
    return render_template('accueil.html')


@app.route('/connexion', methods=['GET', 'POST'])
def connexion():
    if request.method == 'POST':
        identifiant = request.form.get('identifiant')
        mot_de_passe = request.form.get('password')

        user = User.query.filter(
            (User.username == identifiant) | (User.email == identifiant)
        ).first()

        if user and user.check_password(mot_de_passe):
            # ✅ VÉRIFICATION POUR LES EMPLOYÉS
            if user.role in ['employe', 'superviseur'] and user.statut != 'actif':
                return render_template('connexion.html',
                                       erreur="Votre compte employé est en attente d'approbation administrative.")

            # ✅ LES CLIENTS PEUVENT TOUJOURS SE CONNECTER
            if user.role == 'client' and user.statut != 'actif':
                return render_template('connexion.html',
                                       erreur="Votre compte client est en attente d'activation.")

            login_user(user)
            print(f"✅ Connexion réussie: {user.role} - Statut: {user.statut}")
            return redirect(url_for('dashboard_redirect'))

        return render_template('connexion.html', erreur="Identifiant ou mot de passe incorrect")

    return render_template('connexion.html')



@app.route('/deconnexion')
def deconnexion():
    logout_user()
    return redirect(url_for('accueil'))



@app.route('/test')
def test():
    return jsonify({
        'status': 'GMES Microcrédit - Système Opérationnel',
        'message': 'Tout fonctionne correctement!'
    })


# ==================== SYSTÈME DE REMBOURSEMENTS ====================

@app.route('/remboursement/nouveau', methods=['GET', 'POST'])
@login_required
def nouveau_remboursement():
    if request.method == 'POST':
        # Récupérer les données du formulaire
        pret_id = request.form.get('pret_id')
        montant = float(request.form.get('montant'))
        type_paiement = request.form.get('type_paiement')
        reference = request.form.get('reference')

        # Vérifier que le prêt appartient à l'utilisateur
        pret = Pret.query.filter_by(id=pret_id, client_id=current_user.id).first()
        if not pret:
            return render_template('nouveau_remboursement.html',
                                   error="Prêt non trouvé ou non autorisé",
                                   prets=Pret.query.filter_by(client_id=current_user.id, statut='approuve').all())

        # Vérifier que le montant est valide
        if montant <= 0:
            return render_template('nouveau_remboursement.html',
                                   error="Montant invalide",
                                   prets=Pret.query.filter_by(client_id=current_user.id, statut='approuve').all())

        # Calculer la date d'échéance (prochaine échéance)
        date_echeance = datetime.utcnow().replace(day=1)
        if date_echeance.month == 12:
            date_echeance = date_echeance.replace(year=date_echeance.year + 1, month=1)
        else:
            date_echeance = date_echeance.replace(month=date_echeance.month + 1)

        # Créer le remboursement
        remboursement = Remboursement(
            pret_id=pret_id,
            client_id=current_user.id,
            montant=montant,
            date_echeance=date_echeance,
            type_paiement=type_paiement,
            reference=reference,
            statut='paye'
        )

        db.session.add(remboursement)
        db.session.commit()

        # 🔔 NOTIFICATION de remboursement réussi
        try:
            notification_manager.notifier_remboursement_reussi(current_user, remboursement)
        except Exception as e:
            print(f"Erreur notification: {e}")

        return redirect(url_for('mes_remboursements'))

    # GET - Afficher le formulaire
    prets = Pret.query.filter_by(client_id=current_user.id, statut='approuve').all()

    # Préparer les données pour le template
    prets_avec_solde = []
    for pret in prets:
        # Calculer le solde restant
        total_rembourse = db.session.query(db.func.sum(Remboursement.montant)).filter_by(
            pret_id=pret.id, statut='paye'
        ).scalar() or 0
        solde_restant = pret.montant_total - total_rembourse

        prets_avec_solde.append({
            'pret': pret,
            'solde_restant': solde_restant,
            'prochaine_echeance': datetime.utcnow().replace(day=1)  # Simplifié
        })

    return render_template('nouveau_remboursement.html', prets=prets_avec_solde)



@app.route('/mes-remboursements')
@login_required
def mes_remboursements():
    remboursements = Remboursement.query.filter_by(client_id=current_user.id).all()

    # Associer les remboursements avec les prêts
    remboursements_avec_prets = []
    for remb in remboursements:
        pret = Pret.query.get(remb.pret_id)
        remboursements_avec_prets.append({
            'remboursement': remb,
            'pret': pret
        })

    return render_template('mes_remboursements.html', remboursements_avec_prets=remboursements_avec_prets)


@app.route('/admin/notifications')
@login_required
def admin_notifications():
    if getattr(current_user, 'role', None) != 'admin':
        return redirect(url_for('tableau_de_bord'))

    config = {
        'SMTP_SERVER': os.getenv('SMTP_SERVER'),
        'SMTP_USERNAME': os.getenv('SMTP_USERNAME'),
        'SMS_API_KEY': os.getenv('SMS_API_KEY')
    }

    return render_template('admin_notifications.html', config=config)

@app.route('/admin/remboursements')
@login_required
def admin_remboursements():
    if getattr(current_user, 'role', None) != 'admin':
        return redirect(url_for('tableau_de_bord'))

    remboursements = Remboursement.query.all()

    # Associer avec clients et prêts
    remboursements_complets = []
    for remb in remboursements:
        pret = Pret.query.get(remb.pret_id)
        client = User.query.get(remb.client_id)  # ✅ CORRECTION: Utiliser User
        remboursements_complets.append({
            'remboursement': remb,
            'pret': pret,
            'client': client
        })

    return render_template('admin_remboursements.html', remboursements_complets=remboursements_complets)

@app.route('/api/calculer-echeancier/<int:pret_id>')
@login_required
def calculer_echeancier(pret_id):
    """Calcule l'échéancier d'un prêt"""
    pret = Pret.query.get_or_404(pret_id)

    # Vérifier que le prêt appartient au client
    if pret.client_id != current_user.id:
        return jsonify({'error': 'Accès non autorisé'})

    echeances = []
    montant_restant = pret.montant_total
    date_courante = datetime.utcnow()

    for i in range(pret.duree_mois):
        date_echeance = date_courante.replace(month=date_courante.month + i)
        echeances.append({
            'numero': i + 1,
            'date': date_echeance.strftime('%d/%m/%Y'),
            'montant': pret.mensualite,
            'capital': pret.mensualite * 0.8,  # Estimation
            'interet': pret.mensualite * 0.2  # Estimation
        })

    return jsonify({
        'pret': {
            'montant': pret.montant,
            'duree': pret.duree_mois,
            'mensualite': pret.mensualite,
            'total_a_rembourser': pret.montant_total
        },
        'echeances': echeances
    })


# ==================== GESTION DES GROUPES DE SOLIDARITÉ ====================



@app.route('/groupes')
@login_required
def liste_groupes():
    """Liste tous les groupes - accessible aux conseillers"""
    if current_user.role != 'employe' or not current_user.has_permission('conseiller'):
        return redirect(url_for('employe_dashboard'))

    groupes = Groupe.query.all()
    return render_template('liste_groupes.html', groupes=groupes)


@app.route('/admin/assigner-groupe/<int:employe_id>', methods=['GET', 'POST'])
@login_required
def assigner_groupe(employe_id):
    """Assigner un groupe à un employé - Admin seulement"""
    if current_user.role != 'admin':
        return redirect(url_for('tableu_de_bord'))

    employe = User.query.get_or_404(employe_id)
    groupes = Groupe.query.all()

    if request.method == 'POST':
        groupe_id = request.form.get('groupe_id')
        employe.groupe_id = groupe_id if groupe_id else None
        db.session.commit()
        return redirect(url_for('gerer_employes'))

    return render_template('assigner_groupe.html', employe=employe, groupes=groupes)


@app.route('/groupe/<int:groupe_id>')
@login_required
def detail_groupe(groupe_id):
    groupe = Groupe.query.get_or_404(groupe_id)
    clients = User.query.filter_by(groupe_id=groupe_id).all()
    prets_du_groupe = Pret.query.filter_by(groupe_id=groupe_id).all()

    # Calculer les statistiques du groupe
    stats = {
        'nombre_membres': len(clients),
        'prets_actifs': len([p for p in prets_du_groupe if p.statut == 'approuve']),
        'montant_total_prets': sum(p.montant for p in prets_du_groupe),
        'taux_remboursement': 95  # À calculer dynamiquement
    }

    return render_template('detail_groupe.html',
                           groupe=groupe,
                           clients=clients,
                           prets=prets_du_groupe,
                           stats=stats)


@app.route('/groupe/creer', methods=['GET', 'POST'])
@login_required
def creer_groupe():
    if getattr(current_user, 'role', None) not in ['admin', 'employe']:
        return redirect(url_for('tableau_de_bord'))

    if request.method == 'POST':
        nom = request.form.get('nom')
        zone = request.form.get('zone')

        # Générer un code de groupe unique
        code_groupe = f"GRP{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        groupe = Groupe(
            nom=nom,
            code_groupe=code_groupe,
            zone=zone
        )

        db.session.add(groupe)
        db.session.commit()

        return redirect(url_for('liste_groupes'))

    return render_template('creer_groupe.html')


@app.route('/groupe/<int:groupe_id>/rejoindre')
@login_required
def rejoindre_groupe(groupe_id):
    # ... code existant ...

    current_user.groupe_id = groupe_id
    db.session.commit()

    # 🔔 NOTIFICATION de nouveau groupe
    notification_manager.notifier_nouveau_groupe(current_user, groupe)

    return redirect(url_for('detail_groupe', groupe_id=groupe_id))


@app.route('/groupe/<int:groupe_id>/quitter')
@login_required
def quitter_groupe(groupe_id):
    # Seuls les clients peuvent quitter des groupes
    if hasattr(current_user, 'role'):
        return redirect(url_for('tableau_de_bord'))

    # Vérifier que le client est bien dans ce groupe
    if current_user.groupe_id != groupe_id:
        return redirect(url_for('tableau_de_bord'))

    current_user.groupe_id = None
    db.session.commit()

    return redirect(url_for('liste_groupes'))


@app.route('/groupe/<int:groupe_id>/demande-pret-solidaire', methods=['GET', 'POST'])
@login_required
def demande_pret_solidaire(groupe_id):
    # Vérifier que l'utilisateur est un client membre du groupe
    if hasattr(current_user, 'role') or current_user.groupe_id != groupe_id:
        return redirect(url_for('tableau_de_bord'))

    groupe = Groupe.query.get_or_404(groupe_id)

    if request.method == 'POST':
        montant = float(request.form.get('montant'))
        duree = int(request.form.get('duree'))
        motif = request.form.get('motif')

        # Calculs automatiques
        taux_mensuel = 12.0 / 100 / 12
        mensualite = montant * taux_mensuel * (1 + taux_mensuel) ** duree / ((1 + taux_mensuel) ** duree - 1)
        montant_interet = (mensualite * duree) - montant
        montant_total = mensualite * duree

        nouveau_pret = Pret(
            client_id=current_user.id,
            groupe_id=groupe_id,
            montant=montant,
            duree_mois=duree,
            motif=motif,
            mensualite=round(mensualite, 2),
            montant_interet=round(montant_interet, 2),
            montant_total=round(montant_total, 2),
            statut='en_attente_solidaire'  # Statut spécial pour prêts solidaires
        )

        db.session.add(nouveau_pret)
        db.session.commit()

        return redirect(url_for('detail_groupe', groupe_id=groupe_id))

    return render_template('demande_pret_solidaire.html', groupe=groupe)


@app.route('/api/statistiques-groupes')
@login_required
def statistiques_groupes():
    if getattr(current_user, 'role', None) != 'admin':
        return jsonify({'error': 'Accès non autorisé'})

    groupes = Groupe.query.all()
    statistiques = []

    for groupe in groupes:
        clients = User.query.filter_by(groupe_id=groupe.id).all()
        prets = Pret.query.filter_by(groupe_id=groupe.id).all()

        stats = {
            'groupe_id': groupe.id,
            'nom_groupe': groupe.nom,
            'nombre_membres': len(clients),
            'prets_actifs': len([p for p in prets if p.statut == 'approuve']),
            'montant_total_prets': sum(p.montant for p in prets),
            'zone': groupe.zone
        }
        statistiques.append(stats)

    return jsonify(statistiques)


# ==================== RAPPORTS ET STATISTIQUES ====================

@app.route('/admin/rapports')
@login_required
def admin_rapports():
    if getattr(current_user, 'role', None) != 'admin':
        return redirect(url_for('tableau_de_bord'))

    # Calculer les statistiques globales
    stats = calculer_statistiques_globales()

    return render_template('admin_rapports.html', stats=stats)


@app.route('/admin/rapport-prets')
@login_required
def rapport_prets():
    if getattr(current_user, 'role', None) != 'admin':
        return redirect(url_for('tableau_de_bord'))

    # Filtres
    date_debut = request.args.get('date_debut')
    date_fin = request.args.get('date_fin')
    statut = request.args.get('statut')

    # Requête de base
    query = Pret.query

    # Appliquer les filtres
    if date_debut:
        query = query.filter(Pret.date_demande >= datetime.strptime(date_debut, '%Y-%m-%d'))
    if date_fin:
        query = query.filter(Pret.date_demande <= datetime.strptime(date_fin, '%Y-%m-%d'))
    if statut:
        query = query.filter(Pret.statut == statut)

    prets = query.all()

    # Préparer les données pour le rapport
    prets_rapport = []
    for pret in prets:
        client = User.query.get(pret.client_id)
        prets_rapport.append({
            'pret': pret,
            'client': client
        })

    return render_template('rapport_prets.html',
                           prets_rapport=prets_rapport,
                           filters={'date_debut': date_debut, 'date_fin': date_fin, 'statut': statut})


@app.route('/admin/rapport-remboursements')
@login_required
def rapport_remboursements():
    if getattr(current_user, 'role', None) != 'admin':
        return redirect(url_for('tableau_de_bord'))

    # Filtres
    date_debut = request.args.get('date_debut')
    date_fin = request.args.get('date_fin')

    query = Remboursement.query

    if date_debut:
        query = query.filter(Remboursement.date_remboursement >= datetime.strptime(date_debut, '%Y-%m-%d'))
    if date_fin:
        query = query.filter(Remboursement.date_remboursement <= datetime.strptime(date_fin, '%Y-%m-%d'))

    remboursements = query.all()

    # Préparer les données
    remboursements_rapport = []
    for remb in remboursements:
        pret = Pret.query.get(remb.pret_id)
        client = User.query.get(remb.client_id)
        remboursements_rapport.append({
            'remboursement': remb,
            'pret': pret,
            'client': client
        })

    return render_template('rapport_remboursements.html',
                           remboursements_rapport=remboursements_rapport,
                           filters={'date_debut': date_debut, 'date_fin': date_fin})


@app.route('/api/statistiques-temps-reel')
@login_required
def statistiques_temps_reel():
    if getattr(current_user, 'role', None) != 'admin':
        return jsonify({'error': 'Accès non autorisé'})

    stats = calculer_statistiques_globales()
    return jsonify(stats)




@app.before_request
def list_routes():
    if not hasattr(app, 'routes_listed'):
        print("=== ROUTES DISPONIBLES ===")
        for rule in app.url_map.iter_rules():
            print(f"{rule.rule} -> {rule.endpoint}")
        print("==========================")
        app.routes_listed = True

@app.route('/test-mobile-routes')
def test_mobile_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        if 'mobile' in str(rule.rule):
            routes.append(f"{rule.rule} -> {rule.endpoint}")
    return "<br>".join(routes) if routes else "Aucune route mobile trouvée"




@app.route('/admin/export-prets-excel')
@login_required
def export_prets_excel():
    if getattr(current_user, 'role', None) != 'admin':
        return redirect(url_for('tableau_de_bord'))

    prets = Pret.query.all()

    # Créer un DataFrame (simulé)
    data = []
    for pret in prets:
        client = User.query.get(pret.client_id)
        data.append({
            'ID Prêt': pret.id,
            'Client': f"{client.prenom} {client.nom}" if client else "N/A",
            'Montant': pret.montant,
            'Durée (mois)': pret.duree_mois,
            'Mensualité': pret.mensualite,
            'Statut': pret.statut,
            'Date Demande': pret.date_demande.strftime('%d/%m/%Y'),
            'Motif': pret.motif
        })

    # Pour l'instant, retourner un JSON (implémentez l'export Excel plus tard)
    return jsonify({
        'message': 'Export Excel des prêts',
        'nombre_prets': len(data),
        'data': data
    })


@app.route('/cron/rappels-quotidiens')
def rappels_quotidiens():
    """
    Route pour les rappels automatiques (à appeler via cron job)
    """
    try:
        # Prêts avec remboursements en retard
        prets_actifs = Pret.query.filter_by(statut='approuve').all()

        for pret in prets_actifs:
            client = User.query.get(pret.client_id)

            # Calculer les jours jusqu'à la prochaine échéance
            # (simplifié pour l'exemple)
            jours_restants = 5  # À calculer dynamiquement

            if jours_restants <= 3:  # Rappel 3 jours avant
                notification_manager.notifier_rappel_remboursement(
                    client, pret, jours_restants
                )

        return jsonify({
            'status': 'success',
            'message': f'Rappels envoyés pour {len(prets_actifs)} prêts'
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/admin/test-notification')
@login_required
def test_notification():
    """Route de test pour les notifications"""
    if getattr(current_user, 'role', None) != 'admin':
        return redirect(url_for('tableau_de_bord'))

    # Test avec le premier client
    client = User.query.first()
    pret = Pret.query.filter_by(client_id=client.id).first()

    if pret and client:
        notification_manager.notifier_approbation_pret(client, pret)
        return jsonify({'status': 'Notification de test envoyée'})

    return jsonify({'error': 'Aucun client/pret trouvé pour le test'})

# Fonctions utilitaires pour les statistiques
def calculer_statistiques_globales():
    """Calcule les statistiques globales du système"""
    total_clients = User.query.filter_by(role='client').count()  # ✅ Seulement les clients
    total_prets = Pret.query.count()
    prets_actifs = Pret.query.filter_by(statut='approuve').count()
    prets_en_attente = Pret.query.filter_by(statut='en_attente').count()

    # Calcul des montants
    montant_total_prets = db.session.query(db.func.sum(Pret.montant)).scalar() or 0
    montant_prets_actifs = db.session.query(db.func.sum(Pret.montant)).filter(
        Pret.statut == 'approuve'
    ).scalar() or 0

    # Remboursements
    total_remboursements = Remboursement.query.count()
    montant_total_rembourse = db.session.query(db.func.sum(Remboursement.montant)).scalar() or 0

    # Groupes
    total_groupes = Groupe.query.count()

    # Calcul du taux de remboursement (simplifié)
    taux_remboursement = 0
    if montant_prets_actifs > 0:
        taux_remboursement = (montant_total_rembourse / montant_prets_actifs) * 100

    # ✅ CORRECTION de la jointure problématique
    clients_avec_prets_count = db.session.query(
        db.func.count(db.func.distinct(User.id))
    ).join(Pret, User.id == Pret.client_id).filter(
        User.role == 'client'
    ).scalar() or 0

    return {
        'clients': {
            'total': total_clients,
            'avec_prets': clients_avec_prets_count,  # ✅ Utiliser la version corrigée
            'nouveaux_ce_mois': User.query.filter(
                User.date_inscription >= datetime.utcnow().replace(day=1),
                User.role == 'client'  # ✅ Seulement les clients
            ).count()
        },
        'prets': {
            'total': total_prets,
            'actifs': prets_actifs,
            'en_attente': prets_en_attente,
            'montant_total': round(montant_total_prets, 2),
            'montant_actifs': round(montant_prets_actifs, 2)
        },
        'remboursements': {
            'total': total_remboursements,
            'montant_total': round(montant_total_rembourse, 2),
            'taux_remboursement': round(taux_remboursement, 1)
        },
        'groupes': {
            'total': total_groupes,
            'membres_moyen': total_clients / total_groupes if total_groupes > 0 else 0
        },
        'performance': {
            'taux_approbation': (prets_actifs / total_prets * 100) if total_prets > 0 else 0,
            'rotation_fonds': calculer_rotation_fonds()
        }
    }
def calculer_rotation_fonds():
    """Calcule la rotation des fonds (simplifié)"""
    # Implémentation simplifiée
    return 2.5  # Exemple fixe


@app.route('/tableau-bord-personnalise')
@login_required
def tableau_bord_personnalise():
    """Tableau de bord avec widgets personnalisables"""
    if getattr(current_user, 'role', None) != 'admin':
        return redirect(url_for('tableau_de_bord'))

    stats = calculer_statistiques_globales()

    # Données pour les graphiques
    prets_par_statut = db.session.query(
        Pret.statut,
        db.func.count(Pret.id)
    ).group_by(Pret.statut).all()

    prets_par_mois = db.session.query(
        db.func.strftime('%Y-%m', Pret.date_demande),
        db.func.count(Pret.id)
    ).group_by(db.func.strftime('%Y-%m', Pret.date_demande)).all()

    return render_template('tableau_bord_personnalise.html',
                           stats=stats,
                           prets_par_statut=prets_par_statut,
                           prets_par_mois=prets_par_mois)

@app.route('/demande-pret', methods=['GET', 'POST'])
@login_required
def demande_pret():
    if request.method == 'POST':
        montant = float(request.form.get('montant'))
        duree = int(request.form.get('duree'))
        motif = request.form.get('motif')

        # Calculs automatiques
        taux_mensuel = 12.0 / 100 / 12
        mensualite = montant * taux_mensuel * (1 + taux_mensuel) ** duree / ((1 + taux_mensuel) ** duree - 1)
        montant_interet = (mensualite * duree) - montant
        montant_total = mensualite * duree

        nouveau_pret = Pret(
            client_id=current_user.id,
            montant=montant,
            duree_mois=duree,
            motif=motif,
            mensualite=round(mensualite, 2),
            montant_interet=round(montant_interet, 2),
            montant_total=round(montant_total, 2)
        )

        db.session.add(nouveau_pret)
        db.session.commit()

        return redirect(url_for('mes_prets'))

    return render_template('demande_pret.html')


@app.route('/mes-prets')
@login_required
def mes_prets():
    prets = Pret.query.filter_by(client_id=current_user.id).all()
    return render_template('mes_prets.html', prets=prets)


@app.route('/api/calcul-pret', methods=['POST'])
def calcul_pret():
    data = request.json
    montant = float(data['montant'])
    duree = int(data['duree'])
    taux_annuel = 12.0

    taux_mensuel = taux_annuel / 100 / 12
    mensualite = montant * taux_mensuel * (1 + taux_mensuel) ** duree / ((1 + taux_mensuel) ** duree - 1)
    total_rembourser = mensualite * duree
    cout_credit = total_rembourser - montant

    return jsonify({
        'mensualite': round(mensualite, 2),
        'total_rembourser': round(total_rembourser, 2),
        'cout_credit': round(cout_credit, 2)
    })


@app.route('/debug/users')
def debug_users():
    """Route de debug pour voir les utilisateurs en base"""
    users = User.query.all()
    clients = User.query.filter_by(role='client').all()  # ✅ CORRECTION

    result = {
        'users': [{'id': u.id, 'username': u.username, 'role': u.role} for u in users],
        'clients': [{'id': c.id, 'email': c.email, 'nom': c.nom} for c in clients]
    }

    return jsonify(result)


@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    # Vérifier que c'est bien un admin
    if getattr(current_user, 'role', None) != 'admin':
        return redirect(url_for('tableau_de_bord'))

    # Votre code pour le dashboard admin
    stats = calculer_statistiques_globales()
    return render_template('admin_dashboard.html', stats=stats)




@app.route('/client/dashboard')
@login_required
def client_dashboard():
    if current_user.role != 'client':
        return redirect(url_for('tableau_de_bord'))

    # Récupérer le groupe du client
    groupe = None
    if current_user.groupe_id:
        groupe = Groupe.query.get(current_user.groupe_id)

    # Vos statistiques existantes
    stats = calculer_statistiques_utilisateur(current_user)

    # Retourner le template avec toutes les variables nécessaires
    return render_template('client_dashboard.html',
                           user=current_user,
                           stats=stats,
                           groupe=groupe)  # ← Le groupe est maintenant disponible

@app.route('/pret/<int:pret_id>/<action>')
@login_required
def gerer_pret(pret_id, action):
    if getattr(current_user, 'role', None) not in ['admin', 'employe']:
        return redirect(url_for('tableau_de_bord'))

    pret = Pret.query.get_or_404(pret_id)
    client = User.query.get(pret.client_id)

    if action == 'approuver':
        pret.statut = 'approuve'
        pret.date_approbation = datetime.utcnow()

        # 🔔 NOTIFICATION d'approbation
        notification_manager.notifier_approbation_pret(client, pret)

    elif action == 'rejeter':
        pret.statut = 'rejete'
        motif = request.args.get('motif', 'Critères non satisfaits')

        # 🔔 NOTIFICATION de rejet
        notification_manager.notifier_rejet_pret(client, pret, motif)

    db.session.commit()
    return redirect(url_for('prets_en_attente'))


@app.route('/prets-en-attente')
@login_required
def prets_en_attente():
    # Vérifier que c'est un admin ou employé (sans hasattr)
    if getattr(current_user, 'role', None) in ['admin', 'employe']:
        prets = Pret.query.filter_by(statut='en_attente').all()

        # Récupérer les informations clients manuellement
        prets_avec_clients = []
        for pret in prets:
            client = User.query.get(pret.client_id)
            prets_avec_clients.append({
                'pret': pret,
                'client': client
            })

        return render_template('admin_prets_attente.html', prets_avec_clients=prets_avec_clients)
    else:
        return redirect(url_for('tableau_de_bord'))


# ==================== NOUVELLES ROUTES POUR LE DASHBOARD ====================

@app.route('/tableau-de-bord')
@login_required
def tableau_de_bord():
    """Tableau de bord principal avec toutes les fonctionnalités"""
    stats = calculer_statistiques_utilisateur(current_user)
    return render_template('tableau_de_bord.html', user=current_user, stats=stats)



# Routes pour les nouvelles fonctionnalités
@app.route('/score-credit')
@login_required
def score_credit():
    """Page détaillée du score de crédit"""
    return render_template('score_credit.html')

# ==================== ROUTES GAMIFICATION ====================

@app.route('/profil-gamification')
@login_required
def profil_gamification():
    """Page principale de gamification"""
    return render_template('profil-gamification.html')

@app.route('/defis')
@login_required
def defis():
    """Page des défis"""
    defis_list = [
        {'nom': 'Premier prêt', 'description': 'Obtenez votre premier prêt', 'recompense': 100, 'termine': False},
        {'nom': 'Remboursement ponctuel', 'description': '3 remboursements à temps', 'recompense': 50, 'termine': True, 'progression': '2/3'},
        {'nom': 'Leader du groupe', 'description': 'Devenir coordinateur', 'recompense': 200, 'termine': False}
    ]
    return render_template('defis.html', defis=defis_list)

@app.route('/badges')
@login_required
def badges():
    """Page des badges"""
    badges_list = [
        {'nom': 'Bronze', 'icone': '🥉', 'description': 'Premier prêt obtenu', 'obtenu': True},
        {'nom': 'Argent', 'icone': '🥈', 'description': '5 remboursements ponctuels', 'obtenu': False},
        {'nom': 'Or', 'icone': '🥇', 'description': 'Score crédit > 750', 'obtenu': False}
    ]
    return render_template('badges.html', badges=badges_list)

@app.route('/classement')
@login_required
def classement():
    """Page du classement"""
    classement_data = [
        {'position': 1, 'nom': 'Marie Dupont', 'points': 850, 'niveau': 3},
        {'position': 2, 'nom': 'Jean Martin', 'points': 720, 'niveau': 2},
        {'position': 3, 'nom': current_user.prenom + ' ' + current_user.nom, 'points': 650, 'niveau': 1},
        {'position': 4, 'nom': 'Sophie Laurent', 'points': 580, 'niveau': 1}
    ]
    return render_template('classement.html', classement=classement_data)

@app.route('/recompenses')
@login_required
def recompenses():
    """Page des récompenses"""
    recompenses_list = [
        {'nom': 'Réduction taux', 'description': '1% de réduction sur le prochain prêt', 'points': 300, 'disponible': True},
        {'nom': 'Frais de dossier offerts', 'description': 'Frais de dossier gratuits', 'points': 500, 'disponible': False},
        {'nom': 'Assurance gratuite', 'description': '3 mois d\'assurance offerte', 'points': 800, 'disponible': False}
    ]
    return render_template('recompenses.html', recompenses=recompenses_list)

@app.route('/recommandations-pret')
@login_required
def recommandations_pret():
    """Page des recommandations de prêt"""
    return render_template('recommandations_pret.html')

# ==================== API GAMIFICATION ====================

@app.route('/api/gamification/points')
@login_required
def get_gamification_points():
    """API pour récupérer les points de gamification"""
    return jsonify({'points': 650, 'niveau': 1, 'progression': '50%'})

@app.route('/api/gamification/complete-defi/<defi_id>')
@login_required
def complete_defi(defi_id):
    """API pour compléter un défi"""
    # Logique pour compléter un défi
    return jsonify({'success': True, 'points_gagnes': 50})

@app.route('/api/gamification/echanger-recompense/<recompense_id>')
@login_required
def echanger_recompense(recompense_id):
    """API pour échanger une récompense"""
    # Logique pour échanger des points contre une récompense
    return jsonify({'success': True, 'message': 'Récompense échangée'})



@app.route('/reconnaissance-faciale')
@login_required
def reconnaissance_faciale():
    """Gestion de la reconnaissance faciale"""
    return render_template('reconnaissance_faciale.html')


@app.route('/analytics-personnel')
@login_required
def analytics_personnel():
    """Analytics et statistiques du personnel"""
    if not current_user.est_admin and not current_user.est_employe:
        return redirect(url_for('tableau_de_bord'))

    # Vos données d'analytics ici
    stats = calculer_statistiques_globales()
    return render_template('analytics_personnel.html', stats=stats)
#

@app.route('/previsions-remboursement')
@login_required
def previsions_remboursement():
    """Prévisions et calendrier de remboursement"""
    if current_user.est_client:
        # Pour les clients : leurs propres prévisions
        prets = Pret.query.filter_by(client_id=current_user.id, statut='approuve').all()
    else:
        # Pour admin/employé : toutes les prévisions
        prets = Pret.query.filter_by(statut='approuve').all()

    return render_template('previsions_remboursement.html', prets=prets)

@app.route('/notifications')
@login_required
def notifications():
    """Page des notifications utilisateur"""
    notifications = Notification.query.filter_by(utilisateur_id=current_user.id).order_by(Notification.date_creation.desc()).all()
    return render_template('notifications.html', notifications=notifications)


@app.route('/parametres')
@login_required
def parametres():
    """Page des paramètres utilisateur"""
    return render_template('parametres.html')


@app.route('/profil')
@login_required
def profil():
    """Page de profil utilisateur - Version simplifiée"""
    # Calculs basiques sans dépendances complexes
    prets_actifs = Pret.query.filter_by(client_id=current_user.id, statut='approuve').count() if hasattr(current_user,
                                                                                                         'groupe_id') else 0

    stats = {
        'score_credit': 650,
        'score_categorie': 'good',
        'prets_actifs': prets_actifs,
        'montant_actifs': 0,
        'niveau': 1,
        'points': 50,
        'badge': 'Bronze'
    }

    return render_template('profil.html', user=current_user, stats=stats)
# ==================== LANCEMENT ====================

@app.route('/securite')
@login_required
def securite():
    """Page de sécurité et paramètres de compte"""
    return render_template('securite.html')

@app.route('/test-mobile')
def test_mobile():
    return redirect(url_for('test_mobile_routes'))


# Route pour la gestion des remboursements (caissier)
@app.route('/employe/remboursements')
@login_required
def employe_remboursements():
    if not (current_user.role == 'employe' and current_user.has_permission(current_user, 'caissier')):
        return redirect(url_for('tableau_de_bord'))
    return render_template('employe_remboursements.html')

# Route pour l'analyse des prêts (analyste crédit)
@app.route('/employe/analyse-prets')
@login_required
def employe_analyse_prets():
    if not (current_user.role == 'employe' and current_user.has_permission(current_user, 'analyste_credit')):
        return redirect(url_for('tableau_de_bord'))
    return render_template('employe_analyse_prets.html')

# Route pour la gestion des clients (conseiller clientèle)
@app.route('/employe/gestion-clients')
@login_required
def employe_gestion_clients():
    if not (current_user.role == 'employe' and current_user.has_permission(current_user, 'conseiller')):
        return redirect(url_for('tableau_de_bord'))
    return render_template('employe_gestion_clients.html')


@app.route('/admin/creer-employe', methods=['GET', 'POST'])
@login_required
def creer_employe():
    """Créer un nouvel employé"""
    if not current_user.est_admin:
        return redirect(url_for('tableau_de_bord'))

    if request.method == 'POST':
        # Récupérer les données du formulaire
        username = request.form.get('username')
        email = request.form.get('email')
        nom = request.form.get('nom')
        prenom = request.form.get('prenom')
        telephone = request.form.get('telephone')
        password = request.form.get('password')

        # Vérifier si l'utilisateur existe déjà
        if User.query.filter_by(username=username).first():
            return render_template('creer_employe.html', error="Ce nom d'utilisateur existe déjà")

        if User.query.filter_by(email=email).first():
            return render_template('creer_employe.html', error="Cet email existe déjà")

        # Créer le nouvel employé
        nouvel_employe = User(
            username=username,
            email=email,
            nom=nom,
            prenom=prenom,
            telephone=telephone,
            role='employe'
        )
        nouvel_employe.set_password(password)

        db.session.add(nouvel_employe)
        db.session.commit()

        return redirect(url_for('gerer_employes'))

    return render_template('creer_employe.html')
@app.route('/mobile/dashboard')
@login_required
def mobile_dashboard():
    stats = calculer_statistiques_utilisateur(current_user)
    return render_template('mobile_dashboard.html', stats=stats)


# ... votre code existant ...
@app.route("/")
def home():
    return {"message": "✅ API GMES Haiti en ligne et fonctionnelle"}

@app.route('/debug-routes')
def debug_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append(f"{rule.rule} -> {rule.endpoint}")
    return "<br>".join(routes)

@app.route('/list-routes')
def list_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        if 'static' not in rule.rule:
            routes.append(f"{rule.rule} -> {rule.endpoint}")
    return "<br>".sorted(routes)


@app.route('/client/test')
@login_required
def client_test():
    """Route temporaire pour tester l'interface client"""
    if current_user.role != 'client':
        return f"⚠️ Accès refusé. Votre rôle: {current_user.role}"

    return """
    <h1>✅ Interface Client Fonctionnelle</h1>
    <p>Bienvenue {}</p>
    <p>Votre rôle: {}</p>
    <a href="/client/dashboard">Accéder au tableau de bord complet</a>
    """.format(current_user.nom_complet, current_user.role)


@app.route('/admin/debug-stats')
@login_required
def debug_stats():
    if current_user.role != 'admin':
        return redirect(url_for('tableau_de_bord'))

    stats = calculer_statistiques_globales()

    # Debug détaillé
    debug_info = {
        'stats_object': stats,
        'clients_count': User.query.filter_by(role='client').count(),
        'active_loans': Pret.query.filter_by(statut='approuve').count(),
        'pending_loans': Pret.query.filter_by(statut='en_attente').count()
    }

    return jsonify(debug_info)


@app.route('/admin/gerer-permissions/<int:employe_id>', methods=['GET', 'POST'])
@login_required
def gerer_permissions(employe_id):
    if current_user.role != 'admin':
        return redirect(url_for('tableau_de_bord'))

    employe = User.query.get_or_404(employe_id)

    if request.method == 'POST':
        permissions = request.form.getlist('permissions')
        employe.permissions = json.dumps(permissions)
        db.session.commit()
        return redirect(url_for('gerer_employes'))

    # Permissions disponibles
    all_permissions = {
        'caissier': 'Caissier - Gestion des remboursements',
        'analyste_credit': 'Analyste crédit - Analyse des prêts',
        'conseiller': 'Conseiller clientèle - Gestion clients',
        'gestionnaire_groupe': 'Gestionnaire de groupes',
        'rapports': 'Génération de rapports'
    }

    current_permissions = json.loads(employe.permissions) if employe.permissions else []

    return render_template('gerer_permissions.html',
                           employe=employe,
                           all_permissions=all_permissions,
                           current_permissions=current_permissions)

@app.template_filter('has_permission')
def has_permission_filter(user, permission_name):
    return current_user.has_permission(user, permission_name)


@app.route('/debug/all-users')
def debug_all_users():
    """Voir tous les utilisateurs en base"""
    users = User.query.all()
    result = []
    for user in users:
        result.append({
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'role': user.role,
            'nom': user.nom,
            'prenom': user.prenom,
            'has_password': bool(user.password_hash)
        })
    return jsonify(result)


@app.route('/debug/create-employe-now')
def create_employe_now():
    """Créer un employé immédiatement"""
    if User.query.filter_by(username='employe').first():
        return "❌ Employé existe déjà"

    employe = User(
        username='employe',
        email='employe@gmes.com',
        role='employe',
        nom='Martin',
        prenom='Sophie',
        telephone='+50912345678'
    )
    employe.set_password('employe123')  # Mot de passe simple

    db.session.add(employe)
    db.session.commit()

    return """
    ✅ Employé créé avec succès !
    <br><br>
    <strong>Identifiants de test :</strong>
    <br>Identifiant: <strong>employe</strong>
    <br>OU Email: <strong>employe@gmes.com</strong>  
    <br>Mot de passe: <strong>employe123</strong>
    <br><br>
    <a href="/connexion" style="background: blue; color: white; padding: 10px; text-decoration: none;">
    🚀 Se connecter maintenant
    </a>
    """

@app.route('/admin/gerer-employes')
@login_required
def gerer_employes():
    if current_user.role != 'admin':
        return redirect(url_for('tableau_de_bord'))

    # Récupérer tous les employés et superviseurs
    utilisateurs = User.query.filter(User.role.in_(['employe', 'superviseur'])).all()

    # Calculer les statistiques
    stats = {
        'total': len(utilisateurs),
        'en_attente': len([u for u in utilisateurs if u.statut == 'en_attente']),
        'actifs': len([u for u in utilisateurs if u.statut == 'actif']),
        'suspendus': len([u for u in utilisateurs if u.statut == 'suspendu']),
        'employes': len([u for u in utilisateurs if u.role == 'employe']),
        'superviseurs': len([u for u in utilisateurs if u.role == 'superviseur'])
    }

    return render_template('gerer_employes.html', utilisateurs=utilisateurs, stats=stats)

# ✅ APPROUVER un employé
@app.route('/admin/approver-employe/<int:employe_id>')
@login_required
def approver_employe(employe_id):
    if current_user.role != 'admin':
        return redirect(url_for('tableau_de_bord'))

    employe = User.query.get_or_404(employe_id)

    if employe.role != 'employe':
        return redirect(url_for('gerer_employes'))

    # Approuver l'employé
    employe.statut = 'actif'
    employe.approuve_par = current_user.id
    employe.date_approbation = datetime.utcnow()

    db.session.commit()

    print(f"✅ Employé {employe.prenom} {employe.nom} approuvé par {current_user.prenom}")
    return redirect(url_for('gerer_employes'))


# ⏸️ SUSPENDRE un employé
@app.route('/admin/suspendre-employe/<int:employe_id>')
@login_required
def suspendre_employe(employe_id):
    if current_user.role != 'admin':
        return redirect(url_for('tableau_de_bord'))

    employe = User.query.get_or_404(employe_id)

    if employe.role != 'employe':
        return redirect(url_for('gerer_employes'))

    # Suspendre l'employé
    employe.statut = 'suspendu'
    db.session.commit()

    print(f"⏸️ Employé {employe.prenom} {employe.nom} suspendu par {current_user.prenom}")
    return redirect(url_for('gerer_employes'))


# 🔄 RÉACTIVER un employé
@app.route('/admin/reactiver-employe/<int:employe_id>')
@login_required
def reactiver_employe(employe_id):
    if current_user.role != 'admin':
        return redirect(url_for('tableau_de_bord'))

    employe = User.query.get_or_404(employe_id)

    if employe.role != 'employe':
        return redirect(url_for('gerer_employes'))

    # Réactiver l'employé
    employe.statut = 'actif'
    db.session.commit()

    print(f"🔄 Employé {employe.prenom} {employe.nom} réactivé par {current_user.prenom}")
    return redirect(url_for('gerer_employes'))


# ✏️ MODIFIER un employé
@app.route('/admin/modifier-employe/<int:employe_id>', methods=['GET', 'POST'])
@login_required
def modifier_employe(employe_id):
    if current_user.role != 'admin':
        return redirect(url_for('tableau_de_bord'))

    employe = User.query.get_or_404(employe_id)

    if employe.role != 'employe':
        return redirect(url_for('gerer_employes'))

    fonctions_disponibles = {
        'caissier': 'Caissier - Gestion des remboursements',
        'analyste_credit': 'Analyste crédit - Analyse des prêts',
        'conseiller': 'Conseiller clientèle - Gestion clients',
        'gestionnaire_groupe': 'Gestionnaire de groupes',
        'rapports': 'Génération de rapports'
    }

    if request.method == 'POST':
        # Mettre à jour les informations
        employe.username = request.form.get('username')
        employe.email = request.form.get('email')
        employe.nom = request.form.get('nom')
        employe.prenom = request.form.get('prenom')
        employe.telephone = request.form.get('telephone')
        employe.fonction = request.form.get('fonction')
        employe.statut = request.form.get('statut')

        # Si mot de passe fourni, le mettre à jour
        nouveau_password = request.form.get('password')
        if nouveau_password:
            employe.set_password(nouveau_password)

        db.session.commit()
        return redirect(url_for('gerer_employes'))

    return render_template('modifier_employe.html',
                           employe=employe,
                           fonctions=fonctions_disponibles)


# 🗑️ SUPPRIMER un employé
@app.route('/admin/supprimer-employe/<int:employe_id>')
@login_required
def supprimer_employe(employe_id):
    if current_user.role != 'admin':
        return redirect(url_for('tableau_de_bord'))

    employe = User.query.get_or_404(employe_id)

    if employe.role != 'employe':
        return redirect(url_for('gerer_employes'))

    # Supprimer l'employé
    db.session.delete(employe)
    db.session.commit()

    print(f"🗑️ Employé {employe.prenom} {employe.nom} supprimé par {current_user.prenom}")
    return redirect(url_for('gerer_employes'))


# 📊 GÉNÉRATION DE RAPPORTS
@app.route('/employe/rapports')
@login_required
def rapports_dashboard():
    if current_user.role != 'employe' or current_user.fonction != 'rapports':
        return redirect(url_for('tableau_de_bord'))

    # Statistiques pour les rapports
    stats_rapports = {
        'rapports_generes': 45,
        'export_reussis': 38,
        'rapports_urgents': 3
    }

    return render_template('rapports_dashboard.html', stats=stats_rapports)


@app.route('/employe/caissier')
@login_required
def caissier_dashboard():
    """Tableau de bord spécifique au caissier"""
    if current_user.role != 'employe' or not current_user.has_permission('caissier'):
        return redirect(url_for('employe_dashboard'))
    # Remboursements du jour
    aujourdhui = datetime.utcnow().date()
    remboursements_du_jour = Remboursement.query.filter(
        db.func.date(Remboursement.date_remboursement) == aujourdhui
    ).all()
    # Calculer les statistiques
    montant_total = sum(r.montant for r in remboursements_du_jour)

    stats = {
        'remboursements_du_jour': len(remboursements_du_jour),
        'montant_total': montant_total,
        'remboursements_attente': Remboursement.query.filter_by(statut='en_attente').count(),
        'taux_service': 95  # À calculer dynamiquement
    }
    # Remboursements en retard (simulation)
    remboursements_retard = []
    return render_template('caissier_dashboard.html',
                           stats=stats,
                           remboursements_du_jour=remboursements_du_jour,
                           remboursements_retard=remboursements_retard)


# Route Conseiller avec données
@app.route('/employe/conseiller')
@login_required
def conseiller_dashboard():
    if current_user.role != 'employe' or current_user.fonction != 'conseiller':
        return redirect(url_for('tableau_de_bord'))

    clients = User.query.filter_by(role='client').limit(6).all()

    return render_template('conseiller_dashboard.html',
                           clients=clients,
                           total_clients=len(clients),
                           dossiers_actifs=15,
                           demandes_attente=3,
                           rdv_aujourdhui=2)


# Route Analyste avec données
@app.route('/employe/analyste')
@login_required
def analyste_dashboard():
    if current_user.role != 'employe' or current_user.fonction != 'analyste_credit':
        return redirect(url_for('tableau_de_bord'))

    prets_en_attente = Pret.query.filter_by(statut='en_attente').all()

    return render_template('analyste_dashboard.html',
                           prets_en_attente=prets_en_attente,
                           prets_traites=24,
                           taux_approbation=78.5,
                           delai_moyen="4.2")


# Route Gestionnaire avec données
@app.route('/employe/gestionnaire')
@login_required
def gestionnaire_dashboard():
    if current_user.role != 'employe' or current_user.fonction != 'gestionnaire':
        return redirect(url_for('tableau_de_bord'))

    groupes = Groupe.query.all()

    return render_template('gestionnaire_dashboard.html',
                           groupes=groupes,
                           total_groupes=len(groupes),
                           total_membres=User.query.filter_by(role='client').count(),
                           performance_moyenne=85.2,
                           nouveaux_membres=12)

#

# ✅ APPROUVER un employé/superviseur
@app.route('/admin/approver-utilisateur/<int:user_id>')
@login_required
def approver_utilisateur(user_id):
    if current_user.role != 'admin':
        return redirect(url_for('tableau_de_bord'))

    utilisateur = User.query.get_or_404(user_id)

    if utilisateur.role not in ['employe', 'superviseur']:
        return redirect(url_for('gerer_employes'))

    # Approuver l'utilisateur
    utilisateur.statut = 'actif'
    utilisateur.approuve_par = current_user.id
    utilisateur.date_approbation = datetime.utcnow()

    db.session.commit()

    print(f"✅ {utilisateur.role} {utilisateur.prenom} {utilisateur.nom} approuvé par {current_user.prenom}")
    return redirect(url_for('gerer_employes'))


# ⏸️ SUSPENDRE un utilisateur
@app.route('/admin/suspendre-utilisateur/<int:user_id>')
@login_required
def suspendre_utilisateur(user_id):
    if current_user.role != 'admin':
        return redirect(url_for('tableau_de_bord'))

    utilisateur = User.query.get_or_404(user_id)

    if utilisateur.role not in ['employe', 'superviseur']:
        return redirect(url_for('gerer_employes'))

    # Suspendre l'utilisateur
    utilisateur.statut = 'suspendu'
    db.session.commit()

    print(f"⏸️ {utilisateur.role} {utilisateur.prenom} {utilisateur.nom} suspendu")
    return redirect(url_for('gerer_employes'))




# 👥 VOIR TOUS LES EMPLOYÉS
@app.route('/superviseur/employes')
@login_required
def superviseur_tous_employes():
    if current_user.role != 'superviseur':
        return redirect(url_for('tableau_de_bord'))

    employes = User.query.filter_by(role='employe').all()
    return render_template('superviseur_tous_employes.html', employes=employes)


# 🏦 VOIR PAR FONCTION
@app.route('/superviseur/fonction/<fonction>')
@login_required
def superviseur_voir_fonction(fonction):
    if current_user.role != 'superviseur':
        return redirect(url_for('tableau_de_bord'))

    employes = User.query.filter_by(role='employe', fonction=fonction).all()

    # Libellé de la fonction
    libelles_fonctions = {
        'caissier': 'Caissiers',
        'conseiller': 'Conseillers Clientèle',
        'analyste_credit': 'Analystes Crédit',
        'gestionnaire_groupe': 'Gestionnaires de Groupes',
        'rapports': 'Générateurs de Rapports'
    }

    return render_template('superviseur_par_fonction.html',
                           employes=employes,
                           fonction=fonction,
                           libelle_fonction=libelles_fonctions.get(fonction, fonction))


# 👤 VOIR DÉTAILS EMPLOYÉ
@app.route('/superviseur/employe/<int:employe_id>')
@login_required
def superviseur_voir_employe(employe_id):
    """Page de détail d'un employé"""
    if current_user.role != 'superviseur':
        return redirect(url_for('tableau_de_bord'))

    employe = User.query.get_or_404(employe_id)

    # Vérifier que c'est bien un employé
    if employe.role != 'employe':
        return redirect(url_for('superviseur_dashboard'))

    # Calculer les statistiques
    stats = calculer_stats_employe(employe)

    return render_template('superviseur_voir_employe.html',
                           employe=employe,
                           stats=stats)

# 📊 RAPPORTS PERFORMANCE
@app.route('/superviseur/rapports')
@login_required
def superviseur_rapports():
    if current_user.role != 'superviseur':
        return redirect(url_for('tableau_de_bord'))

    # Données de performance pour le template superviseur_rapports.html
    performances = {
        'caissier': {'nombre': 5, 'employes_actifs': 4, 'performance_moyenne': 85, 'taux_activite': 92},
        'conseiller': {'nombre': 8, 'employes_actifs': 7, 'performance_moyenne': 78, 'taux_activite': 88},
        'analyste_credit': {'nombre': 3, 'employes_actifs': 3, 'performance_moyenne': 91, 'taux_activite': 95},
        'gestionnaire_groupe': {'nombre': 4, 'employes_actifs': 4, 'performance_moyenne': 82, 'taux_activite': 90},
        'rapports': {'nombre': 2, 'employes_actifs': 2, 'performance_moyenne': 88, 'taux_activite': 85}
    }

    return render_template('superviseur_rapports.html', performances=performances)


@app.route('/superviseur/employes')
@login_required
def superviseur_employes():
    if current_user.role != 'superviseur':
        return redirect(url_for('tableau_de_bord'))

    employes = User.query.filter_by(role='employe').all()
    return render_template('superviseur_employes.html', employes=employes)


@app.route('/superviseur/dashboard')
@login_required
def superviseur_dashboard():
    if current_user.role != 'superviseur':
        return redirect(url_for('tableau_de_bord'))

    try:
        # Statistiques globales employés
        total_employes = User.query.filter_by(role='employe').count()
        employes_actifs = User.query.filter_by(role='employe', statut='actif').count()
        employes_attente = User.query.filter_by(role='employe', statut='en_attente').count()

        # Compter par fonction - FILTRER les fonctions None
        employes_par_fonction = db.session.query(
            User.fonction,
            db.func.count(User.id)
        ).filter(
            User.role == 'employe',
            User.fonction.isnot(None),  # ← FILTRE IMPORTANT
            User.fonction != ''         # ← FILTRE IMPORTANT
        ).group_by(User.fonction).all()

        # Tâches en retard
        taches_retard = 2

        return render_template('superviseur_dashboard.html',
                               total_employes=total_employes,
                               employes_actifs=employes_actifs,
                               employes_attente=employes_attente,
                               employes_par_fonction=employes_par_fonction,
                               taches_retard=taches_retard)

    except Exception as e:
        print(f"❌ Erreur superviseur dashboard: {e}")
        return render_template('superviseur_dashboard.html',
                               total_employes=0,
                               employes_actifs=0,
                               employes_attente=0,
                               employes_par_fonction=[],
                               taches_retard=0)


@app.route('/admin/ajouter-employe', methods=['GET', 'POST'])
@login_required
def ajouter_employe():
    if current_user.role != 'admin':
        return redirect(url_for('tableau_de_bord'))

    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        nom = request.form.get('nom')
        prenom = request.form.get('prenom')
        telephone = request.form.get('telephone')
        password = request.form.get('password')
        role = request.form.get('role')  # employe ou superviseur
        fonction = request.form.get('fonction')

        # ⚠️ TOUJOURS en attente par défaut
        statut = 'en_attente'

        # Vérifier si l'utilisateur existe déjà
        if User.query.filter_by(username=username).first():
            return render_template('ajouter_employe.html',
                                   error="Ce nom d'utilisateur existe déjà")

        if User.query.filter_by(email=email).first():
            return render_template('ajouter_employe.html',
                                   error="Cet email existe déjà")

        # Créer le nouvel utilisateur
        nouvel_utilisateur = User(
            username=username,
            email=email,
            nom=nom,
            prenom=prenom,
            telephone=telephone,
            role=role,
            fonction=fonction,
            statut=statut  # ⚠️ TOUJOURS en attente
        )
        nouvel_utilisateur.set_password(password)

        db.session.add(nouvel_utilisateur)
        db.session.commit()

        # 🔔 Notification pour l'admin
        print(f"✅ {role.capitalize()} {prenom} {nom} créé en attente d'approbation")

        return redirect(url_for('gerer_employes'))

    return render_template('ajouter_employe.html')


# 👥 VOIR TOUS LES EMPLOYÉS (version superviseur)


# 📊 RAPPORTS PERFORMANCE

# 📝 JOURNAL DES ACTIVITÉS
@app.route('/superviseur/activites')
@login_required
def superviseur_activites():
    if current_user.role != 'superviseur':
        return redirect(url_for('tableau_de_bord'))

    return "<h1>📝 Journal des Activités - En construction</h1><p>Cette fonctionnalité sera disponible prochainement.</p>"


@app.route('/superviseur/init-fonctions')
@login_required
def init_fonctions():
    if current_user.role != 'superviseur':
        return redirect(url_for('tableau_de_bord'))

    try:
        employes = User.query.filter_by(role='employe').all()
        fonctions_disponibles = ['caissier', 'conseiller', 'analyste_credit', 'gestionnaire_groupe', 'rapports']

        for i, employe in enumerate(employes):
            if not employe.fonction:
                # Assigner une fonction cycliquement
                fonction = fonctions_disponibles[i % len(fonctions_disponibles)]
                employe.fonction = fonction
                print(f"✅ {employe.prenom} {employe.nom} -> {fonction}")

        db.session.commit()
        return "✅ Fonctions initialisées avec succès!"

    except Exception as e:
        db.session.rollback()
        return f"❌ Erreur: {e}"


@app.route('/superviseur/debug-fonctions')
@login_required
def debug_fonctions():
    if current_user.role != 'superviseur':
        return redirect(url_for('tableau_de_bord'))

    # Vérifier tous les employés et leurs fonctions
    employes = User.query.filter_by(role='employe').all()

    result = "<h1>Debug Fonctions Employés</h1>"
    for emp in employes:
        result += f"<p>{emp.prenom} {emp.nom} - Fonction: '{emp.fonction}' - Statut: {emp.statut}</p>"

    # Vérifier le regroupement par fonction
    fonctions = db.session.query(
        User.fonction,
        db.func.count(User.id)
    ).filter_by(role='employe').group_by(User.fonction).all()

    result += "<h2>Groupement par fonction:</h2>"
    for fonction, count in fonctions:
        result += f"<p>Fonction '{fonction}': {count} employé(s)</p>"

    return result

@app.route('/superviseur/')
@login_required
def superviseur_index():
    """Redirection vers le dashboard superviseur"""
    if current_user.role != 'superviseur':
        return redirect(url_for('tableau_de_bord'))
    return redirect(url_for('superviseur_dashboard'))


# ==================== ROUTES MANQUANTES ====================

@app.route('/api/notifications/count')
@login_required
def api_notifications_count():
    """API pour compter les notifications non lues"""
    try:
        count = Notification.query.filter_by(
            utilisateur_id=current_user.id,
            lue=False
        ).count()
        return jsonify({'count': count})
    except Exception as e:
        print(f"Erreur notifications count: {e}")
        return jsonify({'count': 0})


@app.route('/init-fonctions-employes')
def init_fonctions_employes():
    """Initialiser les fonctions des employés existants"""
    try:
        employes = User.query.filter_by(role='employe').all()
        fonctions = ['caissier', 'conseiller', 'analyste_credit', 'gestionnaire_groupe', 'rapports']

        results = []
        for i, employe in enumerate(employes):
            if not employe.fonction:
                employe.fonction = fonctions[i % len(fonctions)]
                results.append(f"✅ {employe.prenom} {employe.nom} -> {employe.fonction}")

        db.session.commit()

        html_response = "<h1>Fonctions employés initialisées!</h1>"
        for result in results:
            html_response += f"<p>{result}</p>"

        html_response += "<br><a href='/superviseur/dashboard'>Aller au dashboard superviseur</a>"
        return html_response

    except Exception as e:
        db.session.rollback()
        return f"❌ Erreur: {str(e)}"


@app.route('/debug/employes-fonctions')
def debug_employes_fonctions():
    """Debug des fonctions des employés"""
    employes = User.query.filter_by(role='employe').all()
    result = "<h1>Employés et leurs fonctions</h1>"

    if not employes:
        result += "<p>Aucun employé trouvé</p>"
    else:
        for emp in employes:
            result += f"""
            <div style="border: 1px solid #ccc; padding: 10px; margin: 5px;">
                <strong>{emp.prenom} {emp.nom}</strong><br>
                Email: {emp.email}<br>
                Fonction: <span style="color: {'green' if emp.fonction else 'red'}">
                    {emp.fonction if emp.fonction else 'NON DÉFINIE'}
                </span><br>
                Statut: {emp.statut}
            </div>
            """

    result += "<br><a href='/init-fonctions-employes'>Initialiser les fonctions</a>"
    return result


@app.route('/create-superviseur-test')
def create_superviseur_test():
    """Créer un compte superviseur de test"""
    if User.query.filter_by(username='superviseur').first():
        return """
        <h1>Superviseur existe déjà!</h1>
        <p>Identifiant: <strong>superviseur</strong></p>
        <p>Mot de passe: <strong>superviseur123</strong></p>
        <br>
        <a href="/connexion">Se connecter</a>
        """

    superviseur = User(
        username='superviseur',
        email='superviseur@gmes.com',
        role='superviseur',
        nom='Superviseur',
        prenom='Test',
        telephone='+50900000001',
        fonction='superviseur',
        statut='actif'
    )
    superviseur.set_password('superviseur123')

    db.session.add(superviseur)
    db.session.commit()

    return """
    <h1>✅ Superviseur créé !</h1>
    <p>Identifiant: <strong>superviseur</strong></p>
    <p>Mot de passe: <strong>superviseur123</strong></p>
    <p>Email: <strong>superviseur@gmes.com</strong></p>
    <br>
    <a href="/connexion" style="background: blue; color: white; padding: 10px; text-decoration: none;">
    🚀 Se connecter maintenant
    </a>
    """


def calculer_stats_employe(employe):
    """Calcule les statistiques d'un employé"""
    stats = {
        'performance': 85,  # Valeur par défaut
        'taches_terminees': 0,
        'taches_en_cours': 0,
        'satisfaction_client': 4.2,
        'activite_recente': 'Élevée'
    }

    # Selon la fonction de l'employé, calculer des stats spécifiques
    if employe.fonction == 'caissier':
        stats['taches_terminees'] = Remboursement.query.filter(
            db.func.date(Remboursement.date_remboursement) == datetime.utcnow().date()
        ).count()
        stats['taches_en_cours'] = 3
        stats['performance'] = min(100, stats['taches_terminees'] * 10 + 50)

    elif employe.fonction == 'conseiller':
        stats['taches_terminees'] = User.query.filter_by(role='client').count()
        stats['taches_en_cours'] = 5
        stats['performance'] = 78

    elif employe.fonction == 'analyste_credit':
        stats['taches_terminees'] = Pret.query.filter_by(statut='approuve').count()
        stats['taches_en_cours'] = Pret.query.filter_by(statut='en_attente').count()
        stats['performance'] = 91

    elif employe.fonction == 'gestionnaire_groupe':
        stats['taches_terminees'] = Groupe.query.count()
        stats['taches_en_cours'] = 2
        stats['performance'] = 82

    elif employe.fonction == 'rapports':
        stats['taches_terminees'] = 15
        stats['taches_en_cours'] = 3
        stats['performance'] = 88

    return stats


# def has_permission(user, permission_name):
#     """Vérifie si un employé a une permission spécifique"""
#     if not user:
#         return False
#
#     if user.role == 'admin':
#         return True
#     elif user.role == 'employe':
#         if user.permissions:
#             try:
#                 permissions_list = json.loads(user.permissions)
#                 return permission_name in permissions_list
#             except:
#                 return False
#         return False
#     elif user.role == 'superviseur':
#         # Les superviseurs ont accès à tout
#         return True
#     return False

# def has_permission(user, permission_name):
#     """Vérifie si un utilisateur a une permission spécifique"""
#     if not user:
#         return False
#
#     # Admin a accès à tout
#     if user.role == 'admin':
#         return True
#
#     # Superviseur a accès à tous les employés et leur travail
#     elif user.role == 'superviseur':
#         # Les superviseurs peuvent gérer tous les employés
#         if permission_name in ['caissier', 'conseiller', 'analyste_credit', 'gestionnaire_groupe', 'rapports']:
#             return True
#         # Mais pas accéder aux fonctions admin
#         elif permission_name in ['gerer_superviseurs', 'gerer_admins']:
#             return False
#         return True
#
#     # Employé a accès seulement à ses permissions spécifiques
#     elif user.role == 'employe':
#         if user.permissions:
#             try:
#                 permissions_list = json.loads(user.permissions)
#                 return permission_name in permissions_list
#             except:
#                 return False
#         return False
#
#     # Client n'a pas de permissions spéciales
#     return False
#


@app.route('/employe/dashboard')
@login_required
def employe_dashboard():
    if current_user.role != 'employe':
        return redirect(url_for('tableau_de_bord'))

    # Récupérer les permissions de l'employé
    permissions = []
    if current_user.permissions:
        try:
            permissions = json.loads(current_user.permissions)
        except:
            permissions = []

    # ✅ CORRECTION : Utiliser la FONCTION has_permission()
    stats = {}

    if current_user.has_permission('caissier'):
        stats['remboursements_du_jour'] = Remboursement.query.filter(
            db.func.date(Remboursement.date_remboursement) == datetime.utcnow().date()
        ).count()

    if current_user.has_permission(current_user, 'analyste_credit'):
        stats['prets_en_attente'] = Pret.query.filter_by(statut='en_attente').count()

    # Statistiques communes
    stats.update({
        'clients_assignes': User.query.filter_by(role='client').count() if current_user.has_permission('conseiller') else 0,
        'groupes_geres': Groupe.query.count() if current_user.has_permission('gestionnaire_groupe') else 0,
        'rapports_generes': 12 if current_user.has_permission('rapports') else 0
    })

    return render_template('employe_dashboard.html',
                           permissions=permissions,
                           stats=stats)


@app.route('/dashboard')
@login_required
def dashboard_redirect():
    """Redirige chaque utilisateur vers son dashboard approprié"""
    if current_user.role == 'admin':
        return redirect(url_for('admin_dashboard'))
    elif current_user.role == 'superviseur':
        return redirect(url_for('superviseur_dashboard'))
    elif current_user.role == 'employe':
        # ✅ CORRECTION : Enlever le premier current_user
        if current_user.has_permission('caissier'):
            return redirect(url_for('caissier_dashboard'))
        elif current_user.has_permission('conseiller'):
            return redirect(url_for('conseiller_dashboard'))
        elif current_user.has_permission('analyste_credit'):
            return redirect(url_for('analyste_dashboard'))
        elif current_user.has_permission('gestionnaire_groupe'):
            return redirect(url_for('gestionnaire_dashboard'))
        elif current_user.has_permission('rapports'):
            return redirect(url_for('rapports_dashboard'))
        else:
            return redirect(url_for('employe_dashboard'))
    elif current_user.role == 'client':
        return redirect(url_for('client_dashboard'))
    else:
        return redirect(url_for('tableau_de_bord'))

@app.route('/activate-all-employes')
def activate_all_employes():
    """Activer tous les employés en attente"""
    if current_user.role != 'admin' and current_user.role != 'superviseur':
        return "Accès non autorisé"

    employes = User.query.filter_by(role='employe', statut='en_attente').all()

    for employe in employes:
        employe.statut = 'actif'
        employe.approuve_par = current_user.id
        employe.date_approbation = datetime.utcnow()
        print(f"✅ {employe.prenom} {employe.nom} activé")

    db.session.commit()

    return f"✅ {len(employes)} employé(s) activé(s)!"


@app.route('/mes-groupes')
@login_required
def mes_groupes():
    """Voir mon groupe - Pour employés seulement"""
    if current_user.role != 'employe':
        return redirect(url_for('tableau_de_bord'))

    # Si l'employé a un groupe_id, montrer seulement son groupe
    if current_user.groupe_id:
        groupe = Groupe.query.get(current_user.groupe_id)
        return render_template('mon_groupe.html', groupe=groupe)
    else:
        return render_template('mon_groupe.html', groupe=None)


if __name__ == '__main__':
    with app.app_context():
        initialiser_donnees()

    # Configuration pour Render
    import os

    port = int(os.environ.get("PORT", 5000))

    print("🚀 GMES Microcrédit - Prêt pour la production")
    print(f"🌐 URL: https://votre-app.onrender.com")

    app.run(host="0.0.0.0", port=port, debug=False)