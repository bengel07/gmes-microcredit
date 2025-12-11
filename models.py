from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from database import db


class Client(db.Model):
    __tablename__ = 'clients'
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True)
    code_client = db.Column(db.String(20), unique=True)
    nom = db.Column(db.String(100))
    prenom = db.Column(db.String(100))
    telephone = db.Column(db.String(20))
    email = db.Column(db.String(120))
    adresse = db.Column(db.Text)
    cin = db.Column(db.String(50))
    date_naissance = db.Column(db.DateTime)
    profession = db.Column(db.String(100))
    revenu_mensuel = db.Column(db.Float)
    date_inscription = db.Column(db.DateTime, default=datetime.utcnow)
    statut = db.Column(db.String(20), default='actif')
    mot_de_passe_hash = db.Column(db.String(255))
    groupe_id = db.Column(db.Integer)

    def definir_mot_de_passe(self, mot_de_passe):
        self.mot_de_passe_hash = generate_password_hash(mot_de_passe)

    def verifier_mot_de_passe(self, mot_de_passe):
        return check_password_hash(self.mot_de_passe_hash, mot_de_passe)

    def get_id(self):
        return self.id

    @property
    def is_authenticated(self):
        return True

    @property
    def is_active(self):
        return self.statut == 'actif'

    @property
    def is_anonymous(self):
        return False


class Groupe(db.Model):
    __tablename__ = 'groupes'
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True)
    nom = db.Column(db.String(100))
    code_groupe = db.Column(db.String(20), unique=True)
    zone = db.Column(db.String(100))
    date_creation = db.Column(db.DateTime, default=datetime.utcnow)
    statut = db.Column(db.String(20), default='actif')
    responsable_id = db.Column(db.Integer)

    def get_id(self):
        return self.id

    @property
    def is_authenticated(self):
        return True

    @property
    def is_active(self):
        return self.statut == 'actif'

    @property
    def is_anonymous(self):
        return False


class Pret(db.Model):
    __tablename__ = 'prets'
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True)
    client_id = db.Column(db.Integer)
    groupe_id = db.Column(db.Integer)
    montant = db.Column(db.Float)
    taux_interet = db.Column(db.Float)
    duree_mois = db.Column(db.Integer)
    date_demande = db.Column(db.DateTime, default=datetime.utcnow)
    date_approbation = db.Column(db.DateTime)
    statut = db.Column(db.String(20), default='en_attente')
    motif = db.Column(db.String(100))
    montant_interet = db.Column(db.Float)
    montant_total = db.Column(db.Float)
    mensualite = db.Column(db.Float)

    def get_id(self):
        return self.id

    @property
    def is_authenticated(self):
        return True

    @property
    def is_active(self):
        return True

    @property
    def is_anonymous(self):
        return False


class Remboursement(db.Model):
    __tablename__ = 'remboursements'
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True)
    pret_id = db.Column(db.Integer)
    client_id = db.Column(db.Integer)
    montant = db.Column(db.Float)
    date_remboursement = db.Column(db.DateTime, default=datetime.utcnow)
    date_echeance = db.Column(db.DateTime)
    statut = db.Column(db.String(20), default='en_attente')
    type_paiement = db.Column(db.String(20))

    def get_id(self):
        return self.id

    @property
    def is_authenticated(self):
        return True

    @property
    def is_active(self):
        return True

    @property
    def is_anonymous(self):
        return False


class Employe(db.Model):
    __tablename__ = 'employes'
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True)
    matricule = db.Column(db.String(20), unique=True)
    nom = db.Column(db.String(100))
    prenom = db.Column(db.String(100))
    email = db.Column(db.String(120))
    telephone = db.Column(db.String(20))
    poste = db.Column(db.String(100))
    date_embauche = db.Column(db.DateTime)
    statut = db.Column(db.String(20), default='actif')
    mot_de_passe_hash = db.Column(db.String(255))

    def definir_mot_de_passe(self, mot_de_passe):
        self.mot_de_passe_hash = generate_password_hash(mot_de_passe)

    def verifier_mot_de_passe(self, mot_de_passe):
        return check_password_hash(self.mot_de_passe_hash, mot_de_passe)

    def get_id(self):
        return self.id

    @property
    def is_authenticated(self):
        return True

    @property
    def is_active(self):
        return self.statut == 'actif'

    @property
    def is_anonymous(self):
        return False


class Admin(db.Model):
    __tablename__ = 'admins'
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True)
    nom_utilisateur = db.Column(db.String(80), unique=True)
    email = db.Column(db.String(120))
    mot_de_passe_hash = db.Column(db.String(255))
    role = db.Column(db.String(50), default='gestionnaire')
    date_creation = db.Column(db.DateTime, default=datetime.utcnow)

    def definir_mot_de_passe(self, mot_de_passe):
        self.mot_de_passe_hash = generate_password_hash(mot_de_passe)

    def verifier_mot_de_passe(self, mot_de_passe):
        return check_password_hash(self.mot_de_passe_hash, mot_de_passe)

    def get_id(self):
        return self.id

    @property
    def is_authenticated(self):
        return True

    @property
    def is_active(self):
        return True

    @property
    def is_anonymous(self):
        return False


def show_loan_recommendation(self, e=None):
    """Recommandations de prêt personnalisées"""
    try:
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.get(f"{self.api_base_url}/recommandations-pret", headers=headers)

        if response.status_code == 200:
            data = response.json()

            view = ft.Column([
                ft.Row([
                    ft.IconButton(icon=ft.icons.ARROW_BACK, on_click=lambda _: self.show_dashboard()),
                    ft.Text("Recommandations", size=20, weight=ft.FontWeight.BOLD)
                ]),

                # Score de crédit
                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.Row([
                                ft.Text("🎯 Score de Crédit", size=18, weight=ft.FontWeight.BOLD),
                                ft.Container(
                                    content=ft.Text(
                                        f"{data['score']}/850",
                                        color=ft.colors.WHITE,
                                        weight=ft.FontWeight.BOLD
                                    ),
                                    bgcolor=self.get_score_color(data['score']),
                                    padding=10,
                                    border_radius=20
                                )
                            ]),
                            ft.Text(f"Catégorie: {data['categorie']}"),

                            # Facteurs d'influence
                            ft.Text("Facteurs influençant votre score:", size=14, weight=ft.FontWeight.BOLD),
                            *[ft.Text(f"• {factor}") for factor in data['facteurs']]
                        ]),
                        padding=20
                    )
                ),

                # Prêts recommandés
                ft.Text("Prêts Recommandés", size=16, weight=ft.FontWeight.BOLD),
                *[self.create_loan_recommendation_card(pret) for pret in data['prets_recommandes']],

                # Améliorer son score
                ft.ExpansionTile(
                    title=ft.Text("💡 Comment améliorer votre score?"),
                    controls=[
                        ft.ListTile(title=ft.Text("• Effectuez vos remboursements à temps")),
                        ft.ListTile(title=ft.Text("• Maintenez une activité régulière")),
                        ft.ListTile(title=ft.Text("• Évitez les retards de paiement")),
                        ft.ListTile(title=ft.Text("• Diversifiez vos sources de revenus")),
                    ]
                )
            ])

            self.page.clean()
            self.page.add(view)

    except Exception as e:
        self.show_error(f"Erreur: {str(e)}")


def get_score_color(self, score):
    """Couleur selon le score"""
    if score >= 750:
        return ft.colors.GREEN
    elif score >= 650:
        return ft.colors.BLUE
    elif score >= 550:
        return ft.colors.ORANGE
    else:
        return ft.colors.RED


class Transaction(db.Model):
    __tablename__ = 'transactions'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    pret_id = db.Column(db.Integer)
    montant = db.Column(db.Float)
    gateway = db.Column(db.String(20))  # moncash, natcash, etc.
    transaction_id = db.Column(db.String(100))  # ID de la transaction du gateway
    statut = db.Column(db.String(20), default='en_attente')  # en_attente, paye, echoue
    date_creation = db.Column(db.DateTime, default=datetime.utcnow)
    date_confirmation = db.Column(db.DateTime)
    metadata_info = db.Column(db.Text)  # Données supplémentaires au format JSON


# === AJOUTEZ CETTE FONCTION À LA FIN DE models.py ===

def get_statistiques():
    """Récupère les statistiques réelles depuis la base de données"""
    from sqlalchemy import func

    # 1. Clients actifs
    clients_actifs = Client.query.filter_by(statut='actif').count()

    # 2. Total des prêts accordés (statut='approuvé' ou 'accordé' selon votre système)
    total_prets = db.session.query(
        func.coalesce(func.sum(Pret.montant), 0)
    ).filter(Pret.statut.in_(['approuvé', 'accordé', 'actif'])).scalar()

    # 3. Taux de remboursement (calcul basé sur les remboursements)
    total_rembourse = db.session.query(
        func.coalesce(func.sum(Remboursement.montant), 0)
    ).filter(Remboursement.statut == 'payé').scalar()

    total_a_rembourser = db.session.query(
        func.coalesce(func.sum(Pret.montant_total), 1)
    ).filter(Pret.statut.in_(['approuvé', 'accordé', 'actif'])).scalar() or 1

    if total_a_rembourser and total_a_rembourser > 0:
        taux_remboursement = int((total_rembourse / total_a_rembourser) * 100)
        taux_remboursement = min(taux_remboursement, 100)  # Ne pas dépasser 100%
    else:
        taux_remboursement = 0

    # 4. Communautés (basé sur les zones des groupes)
    communautes = db.session.query(
        func.count(func.distinct(Groupe.zone))
    ).filter(Groupe.zone.isnot(None)).scalar() or 0

    return {
        'clients_actifs': clients_actifs,
        'prets_accordes': float(total_prets) if total_prets else 0,
        'taux_remboursement': taux_remboursement,
        'communautes': communautes
    }