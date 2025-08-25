from flask_login import UserMixin
from models import db

class User(db.Model, UserMixin):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    user_name = db.Column(db.String(250), nullable=False)
    password = db.Column(db.String(250), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
    user_id = db.Column(db.String, nullable=False)