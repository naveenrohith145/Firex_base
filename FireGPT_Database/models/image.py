
from models import db


class Image(db.Model):
    __tablename__ = 'images'

    id = db.Column(db.Integer, primary_key=False, nullable=False)
    image_id = db.Column(db.String, primary_key=True)
    # image_description=db.Column(db.Text, nullable=False)
    image_name=db.Column(db.String(100), nullable=False)
    image_large_binary = db.Column(db.LargeBinary, nullable=True)
    image_hash = db.Column(db.String(64), nullable=True)
    user_id = db.Column(db.String, nullable=False)
    image_path=db.Column(db.String, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)

    # feedback = db.relationship("Feedback", backref="image")
    # saved_chats=db.relationship("SavedChat", backref="image")
