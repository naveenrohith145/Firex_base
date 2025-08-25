from models import db


class SavedChat(db.Model):
    __tablename__ = 'saved_chat'
    user_id = db.Column(db.String, nullable=False)
    session_id = db.Column(db.String, nullable=False)
    userprompt = db.Column(db.String(4098), nullable=False)
    response = db.Column(db.String(4098), nullable=False)
    image_id = db.Column(db.String, db.ForeignKey("images.image_id"), nullable=True)
    image_hash = db.Column(db.String(64), nullable=True)
    id = db.Column(db.String, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False)