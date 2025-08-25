from models import db


class Feedback(db.Model):
    __tablename__ = 'feedback'

    id = db.Column(db.Integer, primary_key=True)
    user_prompt = db.Column(db.String(4098), nullable=False)
    response = db.Column(db.String(4098), nullable=False)
    feedback = db.Column(db.String, nullable=True)
    suggestion = db.Column(db.String(300), nullable=True)
    image_id = db.Column(db.String, db.ForeignKey("images.image_id"), nullable=True)
    image_hash = db.Column(db.String(64), nullable=True)
    # user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)

