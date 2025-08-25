from flask import Flask
from flask_migrate import Migrate
from flask_session import Session
from flask_login import LoginManager, UserMixin, LoginManager
from config import AppConfig
from db import db
from endpoint import API_BLUEPRINT
from models import *

from models import User

app=Flask(__name__)
app.config.from_object(AppConfig)
login_manager=LoginManager()
login_manager.init_app(app)
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = True
app.config["SECRET_KEY"] = "flaskdemoapp"
app.config["SESSION_USE_SIGNER"] = True
app.config["SESSION_FILE_THRESHOLD"] = 100
app.config["SESSION_FILE_DIR"] = "/tmp/flask_session"
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(user_id) if user_id else None

Session(app)

db.init_app(app)
migrate=Migrate(app,db)
app.register_blueprint(API_BLUEPRINT)

@app.route('/')
def home():
    return "Welcome"

if __name__ == "__main__":
    app.run(debug=False)