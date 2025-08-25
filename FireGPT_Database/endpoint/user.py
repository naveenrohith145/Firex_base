from flask import request, session, jsonify
from flask_login import login_user, logout_user, login_required
from flask_restx import Namespace, Resource

from db import db
from models import Feedback
from models.user import User


user_login = Namespace('user', strict_slashes=False)

@user_login.route('/login')
class Login(Resource):
    def post(self):
        data = request.get_json()
        username = data.get("name")
        email = data.get("email_id")
        user = User.query.filter_by(name=username, email_id=email).first()
        if user:
            login_user(user)
            return jsonify({"message": "Login successful", "user_id": user.id})
        else:
            return jsonify({"message": "User not authorized"})
        return jsonify({"message": "Invalid credentials"}), 401

@user_login.route('/logout')
class Logout(Resource):
    @login_required
    def post(self):

        logout_user()
        return "User logged out successfully", 200

