import hashlib
import os
import random
from flask import jsonify, session, request, make_response
from dao.feedback import FeedbackDao
from db import db
from models import Feedback
from schema.feedback import FeedbackSchema

class FeedbackProvider:

    dummy_data = [
        {"user_prompt": "What is Python?", "response": "It is a programming language used in web development, AI, data science", "feedback": None},
        {"user_prompt": "What is Python?", "response": "It is a speaking language", "feedback": None},
        {"user_prompt": "Capital of India?", "response": "Capital of India is Mumbai", "feedback": None},
        {"user_prompt": "Capital of India?", "response": "Capital of India is New Delhi", "feedback": None},
    ]

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    IMAGE_FOLDER = os.path.join(BASE_DIR, "images")

    @staticmethod
    def get_feedback():
        feedback_data = FeedbackDao.get_feedback()
        return FeedbackSchema(many=True).dump(feedback_data)

    @staticmethod
    def get_feedback_by_id(user_id):
        feedback_data = FeedbackDao.feedback_by_id(user_id)
        return FeedbackSchema(many=False).dump(feedback_data)

    @staticmethod
    def create_feedback(user_id):
        #
        feedback_text = request.args.get("feedback")
        feedback_bool = feedback_text.lower() == "yes" if feedback_text else None
        if feedback_bool is None:
            random_data = random.choice(FeedbackProvider.dummy_data)
            session["user_prompt"] = random_data["user_prompt"]
            session["response"] = random_data["response"]
            return jsonify({"user_prompt": random_data["user_prompt"], "response": random_data["response"],
                            "message": "Provide Feedback (True/False) before submitting."})

        user_prompt, response = session.pop("user_prompt", None), session.pop("response", None)
        if not user_prompt or not response:
            return make_response(jsonify({"message": "No data found. Please generate a prompt first."}), 400)
        new_feedback = FeedbackDao.insert(user_prompt=user_prompt, response=response, user_id=user_id, feedback=feedback_bool)
        print(new_feedback)
        return FeedbackSchema().dump(new_feedback)

    @staticmethod
    def upload_image_for_bytea(record_id):
        try:
            image_files = [f for f in os.listdir(FeedbackProvider.IMAGE_FOLDER) if f.endswith((".png", ".jpg", ".jpeg"))]
            if not image_files:
                return {"message": "No images available"}, 404

            selected_image = random.choice(image_files)
            image_path = os.path.join(FeedbackProvider.IMAGE_FOLDER, selected_image)

            with open(image_path, "rb") as img_file:
                image_data = img_file.read()
            image_hash = hashlib.sha256(image_data).hexdigest()
            if Feedback.query.filter_by(image_hash=image_hash).first():
                return {"message": "Duplicate image already uploaded"}, 400
            feedback = Feedback.query.get(record_id)
            if not feedback:
                return {"message": "Feedback record not found"}, 404

            feedback.image_large_binary = image_data
            feedback.image_hash = image_hash
            db.session.commit()

            return {"message": f"Image uploaded successfully for record {record_id}"}, 200

        except Exception as e:
            print(f"Error: {e}")
            return {"message": "Internal server error"}, 500
