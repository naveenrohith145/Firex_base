from flask import request, Response, jsonify, session
from flask_login import login_required, current_user
from flask_restx import Namespace, Resource

from dao.feedback import FeedbackDao
from db import db
from endpoint import feedback_ref
from providers.feedback import FeedbackProvider

feedback=Namespace('feedback', strict_slashes=False)

@feedback.route('/')
class FeedbackMain(Resource):
    # @login_required
    def get(self):
        all_feedback=FeedbackProvider.get_feedback()
        return jsonify(all_feedback)

    @login_required
    def post(self):
        user_id=current_user.id
        if not user_id:
            return jsonify({"message": "User not authenticated"}), 401
        feedback_data = FeedbackProvider.create_feedback(user_id)
        return (feedback_data)


@feedback.route('/binaryimage/<int:record_id>')
class ImageUpload(Resource):
    # @login_required
    def patch(self, record_id):
        updated_record = FeedbackProvider.upload_image_for_bytea(record_id)
        return updated_record


    # @login_required
    def get(self,record_id):
        data=FeedbackDao.feedback_by_id(record_id)
        if data.image_large_binary is None:
            return jsonify({"meesage": "Image not found"}), 404
        image_data=data.image_large_binary
        return Response(image_data, mimetype="image/jpeg")