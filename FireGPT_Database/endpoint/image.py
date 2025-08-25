from flask import request, Response, jsonify, session
from flask_login import login_required, current_user
from flask_restx import Namespace, Resource

from dao.image import ImageDao
from db import db
from providers.image import ImageProvider

images=Namespace('images', strict_slashes=False)

@images.route('/binaryimage/')
class ImageUpload(Resource):

    @login_required
    def post(self):
        user_id = current_user.id
        if not user_id:
            return jsonify({"message": "User not authenticated"}), 401
        updated_record = ImageProvider.upload_image_in_bytea(user_id)
        return updated_record

@images.route('/binaryimage/<int:record_id>')
class ImageRetreive(Resource):

    @login_required
    def get(self,record_id):
        data=ImageDao.image_by_id(record_id)
        if data.image_large_binary is None:
            return jsonify({"message": "Image not found"}), 404
        image_data=data.image_large_binary
        return Response(image_data, mimetype="image/jpeg")