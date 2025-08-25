import hashlib
import os
import random
from db import db
from models.image import Image
from schema.image import ImageSchema, ImageBinarySchema
from flask import request


class ImageProvider():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    IMAGE_FOLDER = os.path.join(BASE_DIR, "images")
    @staticmethod
    def upload_image_in_bytea(user_id):
        try:
            image_files = [f for f in os.listdir(ImageProvider.IMAGE_FOLDER) if f.endswith((".png", ".jpg", ".jpeg"))]
            selected_image = random.choice(image_files)
            image_path = os.path.join(ImageProvider.IMAGE_FOLDER, selected_image)
            with open(image_path, "rb") as img_file:
                image_data = img_file.read()
            image_hash = hashlib.sha256(image_data).hexdigest()
            existing_image = Image.query.filter_by(image_hash=image_hash).scalar()
            if existing_image:
                return {"message": "Duplicate image already uploaded for another record"}, 400

            image_desc = request.form.get("image_desc", "")

            new_image = Image(
                image_large_binary=image_data,
                image_hash=image_hash,
                user_id=int(user_id),
                image_description=image_desc)
            db.session.add(new_image)
            db.session.flush()

            db.session.commit()

            return ImageBinarySchema().dump(new_image), 200

        except Exception as e:
            print(f"Upload error: {e}")
            return {"message": "Internal server error"}, 500