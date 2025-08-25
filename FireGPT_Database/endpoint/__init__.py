from flask import Blueprint
from flask_restx import Api

from endpoint.save_db_calls import feedback_ref, chat
from endpoint.user import user_login
from endpoint.image import images
from endpoint.save_db_calls import image
from endpoint.feedback_dummy import feedback
from endpoint.save_db_calls import get_image
from endpoint.save_db_calls import login


API_BLUEPRINT = Blueprint('api', __name__, url_prefix='/')

API = Api(
    API_BLUEPRINT,
    default="Feedback POC",
    title='Demo Project',
    )

API.add_namespace(feedback_ref)
API.add_namespace(user_login)
API.add_namespace(images)
API.add_namespace(feedback)
API.add_namespace(image)
API.add_namespace(chat)
API.add_namespace(get_image)
API.add_namespace(login)