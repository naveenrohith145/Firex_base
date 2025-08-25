import base64
from flask import Response, json
from flask_restx import Namespace, Resource
import ast
import os
import glob
from db import db
from models import Feedback
from models.image import Image
from models.saved_chat import SavedChat
from models.user import User  # Import the User model


# Define the output directory for all JSON files
JSON_OUTPUT_DIR = r"C:\Users\z0052pyj\Documents\DemoVNV\data\databaseJson"

feedback_ref = Namespace('feedback', strict_slashes=False)

@feedback_ref.route('/')
class FeedbackMain(Resource):
    def get(self):
        # Use the direct path to feedback.json
        json_path = os.path.join(JSON_OUTPUT_DIR, 'feedback.json')
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            new_records = 0
            for record in data:
                try:
                    id = record.get("id")
                    existing = db.session.query(Feedback).get(id)
                    if existing:
                        print(f"Skipping existing record with id={id}")
                        continue

                    image_id = record.get("image_id")

                    image_hash = record.get("image_hash")
                    # if image_hash:
                    #     duplicate_image = db.session.query(Image).filter_by(image_hash=image_hash).first()
                    #     if duplicate_image:
                    #         print(f"Duplicate image with hash {image_hash} found for record with id={id}. Skipping.")
                    #         continue

                    user_prompt = record.get("user_prompt")
                    raw_response = record.get("response")
                    try:
                        parsed_response = ast.literal_eval(raw_response)
                        response = json.dumps(parsed_response)
                    except Exception:
                        response = raw_response
                    feedback = record.get("feedback")
                    suggestion = record.get("suggestion")
                    timestamp = record.get("timestamp")
                    all_data = Feedback(
                        id=id,
                        user_prompt=user_prompt,
                        response=response,
                        feedback=feedback,
                        suggestion=suggestion,
                        image_id=image_id,
                        image_hash=image_hash,
                        timestamp=timestamp
                    )
                    db.session.add(all_data)
                    new_records += 1
                except Exception as inner_err:
                    print(f"Error processing record with id={record.get('id')}: {inner_err}")
                    continue
            if new_records > 0:
                db.session.commit()
                return {"message": f"{new_records} new record(s) inserted."}, 200
            else:
                return {"message": "No new records to insert."}, 200
        except Exception as e:
            return {"error": f"Unexpected error: {e}"}, 500


image = Namespace('image', strict_slashes=False)

@image.route('/')
class ImageClass(Resource):
    def get(self):
        # Use the direct path to images.json
        json_path = os.path.join(JSON_OUTPUT_DIR, 'images.json')
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            new_records = 0
            for record in data:
                try:
                    image_id = record.get("image_id")
                    existing = db.session.query(Image).get(image_id)
                    if existing:
                        print(f"Skipping existing record with id={image_id}")
                        continue

                    image_hash = record.get("image_hash")
                    if image_hash:
                        duplicate_image = db.session.query(Image).filter_by(image_hash=image_hash).first()
                        if duplicate_image:
                            print(f"Duplicate image with hash {image_hash} found. Skipping.")
                            continue

                    user_id = record.get("user_id")
                    image_name = record.get("image_name")  # Updated to match our field name
                    image_path = record.get("image_path")
                    timestamp = record.get("timestamp")
                    image_large_binary =  record.get("image_large_binary")
                    integer_id = record.get("id", 0)  # Get the integer ID
                    image_data = base64.b64decode(image_large_binary)

                    # print(f"Decoded image data ={image_data}")
                    
                    all_data = Image(
                        image_id=image_id,
                        user_id=user_id,
                        image_name=image_name,
                        image_path=image_path,
                        image_hash=image_hash,
                        image_large_binary=image_data,
                        id=integer_id,  # Include the integer ID
                        timestamp=timestamp
                    )
                    db.session.add(all_data)
                    new_records += 1
                except Exception as inner_err:
                    print(f"Error processing record with id={record.get('id') or record.get('image_id')}: {inner_err}")
                    continue
            if new_records > 0:
                db.session.commit()
                return {"message": f"{new_records} new record(s) inserted."}, 200
            else:
                return {"message": "No new records to insert."}, 200
        except Exception as e:
            return {"error": f"Unexpected error: {e}"}, 500


chat = Namespace('savedchat', strict_slashes=False)

@chat.route('/')
class SaveChat(Resource):
    def get(self):
        # Find all savedchat files in the directory
        savedchat_pattern = os.path.join(JSON_OUTPUT_DIR, "savedchat_*.json")
        savedchat_files = glob.glob(savedchat_pattern)
        
        if not savedchat_files:
            return {"message": "No saved chat files found."}, 200
            
        total_records = 0
        
        # Process each savedchat file
        for chat_file in savedchat_files:
            try:
                with open(chat_file, 'r') as f:
                    data = json.load(f)

                record_count = 0
                for record in data:
                    try:
                        id = record.get("id")
                        existing_data = db.session.query(SavedChat).get(id)
                        if existing_data:
                            print(f"Skipping existing record with id={id}")
                            continue

                        image_id = record.get("image_id")
                        image = db.session.query(Image).get(image_id) if image_id else None

                        image_hash = record.get("image_hash")
                        # if image_hash:
                        #     duplicate_image = db.session.query(Image).filter_by(image_hash=image_hash).first()
                        #     if duplicate_image:
                        #         print(f"Duplicate image with hash {image_hash} found for saved chat with id={id}. Skipping.")
                        #         continue

                        user_prompt = record.get("userprompt")
                        raw_response = record.get("response")
                        try:
                            parsed_response = ast.literal_eval(raw_response)
                            response = json.dumps(parsed_response)
                        except Exception:
                            response = raw_response

                        session_id = record.get("sessionid")
                        user_id = record.get("userid")
                        timestamp = record.get("timestamp")

                        all_data = SavedChat(
                            image_id=image_id,
                            user_id=user_id,
                            session_id=session_id,
                            userprompt=user_prompt,
                            response=response,
                            image_hash=image_hash,
                            id=id,
                            timestamp=timestamp
                        )
                        db.session.add(all_data)
                        record_count += 1

                    except Exception as inner_err:
                        print(f"Error processing record with id={record.get('id')}: {inner_err}")
                        continue
                
                total_records += record_count
                
            except Exception as e:
                print(f"Error processing file {chat_file}: {e}")
                continue

        if total_records > 0:
            db.session.commit()
            return {"message": f"{total_records} new record(s) inserted from {len(savedchat_files)} file(s)."}, 200
        else:
            return {"message": "No new records to insert."}, 200
        
get_image = Namespace('get_image', strict_slashes=False)
 
@get_image.route('/<int:record_id>')
class ImageClass(Resource):
    def get(self, record_id):
        data = Image.query.filter(Image.id == record_id).first()
        if not data or data.image_large_binary is None:
            return {"message": "Image not found"}, 404
        image_data = data.image_large_binary
        return Response(image_data, mimetype="image/jpeg")
    
# Add the new login namespace and UserLogin class
login = Namespace('login', strict_slashes=False)

@login.route('/')
class UserLogin(Resource):
    def get(self):
        user_details_path = os.path.join(JSON_OUTPUT_DIR, "user.json")
        try:
            with open(user_details_path, 'r') as f:
                user_data = json.load(f)
            total_records = 0
            for record in user_data:
                try:
                    id = record.get("id")
                    user_id = record.get("user_id")
                    user_name = record.get("user_name")
                    password = record.get("password")
                    timestamp = record.get("timestamp")
                    all_data = User(
                        id=id,
                        user_id=user_id,
                        user_name=user_name,
                        password=password,
                        timestamp=timestamp
                    )
                    db.session.add(all_data)
                    total_records += 1
                except Exception as inner_err:
                    print(f"Error processing record with user_id={record.get('user_id')}: {inner_err}")
                    continue
            if total_records > 0:
                db.session.commit()
                return {"message": f"{total_records} new record(s) inserted."}, 200
            else:
                return {"message": "No new records to insert."}, 200
        except Exception as e:
            print(f"Error processing user data: {e}")
            return {"error": f"Unexpected error: {e}"}, 500        