import base64
import os
import json
import requests
import hashlib
import uuid
from datetime import datetime
import threading

# This file contains all database functionality independent from vector database operations

# Initialize counters
feedback_counter = 0
image_counter = 0

def initialize_counters(feedback_json, image_json):
    """Initialize counters based on existing data"""
    global feedback_counter, image_counter
    
    # Set feedback counter based on existing data
    try:
        if os.path.exists(feedback_json):
            with open(feedback_json, 'r') as f:
                data = json.load(f)
                if data:
                    highest_id = max(item.get('id', 0) for item in data)
                    feedback_counter = highest_id
    except Exception as e:
        print(f"Error initializing feedback counter: {e}")
        feedback_counter = 0
    
    # Set image counter based on existing data
    try:
        if os.path.exists(image_json):
            with open(image_json, 'r') as f:
                data = json.load(f)
                if data:
                    highest_id = max(item.get('id', 0) for item in data)
                    image_counter = highest_id
    except Exception as e:
        print(f"Error initializing image counter: {e}")
        image_counter = 0
    
    print(f"Initialized counters: feedback={feedback_counter}, image={image_counter}")


def sync_feedback_to_database(api_base_url):
    """Call the feedback API endpoint to sync JSON data to database"""
    try:
        response = requests.get(f"{api_base_url}/feedback/")
        if response.status_code == 200:
            print(f"✅ Feedback data synced to database: {response.json().get('message')}")
            return True
        else:
            print(f"❌ Failed to sync feedback data: {response.json().get('error')}")
            return False
    except Exception as e:
        print(f"❌ Error syncing feedback data: {e}")
        return False


def sync_image_to_database(api_base_url):
    """Call the image API endpoint to sync JSON data to database"""
    try:
        response = requests.get(f"{api_base_url}/image/")
        if response.status_code == 200:
            print(f"✅ Image data synced to database: {response.json().get('message')}")
            return True
        else:
            print(f"❌ Failed to sync image data: {response.json().get('error')}")
            return False
    except Exception as e:
        print(f"❌ Error syncing image data: {e}")
        return False


def sync_chat_to_database(api_base_url):
    """Call the savedchat API endpoint to sync JSON data to database"""
    try:
        response = requests.get(f"{api_base_url}/savedchat/")
        if response.status_code == 200:
            print(f"✅ Chat data synced to database: {response.json().get('message')}")
            return True
        else:
            print(f"❌ Failed to sync chat data: {response.json().get('error')}")
            return False
    except Exception as e:
        print(f"❌ Error syncing chat data: {e}")
        return False


def get_session_id_for_chat(user_id, chat_id, test_output_dir):
    """Get session_id for a specific chat_id"""
    try:
        output_file = os.path.join(test_output_dir, f"savedchat_{user_id}.json")
        
        if not os.path.exists(output_file):
            return None
            
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        # Handle both old and new format
        if isinstance(data, list):
            # Old format - return None as we can't identify specific chats
            return None
        
        # New format - find the chat by chat_id
        for chat in data.get("chats", []):
            if chat.get("chat_id") == chat_id:
                return chat.get("sessionid")
        
        return None
    except Exception as e:
        print(f"Error getting session_id for chat {chat_id}: {e}")
        return None


def load_specific_chat_history(user_id, chat_id, test_output_dir):
    """Load specific chat history for memory initialization - NEW FUNCTION"""
    try:
        output_file = os.path.join(test_output_dir, f"savedchat_{user_id}.json") 
        
        if not os.path.exists(output_file):
            return None
            
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        # Find the specific chat by chat_id
        if "chats" in data:
            for chat in data["chats"]:
                if chat.get("chat_id") == chat_id:
                    return chat
        
        return None
    except Exception as e:
        print(f"Error loading specific chat history: {e}")
        return None


def save_chat_to_db(user_id, user_prompt, response, conversation_histories, 
                   user_file_contexts, test_output_dir, api_base_url, 
                   image_id=None, image_hash=None, chat_id=None):
    """
    Saves chat data to user-specific JSON file with the nested structure.
    Now supports continuing existing chats via chat_id parameter.
    """
    try:
        # Handle session ID generation 
        if user_id not in conversation_histories:
            conversation_histories[user_id] = {'session_id': str(uuid.uuid4()), 'history': []}
        elif not isinstance(conversation_histories[user_id], dict) or 'session_id' not in conversation_histories[user_id]:
            # Convert list to dict if needed
            if isinstance(conversation_histories[user_id], list):
                history = conversation_histories[user_id]
                conversation_histories[user_id] = {'session_id': str(uuid.uuid4()), 'history': history}
            else:
                conversation_histories[user_id] = {'session_id': str(uuid.uuid4()), 'history': []}
        
        session_id = conversation_histories[user_id]['session_id']
        
        # NEW LOGIC: If chat_id is provided, get its session_id
        if chat_id:
            existing_session_id = get_session_id_for_chat(user_id, chat_id, test_output_dir)
            if existing_session_id:
                session_id = existing_session_id
                conversation_histories[user_id]['session_id'] = session_id
                print(f"Continuing existing chat {chat_id} with session_id {session_id}")
            else:
                print(f"Warning: Could not find session_id for chat {chat_id}, using current session")
        
        # Double-check for image context if missing
        if not image_id and not image_hash and user_id in user_file_contexts:
            context = user_file_contexts[user_id]
            if context.get("context_type") == "image":
                image_id = context.get("image_id")
                image_hash = context.get("image_hash")
                print(f"Retrieved missing image context: id={image_id}, hash={image_hash}")
        
        # Debug info
        if image_id or image_hash:
            print(f"Creating payload with image data: id={image_id}, hash={image_hash}")
        
        # Create message payload
        message_payload = {
            "id": str(uuid.uuid4()),
            "userprompt": user_prompt,
            "response": response,
            "image_id": image_id,
            "image_hash": image_hash,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Define output file
        output_file = os.path.join(test_output_dir, f"savedchat_{user_id}.json")
        
        # Load existing data or create new structure
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    data = json.load(f)
                
                # Convert old format to new format if needed
                if isinstance(data, list):  # Old format was a list of messages
                    # Create new structure with existing messages in first chat
                    new_data = {
                        "userid": user_id,
                        "chats": [
                            {
                                "chat_id": "chat_001",
                                "chat_prompt": data[0]["userprompt"] if data else user_prompt,
                                "sessionid": session_id,
                                "created_timestamp": data[0]["timestamp"] if data else datetime.now().isoformat(),
                                "messages": data.copy()  # Copy all old messages to first chat
                            }
                        ]
                    }
                    data = new_data
            except Exception as e:
                print(f"Error loading existing chat data: {e}")
                # Create new structure if file exists but is invalid
                data = {
                    "userid": user_id,
                    "chats": []
                }
        else:
            # Create new structure if file doesn't exist
            data = {
                "userid": user_id,
                "chats": []
            }
        
        # NEW LOGIC: Handle chat_id-based message addition
        chat_found = False
        
        if chat_id:
            # Look for specific chat by chat_id
            for chat in data["chats"]:
                if chat["chat_id"] == chat_id:
                    # Add message to existing chat
                    chat["messages"].append(message_payload)
                    print(f"Added message to existing chat {chat_id}")
                    chat_found = True
                    break
                    
            if not chat_found:
                print(f"Warning: Could not find chat {chat_id}, creating new chat")
        else:
            # Original logic: Find chat by session_id
            for chat in data["chats"]:
                if chat["sessionid"] == session_id:
                    # Add message to existing chat
                    chat["messages"].append(message_payload)
                    chat_found = True
                    break
                
        if not chat_found:
            # Generate new chat_id (increment from highest existing)
            if data["chats"]:
                # Extract numbers from chat_ids like "chat_001"
                chat_numbers = []
                for chat in data["chats"]:
                    try:
                        num = int(chat["chat_id"].split('_')[1])
                        chat_numbers.append(num)
                    except:
                        pass
                        
                if chat_numbers:
                    next_chat_number = max(chat_numbers) + 1
                    new_chat_id = f"chat_{next_chat_number:03d}"
                else:
                    new_chat_id = "chat_001"
            else:
                new_chat_id = "chat_001"
                
            # Create new chat entry with current message
            new_chat = {
                "chat_id": new_chat_id,
                "chat_prompt": user_prompt,  # First prompt becomes chat_prompt
                "sessionid": session_id,
                "created_timestamp": datetime.now().isoformat(),
                "messages": [message_payload]
            }
            data["chats"].append(new_chat)
            print(f"Created new chat {new_chat_id}")
            
        # Save updated data
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"✅ Chat data saved to {output_file}" + 
              (f" with image_id={image_id}" if image_id else "") +
              (f" (chat_id: {chat_id})" if chat_id else ""))
        
        # Sync to database in background
        threading.Thread(target=sync_chat_to_database, args=(api_base_url,)).start()
        
        return True
    except Exception as e:
        print(f"❌ Error saving chat data: {e}")
        return False


def save_feedback_to_db(user_prompt, response, feedback, feedback_json, api_base_url,
                       suggestion=None, image_id=None, image_hash=None):
    """
    Saves feedback data to a local JSON file with all required fields
    """
    global feedback_counter
    feedback_counter += 1
    
    try:
        # Create payload with all required fields
        payload = {
            "id": feedback_counter,
            "user_prompt": user_prompt,
            "response": response,  # Use the original response as-is
            "feedback": feedback,
            "suggestion": suggestion if suggestion else None,
            "image_id": image_id if image_id else None,
            "image_hash": image_hash if image_hash else None,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Load existing data
        try:
            with open(feedback_json, 'r') as f:
                data = json.load(f)
        except:
            data = []
            
        # Append new entry
        data.append(payload)
        
        # Save updated data
        with open(feedback_json, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"✅ Feedback data saved to {feedback_json}")
        
        # Sync to database in background
        threading.Thread(target=sync_feedback_to_database, args=(api_base_url,)).start()
        
        return True
    except Exception as e:
        print(f"❌ Error saving feedback data: {e}")
        return False


def save_image_to_db(image_path, user_id, image_json, api_base_url):
    """
    Saves image data to a local JSON file with SHA-256 hashing and integer ID
    """
    global image_counter
    image_counter += 1
    
    try:
        # Generate image ID and hash
        image_id = str(uuid.uuid4())
         
        # Calculate image hash using SHA-256
        with open(image_path, 'rb') as f:
            image_content = f.read()
            image_econtent = base64.b64encode(image_content).decode('utf-8')
            image_hash = hashlib.sha256(image_content).hexdigest()
        
        # Create payload with all required fields including integer id
        payload = {
            "id": image_counter,  # Integer ID field
            "image_id": image_id,
            "image_hash": image_hash,
            "image_large_binary": image_econtent,
            "image_name": os.path.basename(image_path),
            "image_path": image_path,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Load existing data
        try:
            with open(image_json, 'r') as f:
                data = json.load(f)
                
                # If data exists, find the highest ID to avoid duplicates
                if data:
                    highest_id = max(item.get('id', 0) for item in data)
                    if highest_id >= image_counter:
                        image_counter = highest_id + 1
                        payload['id'] = image_counter
        except:
            data = []
            
        # Append new entry
        data.append(payload)
        
        # Save updated data
        with open(image_json, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"✅ Image data saved to {image_json} with ID {image_counter}")
        
        # Sync to database in background
        threading.Thread(target=sync_image_to_database, args=(api_base_url,)).start()
        
        return image_id, image_hash
    except Exception as e:
        print(f"❌ Error saving image data: {e}")
        return None, None