import gradio as gr
import random
import os
import requests
import uuid
import socket
import psutil
import json
import glob
import time
from pathlib import Path
import mimetypes
from datetime import datetime

# Configure your JSON files folder path - UPDATE THIS!
CHAT_HISTORY_FOLDER = r"data\databaseJson"
USERS_FILE = r"data\frontendUsers\users.json"  # File to track MAC address -> user_id mapping

# Create folder if it doesn't exist
if not os.path.exists(CHAT_HISTORY_FOLDER):
    os.makedirs(CHAT_HISTORY_FOLDER)
    print(f"Created chat history folder: {CHAT_HISTORY_FOLDER}")

def get_ethernet_ipv4():
    # Try ethernet interfaces in order: ethernet, ethernet 2, ethernet 3, ethernet 4, ethernet 5
    ethernet_interfaces = ["ethernet 2", "ethernet 3", "ethernet 4", "ethernet 5"]
    
    for target_interface in ethernet_interfaces:
        for interface, addrs in psutil.net_if_addrs().items():
            if interface.lower() == target_interface:
                for addr in addrs:
                    if addr.family == socket.AF_INET:  # IPv4
                        print(f"Found IP address {addr.address} on interface {interface}")
                        return addr.address
    
    print("No ethernet interfaces found, falling back to localhost")
    return None

def get_mac_address():
    """Get the MAC address of the primary network interface"""
    try:
        # Get MAC address using uuid.getnode()
        mac_num = uuid.getnode()
        mac_hex = ':'.join(['{:02x}'.format((mac_num >> i) & 0xff) for i in range(0, 48, 8)])
        return mac_hex
    except Exception as e:
        print(f"Error getting MAC address: {e}")
        # Fallback: try to get MAC from network interfaces
        try:
            for interface_name, interface_addresses in psutil.net_if_addrs().items():
                for address in interface_addresses:
                    if address.family == psutil.AF_LINK:  # MAC address family
                        if address.address and address.address != '00:00:00:00:00:00':
                            return address.address
        except:
            pass
        return "unknown_mac"

def load_users_data():
    """Load users.json file that maps MAC addresses to user IDs"""
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {}
    except Exception as e:
        print(f"Error loading users data: {e}")
        return {}

def save_users_data(users_data):
    """Save users.json file"""
    try:
        with open(USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(users_data, f, indent=2)
        print(f"Saved users data to {USERS_FILE}")
    except Exception as e:
        print(f"Error saving users data: {e}")

def get_or_create_user_id():
    """Get existing user_id for this machine or create a new one"""
    mac_address = get_mac_address()
    print(f"Machine MAC Address: {mac_address}")
    
    # Load existing users data
    users_data = load_users_data()
    
    # Check if this MAC address already has a user_id
    if mac_address in users_data:
        existing_user_id = users_data[mac_address]["user_id"]
        print(f"Found existing user for this machine: {existing_user_id}")
        return existing_user_id
    else:
        # Create new user_id for this MAC address
        new_user_id = str(uuid.uuid4())
        
        # Add to users data
        users_data[mac_address] = {
            "user_id": new_user_id,
            "created_timestamp": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat()
        }
        
        # Save updated users data
        save_users_data(users_data)
        
        print(f"Created new user for this machine: {new_user_id}")
        return new_user_id

def update_user_last_accessed():
    """Update the last accessed timestamp for current user"""
    try:
        mac_address = get_mac_address()
        users_data = load_users_data()
        
        if mac_address in users_data:
            users_data[mac_address]["last_accessed"] = datetime.now().isoformat()
            save_users_data(users_data)
    except Exception as e:
        print(f"Error updating last accessed: {e}")

# Get user_id based on MAC address
user_id = get_or_create_user_id()

# Track if current chat is new (for button creation)
current_chat_is_new = True
# Track the current active chat_id
current_chat_id = None


# Set up API endpoints
IP_ADDRESS = get_ethernet_ipv4()
BASE_URL = f"http://{IP_ADDRESS}:5000"
TEXT_API_URL = f"{BASE_URL}/chat"
FILE_UPLOAD_API_URL = f"{BASE_URL}/upload_file"
FEEDBACK_API_URL = f"{BASE_URL}/feedback"
NEW_CHAT_API_URL = f"{BASE_URL}/new_chat"
INCOGNITO_CHAT_API_URL = f"{BASE_URL}/incognito_chat"
INCOGNITO_FILE_UPLOAD_API_URL = f"{BASE_URL}/incognito_upload_file"
INCOGNITO_DOC_API_URL = f"{BASE_URL}/incognito_doc"

def send_incognito_chat_request(text):
    try:
        response = requests.post(INCOGNITO_CHAT_API_URL, json={"query": text})
        response.raise_for_status()
        data_resp = response.json()
        return data_resp.get("response", "No response from Incognito API")
    except Exception as e:
        return f"Error: {str(e)}"

def send_incognito_doc_request(text, files):
    file_path = files[0]
    filename = Path(file_path).name
    mime_type, _ = mimetypes.guess_type(filename)
    mime_type = mime_type if mime_type else "application/octet-stream"
    data = {"query": text}
    files_data = {"file": (filename, open(file_path, "rb"), mime_type)}
    try:
        response = requests.post(INCOGNITO_DOC_API_URL, data=data, files=files_data)
        response.raise_for_status()
        data_resp = response.json()
        return data_resp.get("response", "No response from Incognito Doc API")
    except Exception as e:
        return f"Error: {str(e)}"

file_responses = ["Thanks for sharing that file!", "I received your file!",
                  "I see you've shared a file with me.", "I got your file attachment."]

# For tracking disliked messages
current_disliked_message = {"response": "", "is_active": False}

def send_text_request(text):
    """Send text to the API endpoint and return the response"""
    global current_chat_id
    
    # Prepare request data
    request_data = {"query": text, "user_id": user_id}
    
    # If we're continuing an existing chat, include the chat_id
    if current_chat_id and not current_chat_is_new:
        request_data["chat_id"] = current_chat_id
        print(f"Continuing existing chat: {current_chat_id}")
    else:
        print("Starting new chat or first message")
    
    response = requests.post(TEXT_API_URL, json=request_data)
    response.raise_for_status()
    data_resp = response.json()
    
    if isinstance(data_resp, dict):
        # Check for diagram image data
        if "image_data" in data_resp:
            image_format = data_resp.get("image_format", "png")
            image_base64 = data_resp["image_data"]

            # Create HTML with embedded image
            image_html = f'<img src="data:image/{image_format};base64,{image_base64}" alt="Generated Diagram" style="max-width:100%;">'

            # Combine text response with image
            full_response = f"{image_html}<br><br>{data_resp.get('response', '')}"
            return full_response

        return data_resp.get("response", "No response from API")
    return data_resp

def send_file_request(text, files):
    """Send files to the API endpoint and return the response"""
    global current_chat_id
    
    file_path = files[0]
    filename = Path(file_path).name
    mime_type, _ = mimetypes.guess_type(filename)
    mime_type = mime_type if mime_type else "application/octet-stream"

    data = {"query": text, "user_id": user_id}
    
    # If we're continuing an existing chat, include the chat_id
    if current_chat_id and not current_chat_is_new:
        data["chat_id"] = current_chat_id
        print(f"Continuing existing chat with file: {current_chat_id}")
    else:
        print("Starting new chat with file or first message")
    
    files_data = {"file": (filename, open(file_path, "rb"), mime_type)}

    response = requests.post(FILE_UPLOAD_API_URL, data=data, files=files_data)
    response.raise_for_status()
    data_resp = response.json()

    if isinstance(data_resp, dict):
        # Check for diagram image data
        if "image_data" in data_resp:
            image_format = data_resp.get("image_format", "png")
            image_base64 = data_resp["image_data"]

            # Create HTML with embedded image
            image_html = f'<img src="data:image/{image_format};base64,{image_base64}" alt="Generated Diagram" style="max-width:100%;">'

            # Combine text response with image
            full_response = f"{image_html}<br><br>{data_resp.get('response', '')}"
            return full_response

        return data_resp.get("response", "No response from API")
    return data_resp

def inference(message, chat_history, incognito_mode=False):
    if not message:
        return chat_history

    if isinstance(message, dict):
        text = message.get("text", "")
        files = message.get("files", [])

        if files and any(files):
            filename = Path(files[0]).name
            user_content = f"{filename} </br>{text}" if text else filename
        else:
            user_content = text

        chat_history.append({"role": "user", "content": user_content})

        try:
            chat_history.append({"role": "assistant", "content": "Processing..."})
            yield chat_history

            if incognito_mode:
                if files and any(files):
                    response_text = send_incognito_doc_request(text, files)
                elif text:
                    response_text = send_incognito_chat_request(text)
                else:
                    response_text = "I didn't receive any input. Try typing something or uploading a file."
            else:
                if files and any(files):
                    response_text = send_file_request(text, files)
                elif text:
                    response_text = send_text_request(text)
                else:
                    response_text = "I didn't receive any input. Try typing something or uploading a file."

            chat_history[-1]["content"] = response_text
            yield chat_history

        except Exception as e:
            if files and any(files):
                print(f"Error processing file: {str(e)}")
                fallback = f"Error processing file. Supported file formats are: {', '.join(['.pdf', '.docx', '.xls', '.xlsx', '.csv', '.png', '.jpg', '.jpeg', '.txt'])}. Please try again."
            else:
                print(f"Error connecting to API: {str(e)}")
                fallback = f"Unable to process your request."
            chat_history[-1]["content"] = fallback
            yield chat_history

    else:
        chat_history.append({"role": "user", "content": message})
        yield chat_history

        try:
            chat_history.append({"role": "assistant", "content": "Processing..."})
            yield chat_history

            if incognito_mode:
                response_text = send_incognito_chat_request(message)
            else:
                response_text = send_text_request(message)

            chat_history[-1]["content"] = response_text
            yield chat_history

        except Exception as e:
            fallback = f"Error connecting to API: {str(e)}. Unable to process your request."
            chat_history[-1]["content"] = fallback
            yield chat_history

def start_new_chat_with_flag():
    """Start new chat and set the flag"""
    global current_chat_is_new, current_chat_id
    try:
        response = requests.post(NEW_CHAT_API_URL, json={"user_id": user_id})
        response.raise_for_status()
        current_chat_is_new = True
        current_chat_id = None  # Reset current chat ID
        print("New chat started - waiting for first message to create button")
        return []
    except Exception as e:
        return [{"role": "assistant", "content": f"Error starting new chat: {str(e)}"}]

def handle_feedback(chat_history, liked: gr.LikeData):
    """Send feedback (like/dislike) to the API"""
    if not chat_history or len(chat_history) == 0:
        return gr.update(visible=False)

    # Get the last assistant message
    for msg in reversed(chat_history):
        if msg.get("role") == "assistant":
            response_text = msg.get("content", "")
            break
    else:
        return gr.update(visible=False)

    # Check if this is an image response (simple heuristic)
    is_image_response = "<img src=" in response_text

    if liked.liked:  # Handle likes
        try:
            feedback_data = {
                "feedback": "like",
                "response_text": response_text,
                "user_id": user_id,
                "is_image_response": is_image_response
            }
            response = requests.post(FEEDBACK_API_URL, json=feedback_data)
            response.raise_for_status()
            return gr.update(visible=False)
        except Exception as e:
            print(f"Error sending feedback: {e}")
            return gr.update(visible=False)
    else:  # Handle dislikes
        current_disliked_message["response"] = response_text
        current_disliked_message["is_active"] = True
        current_disliked_message["is_image_response"] = is_image_response
        return gr.update(visible=True)

def submit_suggestion(suggestion):
    """Submit the suggestion for a disliked message"""
    if not current_disliked_message["is_active"]:
        return gr.update(visible=False)

    try:
        response = requests.post(FEEDBACK_API_URL, json={
            "feedback": "dislike",
            "response_text": current_disliked_message["response"],
            "suggestion": suggestion,
            "user_id": user_id,
            "is_image_response": current_disliked_message.get("is_image_response", False)
        })
        response.raise_for_status()

        # Reset the disliked message tracking
        current_disliked_message["is_active"] = False
        current_disliked_message["response"] = ""

        # Hide suggestion box and clear the input
        return gr.update(visible=False, value="")
    except Exception as e:
        print(f"Error sending feedback: {e}")
        return gr.update(visible=False)

# DYNAMIC CHAT HISTORY FUNCTIONS
def get_user_chat_data():
    """Get all chat data for current user from JSON file"""
    try:
        json_filename = f"savedchat_{user_id}.json"
        # Use the configured folder path
        full_json_path = os.path.join(CHAT_HISTORY_FOLDER, json_filename)
        
        if not os.path.exists(full_json_path):
            print(f"No chat history file found at: {full_json_path}")
            return {"userid": user_id, "chats": []}
        
        with open(full_json_path, 'r', encoding='utf-8') as f:
            user_data = json.load(f)
        
        print(f"Loaded chat data for user {user_id}: {len(user_data.get('chats', []))} chats found")
        return user_data
    except Exception as e:
        print(f"Error reading user chat data: {e}")
        return {"userid": user_id, "chats": []}

def get_available_chats():
    """Get list of available chats for current user"""
    user_data = get_user_chat_data()
    chats = user_data.get("chats", [])
    
    # Sort by created_timestamp (newest first)
    chats.sort(key=lambda x: x.get("created_timestamp", ""), reverse=True)
    return chats

def load_specific_chat(chat_id):
    """Load specific chat by chat_id and convert to Gradio format"""
    try:
        user_data = get_user_chat_data()
        
        # Find the specific chat
        target_chat = None
        for chat in user_data.get("chats", []):
            if chat.get("chat_id") == chat_id:
                target_chat = chat
                break
        
        if not target_chat:
            return [{"role": "assistant", "content": f"Chat with ID '{chat_id}' not found"}]
        
        # Convert messages to Gradio message format
        gradio_messages = []
        for message in target_chat.get("messages", []):
            # Add user message
            gradio_messages.append({
                "role": "user", 
                "content": message.get("userprompt", "")
            })
            # Add assistant message  
            gradio_messages.append({
                "role": "assistant", 
                "content": message.get("response", "")
            })
        
        return gradio_messages
        
    except Exception as e:
        return [{"role": "assistant", "content": f"Error loading chat: {str(e)}"}]

def update_chat_buttons():
    """Update chat history buttons based on available chats"""
    chats = get_available_chats()
    
    button_updates = []
    for i in range(10):  # Assuming max 10 chat buttons
        if i < len(chats):
            chat = chats[i]
            chat_prompt = chat.get("chat_prompt", "Untitled Chat")
            # Truncate long prompts
            truncated_prompt = chat_prompt if len(chat_prompt) <= 25 else chat_prompt[:25] + "..."
            button_updates.append(gr.update(value=truncated_prompt, visible=True))
        else:
            button_updates.append(gr.update(visible=False))
    
    return button_updates

def check_and_update_buttons_on_first_message(chat_history):
    """Check if we need to update buttons after a new message"""
    global current_chat_is_new, current_chat_id
    
    # Only update buttons if this was a truly new chat (not continuing existing)
    if not current_chat_is_new and len(chat_history) == 2 and current_chat_id is None:  
        print("First message in new chat detected - updating sidebar buttons")
        time.sleep(0.5)  # Small delay to ensure backend has saved the chat
        return update_chat_buttons()
    else:
        return [gr.update() for _ in range(10)]  # No updates needed

def create_chat_loader(chat_index):
    """Create a chat loader function for a specific chat index"""
    def load_chat():
        global current_chat_is_new, current_chat_id
        chats = get_available_chats()
        if chat_index < len(chats):
            chat_id = chats[chat_index]["chat_id"]
            current_chat_is_new = False  # Mark as not new when loading existing chat
            current_chat_id = chat_id    # Set the active chat ID
            print(f"Loaded existing chat: {chat_id}")
            return load_specific_chat(chat_id)
        return [{"role": "assistant", "content": "Chat not found"}]
    return load_chat

# Create individual chat loader functions
load_chat_0 = create_chat_loader(0)
load_chat_1 = create_chat_loader(1)
load_chat_2 = create_chat_loader(2)
load_chat_3 = create_chat_loader(3)
load_chat_4 = create_chat_loader(4)
load_chat_5 = create_chat_loader(5)
load_chat_6 = create_chat_loader(6)
load_chat_7 = create_chat_loader(7)
load_chat_8 = create_chat_loader(8)
load_chat_9 = create_chat_loader(9)

# Create the Gradio interface with Blocks and Sidebar
with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal"),title="FireX",fill_height=True) as demo:
    incognito_state = gr.State(False)
    # Custom CSS to left-align button text and set logout button color
    demo.head += """
    <style>
    body{
        background-color: #36454F !important;
        font-family: Open Sans !important;

    }
    
    .gradio-container{
        font-family: Open Sans !important;
    }
    
    .left-align-button button {
        text-align: left !important;
        justify-content: left !important;
        bakground: none !important;
    }
    
    .sm.svelte-1ixn6qd{
        background: none !important;
    }
    
    .left-align-button button:hover {
        background-color: teal !important;
    }
    .suggestion-box {
        background-color: transparent !important;
        border: none;
        padding: 10px;
        margin-top: 10px;
        color: white;
        position: relative;
    }
    .incognito-on {
        background-color: orange !important;
        color: white !important;
        border-color: orange !important;
    }
    p {
        font-size: 18px !important;
    }
    ul,li,ol{
        font-size: 18px !important;
    }
    button {
        font-size: 16px !important;
    }
    .gradio-container button#clear-button {
    width: 300px !important;
    height: 40px !important;
    font-size: 16px !important; /* Example of other style changes */
    display: block !important;
    margin-left: auto !important;
    margin-right: auto !important;
    background-color: transparent !important;
    border: 1px solid #36454F !important;
    border-radius: 22px !important;
    }
    
    .gradio-container button#clear-button:hover {
        background-color: #E35335 !important;
    }
    
    .transparent-chatbot{
        background: transparent !important;
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
        overflow-y: hidden !important;
    }
    
    .bubble-wrap.svelte-gjtrl6.svelte-gjtrl6{
        background: transparent !important;
        
    }
        
    .top-panel.svelte-9lsba8{
        display: none !important;
    }
    
    # .block.svelte-11xb1hd.flex.auto-margin{
    #     background-color: transparent !important;
    # }

    .bot.svelte-yaaj3.svelte-yaaj3{
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    .bubble.svelte-j7nkv7 .icon-button-wrapper {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    .svelte-vzs2gq.padded{
        background-color: transparent !important;
    }
    .icon-button-wrapper.svelte-9lsba8 a.download-link:not(:last-child):not(.no-border *):after, .icon-button-wrapper.svelte-9lsba8 button:not(:last-child):not(.no-border *):after {
        background-color: transparent !important;
    }
    .wrap.default.full.svelte-ls20lj.generating{
        border-radius: 22px !important;
    }
    .form.svelte-633qhp{
        border-radius: 22px !important;
    }
    .input-textbox{
        background-color: transparent !important;
    }
    footer{
        position: fixed !important;
        right: 0 !important;
        top: 0 !important;
        margin-right: 1% !important;
        margin-top: 1% !important;
        }
    </style>
    """
    # Create the sidebar
    with gr.Sidebar(open=False):
        gr.Markdown('<span style="font-size:25px; font-family: Arial; color:#13acac; font-weight:bold;">SIEMENS</span>')
        gr.Markdown("Welcome to FireX Technical Agent!")

        # Add New Chat button (positioned above Prompt Library)
        new_chat_btn = gr.Button("✚ New Chat", variant="primary", size="md")

        # Add Incognito toggle button below New Chat
        incognito_toggle_btn = gr.Button("Incognito OFF", variant="secondary", size="md")

        gr.Markdown(" ")
        # gr.Markdown("### Prompt Library")

        # # Pinned chats as interactive buttons
        # with gr.Column(elem_classes="left-align-button"):
        #     def truncate_text(text, max_length=30):
        #         return text if len(text) <= max_length else text[:max_length] + "..."

        #     pinned_chat1 = gr.Button(
        #         truncate_text("Detailed circuit diagram for the DLC"),
        #         variant="secondary",
        #         size="sm",
        #         scale=1
        #     )
        #     pinned_chat2 = gr.Button(
        #         truncate_text("From Zues manual tell me the configuration of DLC"),
        #         variant="secondary",
        #         size="sm",
        #         scale=1
        #     )
        
        gr.Markdown('<span style="color:#818589">🕓Chat History</span>')    
        
        # Dynamic chat history buttons (initially hidden)
        with gr.Column(elem_classes="left-align-button"):
            chat_btn_0 = gr.Button("", variant="secondary", size="sm", scale=1, visible=False)
            chat_btn_1 = gr.Button("", variant="secondary", size="sm", scale=1, visible=False)
            chat_btn_2 = gr.Button("", variant="secondary", size="sm", scale=1, visible=False)
            chat_btn_3 = gr.Button("", variant="secondary", size="sm", scale=1, visible=False)
            chat_btn_4 = gr.Button("", variant="secondary", size="sm", scale=1, visible=False)
            chat_btn_5 = gr.Button("", variant="secondary", size="sm", scale=1, visible=False)
            chat_btn_6 = gr.Button("", variant="secondary", size="sm", scale=1, visible=False)
            chat_btn_7 = gr.Button("", variant="secondary", size="sm", scale=1, visible=False)
            chat_btn_8 = gr.Button("", variant="secondary", size="sm", scale=1, visible=False)
            chat_btn_9 = gr.Button("", variant="secondary", size="sm", scale=1, visible=False)

    # Main chat interface
    gr.Markdown('<span style="font-size:28px; font-weight:bold;">🔥FireX</span>')
    # gr.Markdown("Welcome to FireX! Ready to assist you.")

    # Initialize chat state - this is crucial for proper functioning
    chat_history = gr.State([])

    # Chatbot with proper message format and the like functionality
    # Declare a temporary variable to hold
    chatbot_component = gr.Chatbot(
        value=[],
        type="messages",
        placeholder="<h1>What can I help with?",
        show_copy_button=True,
        show_label=False,
        autoscroll=True,
        elem_classes="transparent-chatbot",
        container=False,
        scale=2,
    )
        
    msg = gr.MultimodalTextbox(placeholder="Enter query or upload a file...", show_label=False, elem_id="input-textbox")
    gr.Examples(
        examples=["Key features of XDLC", "Generate diagram for ZIC-4a wiring with its devices", "How to resolve ground fault issue"],
        inputs=msg,
        label="Suggestions",
    )

    # Add suggestion box for feedback
    with gr.Column(visible=False, elem_classes="suggestion-box") as suggestion_container:
        gr.Markdown("### Please explain what was wrong with the response")
        suggestion_box = gr.Textbox(
            placeholder="Your feedback helps us improve",
            lines=3,
            show_label=False,
        )
        submit_btn = gr.Button("Submit", variant="primary")

    # Function to respond to messages - connecting with inference function
    def respond(message, chat_history, incognito_mode):
        global current_chat_is_new, current_chat_id
        
        # Check if this is the first message in a new chat
        is_first_message_in_new_chat = current_chat_is_new and len(chat_history) == 0
        
        for response in inference(message, chat_history, incognito_mode):
            yield "", response
        
        # Mark that this chat is no longer new after first message
        if is_first_message_in_new_chat:
            current_chat_is_new = False

    # Function to clear chat
    def clear_chat_with_flag():
        global current_chat_is_new, current_chat_id
        current_chat_is_new = True
        current_chat_id = None  # Reset current chat ID
        print("Chat cleared - next message will create new button")
        return "", []

    # Connect the UI components - Enhanced to update buttons on first message
    msg.submit(
        respond,
        [msg, chat_history, incognito_state],
        [msg, chatbot_component],
    ).then(
        lambda h: h,
        [chatbot_component],
        [chat_history],
    ).then(
        check_and_update_buttons_on_first_message,
        [chat_history],
        [chat_btn_0, chat_btn_1, chat_btn_2, chat_btn_3, chat_btn_4, 
         chat_btn_5, chat_btn_6, chat_btn_7, chat_btn_8, chat_btn_9]
    )

    # Updated new chat button - no immediate button update
    new_chat_btn.click(
        lambda: (start_new_chat_with_flag(), ""),
        None,
        [chat_history, msg]
    ).then(
        lambda h: h,
        [chat_history],
        [chatbot_component]
    ).then(
        lambda: gr.update(visible=False),
        None,
        [suggestion_container]
    )

    # Connect feedback functionality
    chatbot_component.like(
        handle_feedback,
        [chat_history],
        [suggestion_container]
    )

    # Connect suggestion submission
    submit_btn.click(
        submit_suggestion,
        [suggestion_box],
        [suggestion_container]
    ).then( # Add this .then to clear the suggestion_box
        lambda: gr.update(value=""),
        None,
        [suggestion_box]
    )

    
    # Define functions that return specific messages without taking inputs
    def feedback_message(): return "Detailed circuit diagram for the DLC"
    def feedback_message2(): return "From Zues manual tell me the configuration of DLC"

    # # Connect the sidebar buttons
    # pinned_chat1.click(feedback_message, None, [msg])
    # pinned_chat2.click(feedback_message2, None, [msg])

    # Connect dynamic chat history buttons
    chat_btn_0.click(load_chat_0, None, [chat_history]).then(lambda h: h, [chat_history], [chatbot_component])
    chat_btn_1.click(load_chat_1, None, [chat_history]).then(lambda h: h, [chat_history], [chatbot_component])
    chat_btn_2.click(load_chat_2, None, [chat_history]).then(lambda h: h, [chat_history], [chatbot_component])
    chat_btn_3.click(load_chat_3, None, [chat_history]).then(lambda h: h, [chat_history], [chatbot_component])
    chat_btn_4.click(load_chat_4, None, [chat_history]).then(lambda h: h, [chat_history], [chatbot_component])
    chat_btn_5.click(load_chat_5, None, [chat_history]).then(lambda h: h, [chat_history], [chatbot_component])
    chat_btn_6.click(load_chat_6, None, [chat_history]).then(lambda h: h, [chat_history], [chatbot_component])
    chat_btn_7.click(load_chat_7, None, [chat_history]).then(lambda h: h, [chat_history], [chatbot_component])
    chat_btn_8.click(load_chat_8, None, [chat_history]).then(lambda h: h, [chat_history], [chatbot_component])
    chat_btn_9.click(load_chat_9, None, [chat_history]).then(lambda h: h, [chat_history], [chatbot_component])

    def toggle_incognito(current_state, chat_history):
        new_state = not current_state
        btn_text = "Incognito ON" if new_state else "Incognito OFF"
        btn_class = "incognito-on" if new_state else ""
        cleared_history = []
        return new_state, gr.update(value=btn_text, elem_classes=btn_class), cleared_history, cleared_history

    incognito_toggle_btn.click(
        toggle_incognito,
        [incognito_state, chat_history],
        [incognito_state, incognito_toggle_btn, chatbot_component, chat_history]
    )

    # Initialize chat buttons when app starts
    def initialize_chat_buttons():
        # print(f"Initializing chat buttons for user: {user_id}")
        # print(f"Looking for JSON file: {os.path.join(CHAT_HISTORY_FOLDER, f'savedchat_{user_id}.json')}")
        updates = update_chat_buttons()
        visible_count = len([u for u in updates if u.get('visible', False)])
        # print(f"Found {visible_count} visible chat buttons")
        return updates

    # Load existing chats on startup
    demo.load(
        initialize_chat_buttons,
        None,
        [chat_btn_0, chat_btn_1, chat_btn_2, chat_btn_3, chat_btn_4, 
         chat_btn_5, chat_btn_6, chat_btn_7, chat_btn_8, chat_btn_9]
    )

# Update last accessed timestamp when app starts
update_user_last_accessed()

def launch():
    demo.launch(show_api=False,inbrowser=True,server_port=5005)

# Launch the app
if __name__ == "__main__":
    # demo.launch(show_api=False,inbrowser=True,auth_message="Please enter credentials to access FireX.",auth=("user", "123"),server_port=5005)
    launch()