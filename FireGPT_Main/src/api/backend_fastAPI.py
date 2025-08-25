import os
import shutil
import tempfile
import uuid
import time
import hashlib
import json
import base64
import pandas as pd
import socket
import psutil
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pydantic import BaseModel
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.document_loaders import (
    PyPDFLoader, 
    CSVLoader,
    TextLoader,
    Docx2txtLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.retrievers import EnsembleRetriever
from functools import lru_cache
from datetime import datetime
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

from src.utils.agents.agent import GraphvizTool, MermaidTool
from src.database import databaseFunctionality as dbf
# Import the modules for manual test case functionality
from src.utils.tools.context import contextTool as contextTool
from src.utils.tools.manualTest import manualTestCase as manualTestCase

# --- Load config and initialize global variables ---
# Load configuration from config\config.json
with open(r"src\config\config.json", "r") as config_file:
    config = json.load(config_file)

# Function to get IPv4 address
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
    
server_ip = config["server_ip"]
current_directory = config["current_directory"]
vector_db_path = config["vector_db_path"]
index_name = config["index_name"]
version = config["version"]
TEST_OUTPUT_DIR = config["test_output_dir"]
FEEDBACK_JSON = os.path.join(TEST_OUTPUT_DIR, config["feedback_json"])
IMAGE_JSON = os.path.join(TEST_OUTPUT_DIR, config["image_json"])
API_BASE_URL = config["api_base_url"]
DEFAULT_CHUNK_SIZE = config["default_chunk_size"]
DEFAULT_CHUNK_OVERLAP = config["default_chunk_overlap"]
DEFAULT_RETRIEVER_K = config["default_retriever_k"]
DEFAULT_SCORE_THRESHOLD = config["default_score_threshold"]
DEFAULT_MMR_FETCH_K = config["default_mmr_fetch_k"]
DEFAULT_MMR_LAMBDA_MULT = config["default_mmr_lambda_mult"]
system_template1 = config["system_template1"]
DEFAULT_IMAGE_PROMPT = config["default_image_prompt"]
DIAGRAM_KEYWORDS = config["diagram_keywords"]
FORMAT_KEYWORDS = config["format_keywords"]
MAX_HISTORY_LENGTH = config["max_history_length"]
MAX_MEMORY_MESSAGES = 20  # Keep last 20 messages for context
CODE_KEYWORDS = config['code_keywords']
embedding_model = config["models"]["embedding_model"]
llm_model = config["models"]["llm_model"]
image_model_name = config["models"]["image_model"]
code_model_name = config["models"]["code_model"]
base_url = config["base_url"]
code_prompt_template = config["code_prompt_template"]

print('=='*50)
print(f"Welcome to fireGPT version {version} - FastAPI Backend")
print(os.getcwd())
print('=='*50)

os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
for json_file in [FEEDBACK_JSON, IMAGE_JSON]:
    if not os.path.exists(json_file):
        with open(json_file, 'w') as f:
            json.dump([], f)
dbf.initialize_counters(FEEDBACK_JSON, IMAGE_JSON)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

system_prompt_template = PromptTemplate(
    input_variables=["query", "context"],
    template=system_template1
)

embed = OllamaEmbeddings(
    model=embedding_model,
    base_url=base_url
)

llm = OllamaLLM(
    base_url=base_url,
    model=llm_model
)

image_model = OllamaLLM(
    base_url=base_url,
    model=image_model_name
)

temp_dir = tempfile.TemporaryDirectory()
temporary_Vector_db_path = temp_dir.name

conversation_histories = {}
user_file_contexts = {}

# Memory storage for each user - message history approach
user_message_histories = {}
# Memory storage for incognito sessions
incognito_message_histories = {}

main_faiss_db = None

# --- Memory Management Functions ---

def get_or_create_message_history(user_id, session_id=None):
    """Get or create message history for a user session"""
    memory_key = f"{user_id}_{session_id}" if session_id else user_id
    
    if memory_key not in user_message_histories:
        user_message_histories[memory_key] = ChatMessageHistory()
    
    return user_message_histories[memory_key]

def get_or_create_incognito_history(user_id):
    """Get or create temporary message history for an incognito session"""
    incognito_key = f"incognito_{user_id}"
    
    if incognito_key not in incognito_message_histories:
        incognito_message_histories[incognito_key] = ChatMessageHistory()
    
    return incognito_message_histories[incognito_key]

def flush_incognito_history(user_id):
    """Flush incognito history when incognito is turned off"""
    incognito_key = f"incognito_{user_id}"
    
    if incognito_key in incognito_message_histories:
        del incognito_message_histories[incognito_key]
        print(f"Flushed incognito history for user {user_id}")
        return True
    return False

def load_existing_chat_into_history(user_id, chat_id, session_id):
    """Load existing chat history into message history when user clicks on a chat button"""
    try:
        # Load the specific chat from your JSON storage
        chat_data = dbf.load_specific_chat_history(user_id, chat_id, TEST_OUTPUT_DIR)
        
        if not chat_data:
            return get_or_create_message_history(user_id, session_id)
            
        message_history = get_or_create_message_history(user_id, session_id)
        
        # Clear existing history for this session
        message_history.clear()
        
        # Load messages from the chat into history
        if "messages" in chat_data:
            for message in chat_data["messages"]:
                if "userprompt" in message and "response" in message:
                    # Add to message history
                    message_history.add_user_message(message["userprompt"])
                    message_history.add_ai_message(message["response"])
        
        return message_history
    except Exception as e:
        print(f"Error loading chat into message history: {e}")
        return get_or_create_message_history(user_id, session_id)

# --- Helper Functions ---

def get_appropriate_model(query):
    """
    Determines the appropriate model to use based on the query content.
    If the query contains code-related keywords, the code model is used.
    Otherwise, the default LLM model is used.
    
    Args:
        query (str): The user's query
        
    Returns:
        tuple: (model, is_code_query)
            - model: The LLM model instance to use
            - is_code_query: Boolean indicating if this is a code-related query
    """
    # Get code keywords from config
    code_keywords = config.get("code_keywords", [])
    
    # Check if any code keywords are in the query
    is_code_query = any(keyword.lower() in query.lower() for keyword in code_keywords)
    
    if is_code_query:
        # Initialize code model if needed
        if not hasattr(get_appropriate_model, "code_model"):
            # Create code model instance
            get_appropriate_model.code_model = OllamaLLM(
                base_url=base_url,
                model=code_model_name
            )
            print(f"Code model {code_model_name} initialized")
        
        print(f"Using code model for query: '{query[:50]}...' (if truncated)")
        return get_appropriate_model.code_model, True
    else:
        # Use regular LLM
        print(f"Using standard LLM for query: '{query[:50]}...' (if truncated)")
        return llm, False

def remove_think_blocks(text):
    """
    Removes <think>...</think> blocks from the response text.
    
    Args:
        text (str): The text containing potential think blocks
        
    Returns:
        str: Text with think blocks removed
    """
    # Check if the text contains think blocks
    if "<think>" in text and "</think>" in text:
        # Split by think blocks and rejoin without them
        parts = text.split("<think>")
        result = parts[0]  # First part before any think block
        
        for part in parts[1:]:
            if "</think>" in part:
                # Add only the content after the think block
                result += part.split("</think>", 1)[1]
            else:
                # If no closing tag, add the whole part (should not happen)
                result += part
                
        return result.strip()
    
    # If no think blocks found, return the original text
    return text

def create_contextual_prompt_with_model(query, context, message_history, model, is_code_query):
    """
    Create a contextual prompt with conversation history, optimized for the selected model
    """
    # Let the model decide if conversation history is relevant
    use_history = determine_context_relevance(query, message_history, model)
    
    # Format conversation history if relevant
    conversation_context = ""
    if use_history and message_history.messages:
        # Include history formatting code...
        messages = message_history.messages[-10:] if len(message_history.messages) > 10 else message_history.messages
        conversation_context = "\n\nPrevious conversation:\n"
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                human_msg = messages[i].content if hasattr(messages[i], 'content') else str(messages[i])
                ai_msg = messages[i + 1].content if hasattr(messages[i + 1], 'content') else str(messages[i + 1])
                conversation_context += f"Human: {human_msg}\nAssistant: {ai_msg}\n\n"
    
    # Create appropriate prompt with or without history
    if is_code_query:
        full_prompt = code_prompt_template.format(context=context, query=query)
        if use_history:
            full_prompt += conversation_context
    else:
        if use_history:
            full_prompt = f"""{system_template1} Context from documents:{context}{conversation_context}Current question: {query} Please provide a comprehensive answer based on the context above and our conversation history."""
        else:
            full_prompt = f"""{system_template1} Context from documents:{context}\nCurrent question: {query} Please provide a comprehensive answer based on the context."""
    
    return full_prompt

def determine_context_relevance(query, message_history, model):
    """Let the LLM itself decide if previous context is relevant"""
    
    # If no substantial history, no need to check
    if len(message_history.messages) < 4:  # Need at least 2 exchanges
        return False
    
    # Extract last 2-3 conversation exchanges
    recent_exchanges = []
    messages = message_history.messages[-6:]  # Last 3 exchanges (6 messages)
    
    for i in range(0, len(messages), 2):
        if i + 1 < len(messages):
            human_msg = messages[i].content if hasattr(messages[i], 'content') else str(messages[i])
            ai_msg = messages[i + 1].content if hasattr(messages[i + 1], 'content') else str(messages[i + 1])
            recent_exchanges.append(f"Human: {human_msg}\nAssistant: {ai_msg}")
    
    recent_context = "\n\n".join(recent_exchanges)
    
    # Create prompt for the model to evaluate context relevance
    relevance_prompt = f"""
    I need to determine if the previous conversation context is relevant to the new query.
    
    Previous conversation:
    {recent_context}
    
    New query: "{query}"
    
    Is the previous conversation context relevant to properly answering this new query?
    Answer with just "RELEVANT" or "NOT_RELEVANT".
    """
    
    # Ask the model to evaluate
    try:
        response = model.invoke(relevance_prompt).strip()
        return "RELEVANT" in response.upper()
    except Exception as e:
        print(f"Error determining context relevance: {e}")
        # Default to including context if we can't determine
        return True

def setup_retrievers(vector_db, llm):
    """
    Set up retrievers with hybrid search capabilities
    """
    # Base retriever with lower threshold for better results
    base_retriever = vector_db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 8,
            "score_threshold": 0.3  # Lowered from 0.5 to get more results
        }
    )
    
    # MMR retriever for diversity in results
    mmr_retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 12,
            "lambda_mult": 0.7
        }
    )
    
    # Fallback similarity retriever (no threshold)
    similarity_retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8}
    )
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[base_retriever, mmr_retriever],
        weights=[0.7, 0.3]
    )
    
    return {
        "Base": base_retriever,
        "MMR": mmr_retriever,
        "Similarity": similarity_retriever,
        "Ensemble": ensemble_retriever
    }

@lru_cache(maxsize=100)
def cached_retrieve_docs(query_hash, db_path, k=8):
    """Cache document retrieval for performance"""
    vector_db = FAISS.load_local(
        folder_path=db_path,
        index_name=index_name,
        embeddings=embed,
        allow_dangerous_deserialization=True
    )
    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    return retriever.invoke(query_hash)

def retrieve_and_answer_with_memory(vector_db, query, user_id, session_id=None, chat_id=None):
    """Enhanced retrieval that uses message history for context AND appropriate model selection"""
    
    try:
        # Get or create message history for this user/session
        if chat_id and session_id:
            # If loading existing chat, load its history
            message_history = load_existing_chat_into_history(user_id, chat_id, session_id)
        else:
            # Use or create history for current session
            message_history = get_or_create_message_history(user_id, session_id)
        
        # SELECT APPROPRIATE MODEL BASED ON QUERY CONTENT
        model, is_code_query = get_appropriate_model(query)
        
        # Log which model is being used
        if is_code_query:
            print(f"\n--- USING MODEL: {code_model_name} FOR MEMORY-AWARE RETRIEVAL ---")
        else:
            print(f"\n--- USING MODEL: {llm_model} FOR MEMORY-AWARE RETRIEVAL ---")
        
        # Set up retrievers and try multiple approaches
        retrievers = setup_retrievers(vector_db, model)  # Pass the selected model
        docs = []
        
        # Try retrievers in order of preference
        for retriever_name in ["Base", "Similarity", "MMR"]:
            if retriever_name in retrievers:
                try:
                    retriever = retrievers[retriever_name]
                    docs = retriever.invoke(query)
                    if docs:
                        print(f"Successfully retrieved {len(docs)} documents using {retriever_name} retriever")
                        break
                except Exception as e:
                    print(f"Error with {retriever_name} retriever: {e}")
                    continue
        
        # If no documents found, provide a general response
        if not docs:
            print("No relevant documents found, providing general response")
            context = "No specific document context available."
        else:
            # Extract context from documents
            context_parts = []
            sources = set()
            
            for doc in docs:
                context_parts.append(doc.page_content)
                if "source" in doc.metadata:
                    sources.add(os.path.basename(doc.metadata["source"]))
            
            context = "\n\n---\n\n".join(context_parts)
        
        # Create contextual prompt with conversation history and appropriate model
        full_prompt = create_contextual_prompt_with_model(query, context, message_history, model, is_code_query)
        
        # Get response from the SELECTED MODEL
        response = model.invoke(full_prompt)
        answer = str(response)
        
        # Remove think blocks if present (for code model)
        if is_code_query:
            answer = remove_think_blocks(answer)
        
        # Add source citations if available
        if docs:
            sources = set()
            for doc in docs:
                if "source" in doc.metadata:
                    sources.add(os.path.basename(doc.metadata["source"]))
            
            if sources and "Sources:" not in answer:
                answer += f"\n\nSources: {', '.join(sources)}"
        
        # Update message history with new exchange
        message_history.add_user_message(query)
        message_history.add_ai_message(answer)
        
        # Limit history size
        if len(message_history.messages) > MAX_MEMORY_MESSAGES * 2:  # *2 because we have pairs
            # Keep only the most recent messages
            message_history.messages = message_history.messages[-MAX_MEMORY_MESSAGES * 2:]
        
        print(f"--- END OF {code_model_name if is_code_query else llm_model} RESPONSE ---\n")
        return answer
        
    except Exception as e:
        print(f"Error in memory-aware retrieval: {e}")
        # Fallback to original method
        return retrieve_and_answer_with_best_retriever(vector_db, query, [])

def handle_query_with_appropriate_model(query, vector_db, conversation_history, user_id=None, session_id=None, chat_id=None):
    """Use memory-based retrieval if user_id is provided, otherwise fall back to conversation history"""
    if user_id:
        # Use memory-based retrieval
        return retrieve_and_answer_with_memory(vector_db, query, user_id, session_id, chat_id)
    else:
        # Legacy method using conversation history
        model, is_code_query = get_appropriate_model(query)
        if is_code_query:
            retrievers = setup_retrievers(vector_db, model)
            retriever = retrievers.get("Ensemble", retrievers["Base"])
            docs = retriever.invoke(query)
            context_parts = []
            sources = set()
            for doc in docs:
                context_parts.append(doc.page_content)
                if "source" in doc.metadata:
                    sources.add(doc.metadata["source"])
                elif "filename" in doc.metadata:
                    sources.add(doc.metadata["filename"])
            context = "\n\n---\n\n".join(context_parts)
            code_prompt = config["code_prompt_template"].format(
                context=context,
                query=query
            )
            response = model.invoke(code_prompt)
            raw_result = str(response)
            result = remove_think_blocks(raw_result)
            if sources:
                source_list = ", ".join(os.path.basename(source) for source in sources)
                if "Sources:" not in result:
                    result += f"\n\nSources: {source_list}"
        else:
            result = retrieve_and_answer_with_best_retriever(vector_db, query, conversation_history)
        return result

def retrieve_and_answer_with_best_retriever(vector_db, query, conversation_history):
    """Use advanced retrieval techniques to get the best response with model selection"""
    
    # SELECT APPROPRIATE MODEL
    model, is_code_query = get_appropriate_model(query)
    
    # Log which model is being used
    if is_code_query:
        print(f"\n--- USING MODEL: {code_model_name} FOR BEST RETRIEVER ---")
    else:
        print(f"\n--- USING MODEL: {llm_model} FOR BEST RETRIEVER ---")
    
    # Query expansion for better retrieval
    expanded_query = query
    if conversation_history and len(conversation_history) > 0:
        # Get last exchange to provide context
        last_exchange = conversation_history[-1]
        # Simple context augmentation
        if "Q:" in last_exchange and len(query.split()) < 15:
            expanded_query = f"{query} (Follow-up to: {last_exchange.split('Q:')[1].split('A:')[0].strip()})"
    
    # Setup retrievers
    retrievers = setup_retrievers(vector_db, model)
    
    # Try with different retrievers
    docs = []
    for retriever_name in ["Similarity", "Base", "MMR"]:  # Try similarity first as it's most reliable
        if retriever_name in retrievers:
            try:
                retriever = retrievers[retriever_name]
                docs = retriever.invoke(expanded_query)
                if docs:
                    break
            except Exception as e:
                print(f"Error with {retriever_name} retriever: {e}")
                continue
    
    try:
        # Check if we have sufficient documents
        if len(docs) == 0:
            # Fallback: retry with simpler query if we got no results
            simple_query = " ".join(query.split()[:6]) if len(query.split()) > 6 else query
            try:
                retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 6})
                docs = retriever.invoke(simple_query)
            except:
                pass
            
            if len(docs) == 0:
                return "I couldn't find any relevant information in the documents to answer your question. Could you rephrase your query or ask about a different topic covered in the documentation?"
        
        # Extract context and sources with better formatting
        context_parts = []
        sources = set()
        
        for doc in docs:
            # Add document content with metadata
            doc_part = doc.page_content
            
            # Add page number reference if available
            if 'page' in doc.metadata:
                doc_part += f" [Page {doc.metadata['page']}]"
            
            context_parts.append(doc_part)
            
            # Add source information
            if "source" in doc.metadata:
                sources.add(doc.metadata["source"])
            elif "filename" in doc.metadata:
                sources.add(doc.metadata["filename"])
        
        # Join context with clear document separators
        context = "\n\n---\n\n".join(context_parts)
        
        # Format prompt based on model type
        if is_code_query:
            # Use code-specific prompt
            code_prompt_template = config.get("code_prompt_template")            
            formatted_prompt = code_prompt_template.format(
                context=context,
                query=query
            )
        else:
            # Use regular prompt
            formatted_prompt = system_prompt_template.format(
                query=query,
                context=context
            )
        
        # Get response from SELECTED MODEL
        response = str(model.invoke(formatted_prompt))
        
        # Remove think blocks if code model
        if is_code_query:
            response = remove_think_blocks(response)
        
        # Post-process response to ensure it's well-formatted
        if not response.strip():
            response = "I couldn't generate a meaningful response from the available information."
        
        # Make sure sources are included
        if sources and "Sources:" not in response:
            source_list = ", ".join(os.path.basename(source) for source in sources)
            result = f"{response}\n\nSources: {source_list}"
        else:
            result = response
        
        print(f"--- END OF {code_model_name if is_code_query else llm_model} RESPONSE ---\n")
        return result
    except Exception as e:
        print(f"Error with enhanced retrieval: {e}")
        return f"Unable to answer the query due to an error: {str(e)}"

def extract_code_block(bot_message):
    """
    Extracts the first code block from a bot message.
    Code blocks are identified by triple backticks (```).
    
    Args:
        bot_message (str): The message from the bot containing code blocks
        
    Returns:
        str or None: The extracted code block without language identifier,
                     or None if no code block is found
    """
    # Check if message contains code blocks
    if "```" not in bot_message:
        return None
    
    # Split the message by code block markers
    parts = bot_message.split("```")
    
    # We need at least 3 parts for a complete code block (text before, code, text after)
    if len(parts) < 3:
        return None
    
    # Extract the content of the first code block
    code_block = parts[1].strip()
    
    # Remove language identifier if present (like "python", "javascript", etc.)
    if '\n' in code_block:
        first_line = code_block.split('\n', 1)[0].strip()
        # Check if the first line is a language identifier (no code symbols)
        if first_line and not any(symbol in first_line for symbol in ['#', '//', '/*', '*', '<', '=']):
            if len(code_block.split('\n', 1)) > 1:
                code_block = code_block.split('\n', 1)[1]
    
    return code_block

def handle_code_block_extraction(query, result):
    """
    Checks if the query is related to code and extracts the code block from the result if present.

    Args:
        query (str): The user query.
        result (str): The response from which the code block needs to be extracted.

    Returns:
        None
    """
    code_keywords = CODE_KEYWORDS
    if any(keyword.lower() in query.lower() for keyword in code_keywords):
        # Extract code block from the response
        code_block = extract_code_block(result)
        if code_block:
            print("\n--- EXTRACTED CODE BLOCK ---")
            print(code_block)
            print("--- END OF CODE BLOCK ---\n")

def validate_and_improve_response(response, query):
    """Check response quality and improve if needed"""
    
    # Check if response seems empty or generic
    if len(response.strip()) < 50 or "I don't have enough information" in response:
        return response + "\n\nNote: The information available in the documents may be limited for this specific query."
    
    # Check if response contains source citations
    if "Sources:" not in response and "sources:" not in response:
        return response + "\n\nNote: Please refer to the original documents for verification."
    
    # Check if query asked for specific numeric values and response contains numbers
    if any(term in query.lower() for term in ['how many', 'how much', 'what percentage', 'what value']):
        if not any(char.isdigit() for char in response):
            return response + "\n\nNote: The specific numeric values you requested may not be explicitly stated in the documents."
    
    return response

def process_diagram_request(query):
    """
    Check if the query is requesting a diagram and determine if a specific format is requested.
    If no format is specified, default to Graphviz DOT syntax.
    
    Args:
        query (str): The user query
        
    Returns:
        tuple: (is_diagram_request, modified_query)
            - is_diagram_request (bool): True if the query is asking for a diagram
            - modified_query (str): Original query or modified to specify Graphviz
    """
    # Check if query is asking for a diagram
    diagram_keywords = DIAGRAM_KEYWORDS
    is_diagram_request = any(keyword in query.lower() for keyword in diagram_keywords)
    
    if not is_diagram_request:
        return False, query
        
    # Check if a specific format is already mentioned
    format_keywords = FORMAT_KEYWORDS
    has_format_specified = any(keyword in query.lower() for keyword in format_keywords)
    
    # If no format is specified, add Graphviz to the query with DOT syntax instruction
    if not has_format_specified:
        modified_query = query + " using Graphviz DOT syntax (not Python code)"
        return True, modified_query
    elif "graphviz" in query.lower() and "dot syntax" not in query.lower():
        # Ensure DOT syntax is specified for Graphviz
        modified_query = query + " (use direct DOT syntax, not Python code)"
        return True, modified_query
    
    return True, query

def fix_diagram_syntax_with_llm(original_code, error_message, original_query):
    """
    Ask the LLM to fix syntax errors in diagram code
    
    Args:
        original_code (str): The original code with syntax errors
        error_message (str): The error message from the diagram tool
        original_query (str): The original user query for context
        
    Returns:
        str: Fixed code or None if couldn't fix
    """
    try:
        # Create a prompt for fixing the syntax error
        fix_prompt = config.get("fix_prompt_template", """
You are a diagram code expert. The following diagram code has syntax errors. Please fix the code.

Original query: {original_query}

Original code:
{original_code}

Error message:
{error_message}

Please provide ONLY the fixed code, with no explanations or markdown formatting.
        """).format(
            original_code=original_code,
            error_message=error_message,
            original_query=original_query
        )

        # Use the code model for fixing syntax
        code_model, _ = get_appropriate_model("fix code syntax")  # This will return code model
        
        print("🤖 Sending fix request to LLM...")
        response = code_model.invoke(fix_prompt)
        fixed_code = str(response).strip()
        
        # Remove think blocks if present
        fixed_code = remove_think_blocks(fixed_code)
        
        # Clean up the response (remove any markdown formatting if present)
        if "```" in fixed_code:
            # Extract code from markdown if present
            parts = fixed_code.split("```")
            if len(parts) >= 3:
                fixed_code = parts[1].strip()
                # Remove language identifier if present
                if '\n' in fixed_code:
                    lines = fixed_code.split('\n')
                    if lines[0].strip() in ['dot', 'graphviz', 'digraph']:
                        fixed_code = '\n'.join(lines[1:])
        
        # Basic validation - check if it looks like valid DOT syntax
        if "digraph" in fixed_code and "{" in fixed_code and "}" in fixed_code:
            print("✅ LLM provided what appears to be valid DOT syntax")
            return fixed_code
        else:
            print("❌ LLM response doesn't look like valid DOT syntax")
            return None
            
    except Exception as e:
        print(f"❌ Error asking LLM to fix syntax: {e}")
        return None

def process_and_display_diagram_with_retry(query, result, max_attempts=3):
    """
    Enhanced diagram processing with automatic error correction and retry mechanism
    
    Args:
        query (str): The user query
        result (str): The response containing the code block
        max_attempts (int): Maximum number of retry attempts
    
    Returns:
        dict or str: Dictionary containing success message and image data, or error message string
    """
    
    # Extract the initial code block
    code_block = extract_code_block(result)
    if not code_block:
        return "No code block found in the response."

    # Determine the type of diagram based on the query
    output_path = os.path.join(TEST_OUTPUT_DIR, f"diagram_{int(time.time())}")
    
    # Determine diagram type
    if "graphviz" in query.lower():
        diagram_type = "graphviz"
    elif "mermaid" in query.lower():
        diagram_type = "mermaid"
    else:
        diagram_type = "graphviz"  # Default to graphviz
    
    current_code = code_block
    retry_count = 0
    
    while retry_count < max_attempts:
        try:
            print(f"Attempting diagram generation (attempt {retry_count + 1}/{max_attempts})...")
            
            # Try to generate the diagram
            if diagram_type == "graphviz":
                graphviz_tool = GraphvizTool()
                response = graphviz_tool.run(current_code, f"{output_path}_attempt_{retry_count}")
            elif diagram_type == "mermaid":
                mermaid_tool = MermaidTool()
                response = mermaid_tool.run(current_code, f"{output_path}_attempt_{retry_count}")
            
            # Check if the response is successful
            if isinstance(response, dict) and "image_data" in response:
                # Success! Save image metadata and return
                image_id, image_hash = dbf.save_image_to_db(response["image_path"], "system", IMAGE_JSON, API_BASE_URL)
                
                print(f"✅ Diagram generated successfully on attempt {retry_count + 1}")
                return {
                    "message": f"{response['message']} (Generated after {retry_count + 1} attempt{'s' if retry_count > 0 else ''})",
                    "image_data": response["image_data"],
                    "image_format": response["format"],
                    "image_id": image_id,
                    "image_hash": image_hash
                }
            elif isinstance(response, str) and "Error:" in response:
                # Error occurred, try to fix it
                print(f"❌ Diagram generation failed on attempt {retry_count + 1}: {response}")
                
                if retry_count < max_attempts - 1:  # Don't retry on last attempt
                    print("🔄 Attempting to fix the code...")
                    current_code = fix_diagram_syntax_with_llm(current_code, response, query)
                    if current_code:
                        print("✅ Code correction received, retrying...")
                        retry_count += 1
                        continue
                    else:
                        print("❌ Failed to get code correction")
                        break
                else:
                    print("❌ Max retries reached")
                    break
            else:
                # Unexpected response format
                print(f"❌ Unexpected response format: {response}")
                break
                
        except Exception as e:
            print(f"❌ Exception during diagram generation: {e}")
            if retry_count < max_attempts - 1:
                print("🔄 Attempting to fix the code due to exception...")
                current_code = fix_diagram_syntax_with_llm(current_code, str(e), query)
                if current_code:
                    retry_count += 1
                    continue
                else:
                    break
            else:
                break
    
    # If we get here, all retries failed
    return f"Failed to generate diagram after {max_attempts} attempts. Last error: {response if 'response' in locals() else 'Unknown error'}"

def process_and_display_diagram(query, result):
    """Process and display diagram - now calls the enhanced version with retry"""
    return process_and_display_diagram_with_retry(query, result, max_attempts=3)

# --- Test Case Generation Functions ---
def is_test_case_generation_request(query):
    """
    Detect if the query is asking for manual test case generation.
    
    Args:
        query (str): The user query
        
    Returns:
        bool: True if query is about manual test case generation
    """
    test_case_keywords = [
        "generate test case", "create test case", "test case generation",
        "manual test case", "create manual test", "generate manual test",
        "testing scenario", "test scenario", "qa test case", "quality assurance test"
    ]
    
    # Check if query contains test case related keywords
    return any(keyword.lower() in query.lower() for keyword in test_case_keywords)

def generate_test_cases(query, vector_db_path, user_id, session_id=None, chat_id=None):
    """
    Generate manual test cases using both ContextTool and ManualTestCaseTool.
    
    Args:
        query (str): The user query
        vector_db_path (str): Path to the vector database
        user_id (str): User identifier
        session_id (str, optional): Session identifier for message history
        chat_id (str, optional): Chat identifier for existing conversations
        
    Returns:
        str: Generated test cases or error message
    """
    try:
        print("\n=== BACKEND: GENERATING MANUAL TEST CASES ===")
        print(f"User ID: {user_id}")
        print(f"Session ID: {session_id}")
        print(f"Chat ID: {chat_id}")
        print(f"Vector DB path: {vector_db_path}")
        
        # Get or create message history for this user/session
        if chat_id and session_id:
            # If loading existing chat, load its history
            message_history = load_existing_chat_into_history(user_id, chat_id, session_id)
            print(f"Loaded existing chat history for chat_id: {chat_id}")
        else:
            # Use or create history for current session
            message_history = get_or_create_message_history(user_id, session_id)
            print(f"Using session history for session_id: {session_id}")
        
        print("Generating context using contextTool...")
        context = contextTool.generate_context(query, base_url.split("//")[1].split(":")[0])  # Extract IP from base_url
        
        if context and not context.startswith("Error:"):
            print(f"Context generated successfully ({len(context)} characters)")
        else:
            print("Failed to generate context, proceeding with only vector DB")
            context = None
        
        print("Generating test cases with context and vector DB...")
        result = manualTestCase.generate_manual_test_cases(query, vector_db_path, base_url.split("//")[1].split(":")[0], context)
        print("Test cases generated successfully")
        
        # Update message history with new exchange
        message_history.add_user_message(query)
        message_history.add_ai_message(result)
        print("Updated message history with test case query and result")
        
        # Limit history size
        if len(message_history.messages) > MAX_MEMORY_MESSAGES * 2:  # *2 because we have pairs
            # Keep only the most recent messages
            message_history.messages = message_history.messages[-MAX_MEMORY_MESSAGES * 2:]
            print(f"Trimmed message history to {MAX_MEMORY_MESSAGES} exchanges")
        
        print("=== BACKEND: MANUAL TEST CASE GENERATION COMPLETED ===\n")
        return result
    except Exception as e:
        error_msg = f"Error generating test cases: {e}"
        print(f"❌ {error_msg}")
        print("=== BACKEND: MANUAL TEST CASE GENERATION FAILED ===\n")
        return error_msg

# --- File Processing Functions ---
def clean_user_files(user_id):
    try:
        user_dir = os.path.join(temporary_Vector_db_path, user_id)
        if os.path.exists(user_dir):
            shutil.rmtree(user_dir)
        if user_id in user_file_contexts:
            del user_file_contexts[user_id]
    except Exception as e:
        print(f"Error cleaning user files: {e}")

def image_handling(image_path, prompt=None):
    if not prompt:
        prompt = DEFAULT_IMAGE_PROMPT
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        response = image_model.invoke(prompt, images=[image_data])
        return response
    except Exception as e:
        print(f"Error extracting details from image: {e}")
        return {"error": str(e)}

def get_user_file_directory(user_id):
    user_dir = os.path.join(temporary_Vector_db_path, user_id)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

def extract_text_from_doc(file_path):
    try:
        # Try a simple binary read approach - this won't produce clean text
        # but can extract some content without dependencies
        with open(file_path, 'rb') as file:
            content = file.read()
            
        # Extract text from binary content (very basic approach)
        text = ""
        in_text = False
        for i in range(len(content)-1):
            if content[i] == 0 and content[i+1] == 0:
                in_text = False
            elif 32 <= content[i] <= 126 and in_text:  # ASCII printable characters
                text += chr(content[i])
            elif not in_text and 32 <= content[i] <= 126:
                possible_text = ""
                j = i
                while j < len(content) and 32 <= content[j] <= 126:
                    possible_text += chr(content[j])
                    j += 1
                    
                if len(possible_text) > 20:  # Only consider chunks of reasonable length
                    text += possible_text + "\n"
                    in_text = True
        
        # Clean up extracted text
        text = ' '.join(text.split())  # Remove extra whitespace
        
        # Create document with extracted text
        from langchain.schema.document import Document
        return [Document(page_content=text, metadata={"source": file_path})]
    except Exception as e:
        print(f"Error extracting text from .doc file: {e}")
        from langchain.schema.document import Document
        return [Document(page_content="Error extracting text from document.", metadata={"source": file_path})]

def get_loader(file_path, doc):
    file_extension = os.path.splitext(doc.lower())[1]
    
    loaders = {
        '.pdf': PyPDFLoader(file_path, extract_images=False),
        '.docx': Docx2txtLoader(file_path),  # Doesn't require LibreOffice
        '.txt': TextLoader(file_path, encoding='utf-8'),
        '.csv': CSVLoader(file_path)
    }
    
    # Get the specific loader or return None if not found
    return loaders.get(file_extension)

def load_excel_or_csv(file_path, doc):
    try:
        if doc.lower().endswith('.csv'):
            df = pd.read_csv(file_path, on_bad_lines='skip', low_memory=False)
        else:
            # Read all sheets from Excel file
            excel_file = pd.ExcelFile(file_path)
            sheets = excel_file.sheet_names
            dataframes = []
            
            for sheet in sheets:
                try:
                    sheet_df = pd.read_excel(file_path, sheet_name=sheet)
                    # Add sheet information to dataframe
                    sheet_df['_sheet_name'] = sheet
                    dataframes.append(sheet_df)
                except Exception as e:
                    print(f"Error reading sheet {sheet}: {e}")
                    continue
            
            # Combine all sheets
            if dataframes:
                df = pd.concat(dataframes, ignore_index=True)
            else:
                from langchain.schema.document import Document
                return [Document(page_content=f"Failed to extract data from Excel file {file_path}", 
                                metadata={"source": file_path})]
        
        # Handle non-string data more robustly
        for col in df.columns:
            df[col] = df[col].astype(str).replace('nan', '').replace('None', '')
        
        # More structured text representation of dataframe with headers
        records_as_text = []
        
        # Add column headers for context
        headers = "| " + " | ".join(col for col in df.columns if col != '_sheet_name') + " |"
        header_sep = "| " + " | ".join("-" * len(col) for col in df.columns if col != '_sheet_name') + " |"
        
        for i in range(df.shape[0]):
            row = df.iloc[i]
            if '_sheet_name' in row:
                sheet_name = row['_sheet_name']
                row_text = f"Sheet: {sheet_name}\n"
            else:
                row_text = ""
            
            # Add row data in table format
            row_text += "| "
            for col in df.columns:
                if col != '_sheet_name':
                    cell_value = str(row[col]).strip()
                    row_text += f"{cell_value} | "
            
            if i % 50 == 0:
                # Add headers periodically to maintain context
                row_text = f"{headers}\n{header_sep}\n{row_text}"
            
            records_as_text.append(row_text)
        
        # Advanced text splitting for better context preservation
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200, 
            separators=["\n\n", "\n", " ", ",", "", "\n- "]
        )
        from langchain.schema.document import Document
        documents = text_splitter.create_documents(records_as_text)
        
        # Add enhanced metadata to documents
        for i, doc_chunk in enumerate(documents):
            doc_chunk.metadata = {
                "source": file_path,
                "filename": os.path.basename(file_path),
                "file_type": "excel" if doc.lower().endswith(('.xls', '.xlsx')) else "csv",
                "chunk_id": i,
                "total_chunks": len(documents)
            }
            
        return documents
    except Exception as e:
        print(f"Error processing Excel/CSV file {file_path}: {e}")
        # Fallback to CSV loader if custom processing fails
        if doc.lower().endswith('.csv'):
            try:
                loader = CSVLoader(file_path)
                return loader.load()
            except:
                pass
        # Ultimate fallback - return error document
        from langchain.schema.document import Document
        return [Document(page_content=f"Error processing file: {str(e)}", 
                        metadata={"source": file_path, "error": str(e)})]

def split_document(documents, chunk_size=1000, chunk_overlap=300):
    # Determine document type to use specialized splitting
    doc_types = set()
    for doc in documents:
        if hasattr(doc, 'metadata') and "file_type" in doc.metadata:
            doc_types.add(doc.metadata["file_type"])
    
    # Adjust chunk size based on document type
    if "pdf" in doc_types:
        chunk_size = 1200  # Larger chunks for PDFs
        chunk_overlap = 400
    elif "excel" in doc_types or "csv" in doc_types:
        chunk_size = 800   # Smaller chunks for structured data
        chunk_overlap = 250
    
    # Use character splitter for reliability with improved separators
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", ":", ";", ",", " ", ""]
    )
    
    try:
        # Split documents and preserve all metadata
        docs = char_splitter.split_documents(documents)
        
        # Add chunk position metadata
        for i, doc in enumerate(docs):
            if hasattr(doc, 'metadata'):
                doc.metadata["chunk_position"] = i
                doc.metadata["total_chunks"] = len(docs)
        
        print(f"Split documents into {len(docs)} chunks.")
        return docs
    except Exception as e:
        print(f"Error splitting documents: {e}")
        return documents  # Return original if splitting fails

def handle_document_upload(file, user_id):
    # Create user-specific directory
    user_dir = get_user_file_directory(user_id)
    
    # Save file to user's directory
    file_path = os.path.join(user_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read() if hasattr(file, "file") else file.read())
    
    # Process document based on extension
    file_extension = os.path.splitext(file.filename.lower())[1]
    
    if file_extension == '.doc':
        # Handle .doc files with our custom extraction function
        documents = extract_text_from_doc(file_path)
    else:
        # Handle other document types
        loader = get_loader(file_path, file.filename)
        if loader:
            documents = loader.load()
        else:
            # Fallback for unsupported file types
            from langchain.schema.document import Document
            documents = [Document(page_content=f"Unsupported file type: {file_extension}", metadata={"source": file_path})]
    
    # Add detailed metadata to each document
    for i, doc in enumerate(documents):
        if not hasattr(doc, 'metadata'):
            doc.metadata = {}
        doc.metadata.update({
            "source": file_path,
            "filename": file.filename,
            "page_number": i if 'page' not in doc.metadata else doc.metadata['page'],
            "file_type": file_extension
        })
    
    # Use more optimal chunking for better context preservation
    chunked_docs = split_document(documents, chunk_size=1000, chunk_overlap=400)
    
    # Create vector database in user's directory
    user_db_path = os.path.join(user_dir, "vector_db")
    os.makedirs(user_db_path, exist_ok=True)
    user_db = FAISS.from_documents(chunked_docs, embed)
    user_db.save_local(user_db_path, index_name=index_name)
    
    # Set up user's context
    user_file_contexts[user_id] = {
        "context_type": "file",
        "file_path": file_path,
        "file_name": file.filename,
        "vector_db_path": user_db_path,
        "total_chunks": len(chunked_docs)
    }
    
    print(f"File '{file.filename}' uploaded and processed into {len(chunked_docs)} chunks")
    return user_db

def handle_excel_upload(file, user_id):
    # Create user-specific directory
    user_dir = get_user_file_directory(user_id)
    
    # Save file to user's directory
    excel_path = os.path.join(user_dir, file.filename)
    with open(excel_path, "wb") as f:
        f.write(file.file.read() if hasattr(file, "file") else file.read())
    
    try:
        documents = load_excel_or_csv(excel_path, file.filename)
    except Exception as e:
        print(f"Error loading documents from {file.filename}: {e}")
        return None, f"Error loading documents from {file.filename}: {e}"
   
    # Use more optimal chunking
    chunked_docs = split_document(documents, chunk_size=800, chunk_overlap=300)
    
    # Create vector database in user's directory
    user_db_path = os.path.join(user_dir, "vector_db")
    os.makedirs(user_db_path, exist_ok=True)
    user_db = FAISS.from_documents(chunked_docs, embed)
    user_db.save_local(user_db_path, index_name=index_name)
    
    # Set up user's context
    user_file_contexts[user_id] = {
        "context_type": "file",
        "file_path": excel_path,
        "file_name": file.filename,
        "vector_db_path": user_db_path,
        "total_chunks": len(chunked_docs)
    }
    
    print(f"Excel file '{file.filename}' uploaded and processed into {len(chunked_docs)} chunks")
    return user_db, None

def handle_text_upload(file, user_id):
    # Create user-specific directory
    user_dir = get_user_file_directory(user_id)
    
    # Save file to user's directory
    text_path = os.path.join(user_dir, file.filename)
    with open(text_path, "wb") as f:
        f.write(file.file.read() if hasattr(file, "file") else file.read())
    
    # Use TextLoader for better encoding handling
    loader = TextLoader(text_path, encoding='utf-8')
    documents = loader.load()
    
    # Add metadata to text content
    for i, doc in enumerate(documents):
        if not hasattr(doc, 'metadata'):
            doc.metadata = {}
        doc.metadata.update({
            "source": text_path,
            "filename": file.filename,
            "chunk_id": i,
            "file_type": "text"
        })
    
    # Use more optimal chunking for text files
    chunked_docs = split_document(documents, chunk_size=1200, chunk_overlap=400)
    
    # Create vector database in user's directory
    user_db_path = os.path.join(user_dir, "vector_db")
    os.makedirs(user_db_path, exist_ok=True)
    user_db = FAISS.from_documents(chunked_docs, embed)
    user_db.save_local(user_db_path, index_name=index_name)
    
    # Set up user's context
    user_file_contexts[user_id] = {
        "context_type": "file",
        "file_path": text_path,
        "file_name": file.filename,
        "vector_db_path": user_db_path,
        "total_chunks": len(chunked_docs)
    }
    
    print(f"Text file '{file.filename}' uploaded and processed into {len(chunked_docs)} chunks")
    return user_db

def handle_csv_upload(file, user_id):
    # Create user-specific directory
    user_dir = get_user_file_directory(user_id)
    
    # Save file to user's directory
    csv_path = os.path.join(user_dir, file.filename)
    with open(csv_path, "wb") as f:
        f.write(file.file.read() if hasattr(file, "file") else file.read())
    
    try:
        # Use our enhanced CSV loading function
        documents = load_excel_or_csv(csv_path, file.filename)
    except Exception as e:
        print(f"Error loading documents from {file.filename}: {e}")
        return None, f"Error loading documents from {file.filename}: {e}"
   
    # Use more optimal chunking
    chunked_docs = split_document(documents, chunk_size=800, chunk_overlap=300)
    
    # Create vector database in user's directory
    user_db_path = os.path.join(user_dir, "vector_db")
    os.makedirs(user_db_path, exist_ok=True)
    user_db = FAISS.from_documents(chunked_docs, embed)
    user_db.save_local(user_db_path, index_name=index_name)
    
    # Set up user's context
    user_file_contexts[user_id] = {
        "context_type": "file",
        "file_path": csv_path,
        "file_name": file.filename,
        "vector_db_path": user_db_path,
        "total_chunks": len(chunked_docs)
    }
    
    print(f"CSV file '{file.filename}' uploaded and processed into {len(chunked_docs)} chunks")
    return user_db, None

def handle_image_upload(file, user_id, query):
    # Create user-specific directory
    user_dir = get_user_file_directory(user_id)
    
    # Save image to user's directory
    image_path = os.path.join(user_dir, file.filename)
    with open(image_path, "wb") as f:
        f.write(file.file.read() if hasattr(file, "file") else file.read())
    
    # Get image ID and hash for database using dbf module
    image_id, image_hash = dbf.save_image_to_db(image_path, user_id, IMAGE_JSON, API_BASE_URL)
    
    # Store image context for future use
    user_file_contexts[user_id] = {
        "context_type": "image",
        "file_path": image_path,
        "file_name": file.filename,
        "image_id": image_id,
        "image_hash": image_hash
    }
    
    # First get a detailed general description
    general_description = image_handling(image_path)
    
    # Then process the specific query
    if query and query.strip():
        specific_answer = image_handling(image_path, query)
        # Combine both responses for more comprehensive result
        result = f"{specific_answer}\n\nAdditional image context: {general_description}"
    else:
        result = general_description
        
    return result, image_id, image_hash

# --- FastAPI Models ---
class ChatRequest(BaseModel):
    query: str
    user_id: Optional[str] = "default_user"
    chat_id: Optional[str] = None

class IncognitoChatRequest(ChatRequest):
    incognito_status: bool = True
    new_chat: bool = False
    end_session: bool = False

class FeedbackRequest(BaseModel):
    user_id: Optional[str] = "default_user"
    response_text: str = ""
    feedback: str = ""
    suggestion: str = ""
    is_image_response: bool = False

class NewChatRequest(BaseModel):
    user_id: Optional[str] = "default_user"
    incognito_status: bool = False

# --- FastAPI Endpoints ---

@app.post("/chat")
async def chat(request: ChatRequest):
    query = request.query
    user_id = request.user_id or "default_user"
    chat_id = request.chat_id
    
    is_diagram_request, modified_query = process_diagram_request(query)
    if is_diagram_request:
        query = modified_query

    if user_id not in conversation_histories:
        conversation_histories[user_id] = {'session_id': str(uuid.uuid4()), 'history': []}
    if isinstance(conversation_histories[user_id], dict):
        conversation_history = conversation_histories[user_id].get('history', [])
        session_id = conversation_histories[user_id].get('session_id')
    else:
        history = conversation_histories[user_id]
        session_id = str(uuid.uuid4())
        conversation_histories[user_id] = {'session_id': session_id, 'history': history}
        conversation_history = history
    
    # If continuing existing chat, get session_id for that chat
    if chat_id:
        existing_session_id = dbf.get_session_id_for_chat(user_id, chat_id, TEST_OUTPUT_DIR)
        if existing_session_id:
            session_id = existing_session_id
            conversation_histories[user_id]['session_id'] = session_id
    
    # Process test case generation requests
    is_test_case_request = is_test_case_generation_request(query)
    if is_test_case_request:
        print(f"Test case generation request detected: '{query}'")
        
        # Determine which vector DB to use
        db_path = None
        if user_id in user_file_contexts and user_file_contexts[user_id]["context_type"] == "file":
            db_path = user_file_contexts[user_id]["vector_db_path"]
        else:
            db_path = vector_db_path  # Use main vector DB
        
        # Generate test cases with context
        result = generate_test_cases(query, db_path, user_id, session_id, chat_id)
        
        # Update conversation history
        conversation_history.append(f"Q: {query}\nA: {result}")
        conversation_histories[user_id]['history'] = conversation_history
        
        # Save chat data to database
        dbf.save_chat_to_db(
            user_id=user_id, 
            user_prompt=query, 
            response=result, 
            conversation_histories=conversation_histories,
            user_file_contexts=user_file_contexts, 
            test_output_dir=TEST_OUTPUT_DIR, 
            api_base_url=API_BASE_URL,
            image_id=None, 
            image_hash=None,
            chat_id=chat_id
        )
        
        return JSONResponse(content={"response": result})

    try:
        if user_id in user_file_contexts:
            context = user_file_contexts[user_id]
            if context["context_type"] == "image":
                result = image_handling(context["file_path"], query)
                if "image_id" not in context or "image_hash" not in context:
                    image_id, image_hash = dbf.save_image_to_db(context["file_path"], user_id, IMAGE_JSON, API_BASE_URL)
                    if image_id and image_hash:
                        context["image_id"] = image_id
                        context["image_hash"] = image_hash
                        user_file_contexts[user_id] = context
                else:
                    image_id = context["image_id"]
                    image_hash = context["image_hash"]
            else:
                user_db = FAISS.load_local(
                    folder_path=context["vector_db_path"], 
                    index_name=index_name, 
                    embeddings=embed,
                    allow_dangerous_deserialization=True
                )
                # Use memory-based retrieval instead of conversation history
                result = retrieve_and_answer_with_memory(user_db, query, user_id, session_id, chat_id)
                image_id = None
                image_hash = None
        else:
            main_db = FAISS.load_local(
                folder_path=vector_db_path,
                index_name=index_name,
                embeddings=embed,
                allow_dangerous_deserialization=True
            )
            # Use memory-based retrieval
            result = retrieve_and_answer_with_memory(main_db, query, user_id, session_id, chat_id)
            image_id = None
            image_hash = None

        result = validate_and_improve_response(result, query)
        handle_code_block_extraction(query, result)
        response_data = {
            "response": result
        }
        if is_diagram_request:
            diagram_result = process_and_display_diagram(query, result)
            if isinstance(diagram_result, dict) and "image_data" in diagram_result:
                response_data["image_data"] = diagram_result["image_data"]
                response_data["image_format"] = diagram_result.get("image_format", "png")
                image_id = diagram_result.get("image_id")
                image_hash = diagram_result.get("image_hash")
                result += f"\n\n[Diagram generated and displayed above]"
                response_data["response"] = result

        # Still maintain conversation_history for backward compatibility
        conversation_history.append(f"Q: {query}\nA: {result}")
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]
        conversation_histories[user_id]['history'] = conversation_history

        dbf.save_chat_to_db(
            user_id=user_id, 
            user_prompt=query, 
            response=result, 
            conversation_histories=conversation_histories,
            user_file_contexts=user_file_contexts, 
            test_output_dir=TEST_OUTPUT_DIR, 
            api_base_url=API_BASE_URL,
            image_id=image_id, 
            image_hash=image_hash,
            chat_id=chat_id
        )
        return JSONResponse(content=response_data)
    except Exception as e:
        error_msg = f"Error processing query: {e}"
        print(error_msg)
        return JSONResponse(content={"error": error_msg}, status_code=500)

@app.post("/upload_file")
async def upload_file(
    file: UploadFile = File(...),
    query: str = Form(...),
    user_id: str = Form("default_user"),
    chat_id: Optional[str] = Form(None)
):
    is_diagram_request, modified_query = process_diagram_request(query)
    if is_diagram_request:
        query = modified_query

    file_extension = file.filename.lower().split('.')[-1]
    clean_user_files(user_id)
    if user_id not in conversation_histories:
        conversation_histories[user_id] = {'session_id': str(uuid.uuid4()), 'history': []}
    if isinstance(conversation_histories[user_id], dict):
        conversation_history = conversation_histories[user_id].get('history', [])
        session_id = conversation_histories[user_id].get('session_id')
    else:
        history = conversation_histories[user_id]
        session_id = str(uuid.uuid4())
        conversation_histories[user_id] = {'session_id': session_id, 'history': history}
        conversation_history = history
    
    # If continuing existing chat, get session_id for that chat
    if chat_id:
        existing_session_id = dbf.get_session_id_for_chat(user_id, chat_id, TEST_OUTPUT_DIR)
        if existing_session_id:
            session_id = existing_session_id
            conversation_histories[user_id]['session_id'] = session_id

    try:
        result = None
        error = None
        image_id = None
        image_hash = None

        if file_extension in ['png', 'jpg', 'jpeg']:
            user_dir = get_user_file_directory(user_id)
            image_path = os.path.join(user_dir, file.filename)
            with open(image_path, "wb") as f:
                f.write(await file.read())
            image_id, image_hash = dbf.save_image_to_db(image_path, user_id, IMAGE_JSON, API_BASE_URL)
            user_file_contexts[user_id] = {
                "context_type": "image",
                "file_path": image_path,
                "file_name": file.filename,
                "image_id": image_id,
                "image_hash": image_hash
            }
            general_description = image_handling(image_path)
            if query and query.strip():
                specific_answer = image_handling(image_path, query)
                result = f"{specific_answer}\n\nAdditional image context: {general_description}"
            else:
                result = general_description
        elif file_extension in ['pdf', 'docx', 'doc']:
            user_db = handle_document_upload(file, user_id)
            # Use memory-based retrieval
            result = retrieve_and_answer_with_memory(user_db, query, user_id, session_id, chat_id)
        elif file_extension in ['xls', 'xlsx']:
            user_db, error = handle_excel_upload(file, user_id)
            if error:
                return JSONResponse(content={"error": error}, status_code=400)
            # Use memory-based retrieval
            result = retrieve_and_answer_with_memory(user_db, query, user_id, session_id, chat_id)
        elif file_extension == 'txt':
            user_db = handle_text_upload(file, user_id)
            # Use memory-based retrieval
            result = retrieve_and_answer_with_memory(user_db, query, user_id, session_id, chat_id)
        elif file_extension == 'csv':
            user_db, error = handle_csv_upload(file, user_id)
            if error:
                return JSONResponse(content={"error": error}, status_code=400)
            # Use memory-based retrieval
            result = retrieve_and_answer_with_memory(user_db, query, user_id, session_id, chat_id)
        else:
            return JSONResponse(content={"error": f"Unsupported file type: {file_extension}"}, status_code=400)

        result = validate_and_improve_response(result, query)
        
        # Also update the message history
        message_history = get_or_create_message_history(user_id, session_id)
        message_history.add_user_message(query)
        message_history.add_ai_message(result)
        
        # Keep legacy conversation history format
        if file_extension in ['png', 'jpg', 'jpeg']:
            conversation_history.append(f"Q: {query} [IMAGE:{image_id}]\nA: {result}")
        else:
            conversation_history.append(f"Q: {query}\nA: {result}")
        conversation_histories[user_id]['history'] = conversation_history

        dbf.save_chat_to_db(
            user_id=user_id, 
            user_prompt=query, 
            response=result, 
            conversation_histories=conversation_histories,
            user_file_contexts=user_file_contexts, 
            test_output_dir=TEST_OUTPUT_DIR, 
            api_base_url=API_BASE_URL,
            image_id=image_id, 
            image_hash=image_hash,
            chat_id=chat_id
        )
        handle_code_block_extraction(query, result)
        
        if is_diagram_request:
            diagram_result = process_and_display_diagram(query, result)
            if isinstance(diagram_result, dict) and "image_data" in diagram_result:
                response_data = {
                    "response": result,
                    "image_data": diagram_result["image_data"],
                    "image_format": diagram_result.get("image_format", "png")
                }
                return JSONResponse(content=response_data)
                
        return JSONResponse(content={"response": result})
    except Exception as e:
        return JSONResponse(content={"error": f"Error processing file: {e}"}, status_code=500)

@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    try:
        user_id = request.user_id or "default_user"
        response_text = request.response_text
        feedback_type = request.feedback
        suggestion = request.suggestion
        is_image_response = request.is_image_response

        # Find user prompt from conversation history
        user_prompt = ""
        if user_id in conversation_histories:
            history = conversation_histories[user_id].get('history', []) if isinstance(conversation_histories[user_id], dict) else conversation_histories[user_id]
            for entry in history:
                if isinstance(entry, str) and "Q:" in entry and "A:" in entry:
                    parts = entry.split("A:")
                    if len(parts) > 0:
                        q_part = parts[0].strip()
                        if q_part.startswith("Q:"):
                            user_prompt = q_part[2:].strip()
                            if "[IMAGE:" in user_prompt:
                                pass
                            a_part = "A:".join(parts[1:]).strip()
                            if response_text in a_part:
                                break

        image_id = None
        image_hash = None
        if user_id in user_file_contexts:
            context = user_file_contexts[user_id]
            if context.get("context_type") == "image":
                image_id = context.get("image_id")
                image_hash = context.get("image_hash")
        if not image_id and "[IMAGE:" in user_prompt:
            try:
                image_id = user_prompt.split("[IMAGE:")[1].split("]")[0].strip()
            except:
                pass

        dbf.save_feedback_to_db(
            user_prompt=user_prompt,
            response=response_text,
            feedback=feedback_type,
            feedback_json=FEEDBACK_JSON,
            api_base_url=API_BASE_URL,
            suggestion=suggestion if suggestion else None,
            image_id=image_id,
            image_hash=image_hash
        )
        return JSONResponse(content={"message": "Feedback received."})
    except Exception as e:
        return JSONResponse(content={"error": f"Error processing feedback: {e}"}, status_code=500)

@app.post("/new_chat")
async def new_chat(request: NewChatRequest):
    user_id = request.user_id or "default_user"
    incognito_status = getattr(request, "incognito_status", False)
    
    # Create new session ID
    new_session_id = str(uuid.uuid4())
    
    # Handle based on incognito status
    if incognito_status:
        # In incognito mode, just flush the history but keep incognito on
        flush_incognito_history(user_id)
        print(f"Started new chat in incognito mode for user {user_id}")
        return JSONResponse(content={
            "message": "Incognito chat session restarted successfully.",
            "user_id": user_id,
            "session_id": new_session_id,
            "incognito": True
        })
    else:
        # Regular new chat behavior
        if user_id in conversation_histories:
            conversation_histories[user_id] = {'session_id': new_session_id, 'history': []}
        else:
            conversation_histories[user_id] = {'session_id': new_session_id, 'history': []}
        
        # Clear message history for this user's sessions
        keys_to_clear = [key for key in user_message_histories.keys() if key.startswith(f"{user_id}_")]
        for key in keys_to_clear:
            user_message_histories[key].clear()
            
        # Also clear any existing message history for this user
        if user_id in user_message_histories:
            user_message_histories[user_id].clear()
        
        # Clean up user files and reset context in background
        clean_user_files(user_id)
        
        return JSONResponse(content={
            "message": "Chat session restarted successfully.",
            "user_id": user_id,
            "session_id": new_session_id,
            "incognito": False
        })

@app.post("/incognito_chat")
async def incognito_chat(request: IncognitoChatRequest):
    query = request.query
    user_id = request.user_id or "default_user"
    incognito_status = request.incognito_status
    new_chat = request.new_chat
    
    print(f"Received incognito query: {query!r} with status: {incognito_status}, new_chat: {new_chat}")

    try:
        # If incognito is turned off, flush the incognito history
        if not incognito_status:
            flush_incognito_history(user_id)
            return JSONResponse(content={
                "response": "Incognito session ended. Chat history has been cleared.", 
                "incognito": False
            })

        # If this is a new chat but still in incognito mode, clear the history but keep incognito on
        if new_chat:
            flush_incognito_history(user_id)
            print(f"Started new chat in incognito mode for user {user_id}")
            
        # Get or create the incognito message history
        message_history = get_or_create_incognito_history(user_id)
        
        # Use appropriate model selection based on the query
        model, is_code_query = get_appropriate_model(query)
        
        # Create a context-aware prompt (if not a new chat and we have history)
        has_previous_context = len(message_history.messages) > 0
        
        if has_previous_context:
            # Format conversation history
            messages = message_history.messages[-10:] if len(message_history.messages) > 10 else message_history.messages
            
            conversation_context = "\n\nPrevious conversation (Incognito Mode):\n"
            for i in range(0, len(messages), 2):
                if i + 1 < len(messages):
                    human_msg = messages[i].content if hasattr(messages[i], 'content') else str(messages[i])
                    ai_msg = messages[i + 1].content if hasattr(messages[i + 1], 'content') else str(messages[i + 1])
                    conversation_context += f"Human: {human_msg}\nAssistant: {ai_msg}\n\n"
            
            # Use incognito prompt template with conversation context
            incognito_prompt = config.get("incognito_prompt", 
                "You are in incognito mode. Answer the following user query using only your general knowledge. "
                "Do not reference any documents or external context.")
            full_prompt = f"{incognito_prompt}{conversation_context}Current question: {query}"
        else:
            # First message in incognito session or new chat
            incognito_prompt = config.get("incognito_prompt", 
                "You are in incognito mode. Answer the following user query using only your general knowledge. "
                "Do not reference any documents or external context.")
            full_prompt = f"{incognito_prompt}\n\nUser query: {query}"
        
        print("About to invoke LLM with context-aware incognito prompt...")
        response = model.invoke(full_prompt)
        
        # Remove think blocks if code model was used
        if is_code_query:
            response = remove_think_blocks(str(response))
        
        result = str(response).strip()
        
        # Update incognito message history
        message_history.add_user_message(query)
        message_history.add_ai_message(result)
        
        # Limit history size
        if len(message_history.messages) > MAX_MEMORY_MESSAGES * 2:
            message_history.messages = message_history.messages[-MAX_MEMORY_MESSAGES * 2:]
        
        print("LLM response received with incognito context maintained.")
        return JSONResponse(content={"response": result, "incognito": True})

    except Exception as e:
        error_msg = f"Error in incognito chat: {e}"
        print(error_msg)
        return JSONResponse(content={"error": error_msg, "incognito": True}, status_code=500)

@app.post("/incognito_doc")
async def incognito_doc(request: Request):
    try:
        form = await request.form()
        file = form.get("file")
        query = form.get("query", "")
        user_id = form.get("user_id", "default_user")
        incognito_status = form.get("incognito_status", "true").lower() == "true"
        new_chat = form.get("new_chat", "false").lower() == "true"
        
        print(f"Received incognito doc query: {query!r} with status: {incognito_status}, new_chat: {new_chat}")
        
        # If incognito is turned off, flush the history
        if not incognito_status:
            flush_incognito_history(user_id)
            return JSONResponse(content={
                "response": "Incognito session ended. Document context has been cleared.", 
                "incognito": False
            })
                   
        # If this is a new chat but still in incognito mode, clear the history
        if new_chat:
            flush_incognito_history(user_id)
            print(f"Started new chat in incognito mode for user {user_id}")
        
        # Save to temp_dir
        temp_path = os.path.join(temp_dir.name, file.filename)
        with open(temp_path, "wb") as f:
            contents = await file.read()
            f.write(contents)
        
        # Get incognito message history
        message_history = get_or_create_incognito_history(user_id)
        
        # Process document as usual
        loader = get_loader(temp_path, file.filename)
        if loader:
            documents = loader.load()
            # Use a temporary vector DB in temp_dir
            temp_vector_db_path = os.path.join(temp_dir.name, "vector_db")
            os.makedirs(temp_vector_db_path, exist_ok=True)
            chunked_docs = split_document(documents)
            temp_db = FAISS.from_documents(chunked_docs, embed)
            temp_db.save_local(temp_vector_db_path, index_name)
            
            # Use appropriate model selection
            model, is_code_query = get_appropriate_model(query)
            
            # Set up retrievers
            retriever = temp_db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 8, "score_threshold": 0.3}
            )
            
            # Get documents
            docs = retriever.invoke(query)
            
            # Extract context from documents
            context_parts = []
            sources = set()
            
            for doc in docs:
                context_parts.append(doc.page_content)
                if "source" in doc.metadata:
                    sources.add(os.path.basename(doc.metadata["source"]))
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Create contextual prompt with incognito conversation history
            has_previous_context = len(message_history.messages) > 0
            
            if has_previous_context:
                messages = message_history.messages[-10:] if len(message_history.messages) > 10 else message_history.messages
                conversation_context = "\n\nPrevious conversation (Incognito Mode):\n"
                for i in range(0, len(messages), 2):
                    if i + 1 < len(messages):
                        human_msg = messages[i].content if hasattr(messages[i], 'content') else str(messages[i])
                        ai_msg = messages[i + 1].content if hasattr(messages[i + 1], 'content') else str(messages[i + 1])
                        conversation_context += f"Human: {human_msg}\nAssistant: {ai_msg}\n\n"
                
                full_prompt = f"""You are a helpful assistant answering in incognito mode. Context from documents:{context}{conversation_context}Current question: {query} Please provide a comprehensive answer based on the context above and our conversation history."""
            else:
                full_prompt = f"""You are a helpful assistant answering in incognito mode. Context from documents:{context} Question: {query} Please provide a comprehensive answer based on the context above."""
            
            # Get response from model
            response = model.invoke(full_prompt)
            result = str(response)
            
            # Remove think blocks if present
            if is_code_query:
                result = remove_think_blocks(result)
            
            # Update incognito message history
            message_history.add_user_message(query)
            message_history.add_ai_message(result)
            
            # Limit history size
            if len(message_history.messages) > MAX_MEMORY_MESSAGES * 2:
                message_history.messages = message_history.messages[-MAX_MEMORY_MESSAGES * 2:]
            
            # Add source citations
            if sources and "Sources:" not in result:
                result += f"\n\nSources: {', '.join(sources)}"
            
            # Optionally delete temp files after use
            shutil.rmtree(temp_vector_db_path, ignore_errors=True)
            os.remove(temp_path)
            
            return JSONResponse(content={"response": result, "incognito": True})
        else:
            return JSONResponse(content={"error": "Unsupported file type", "incognito": True}, status_code=400)
    except Exception as e:
        error_msg = f"Error processing file in incognito mode: {e}"
        print(error_msg)
        return JSONResponse(content={"error": error_msg, "incognito": True}, status_code=500)

# Initialize the main vector database by loading the existing one
def load_main_vector_db():
    try:
        vector_db = FAISS.load_local(
            folder_path=vector_db_path,
            index_name=index_name,
            embeddings=embed,
            allow_dangerous_deserialization=True
        )
        print(f"✅ Vector database loaded from {vector_db_path}")
        return vector_db
    except Exception as e:
        print(f"❌ Error loading vector database: {e}")
        print("Please run vector.py first to create the vector database.")
        return None

# Load the vector database at startup
main_faiss_db = load_main_vector_db()

if __name__ == "__main__":
    import uvicorn
    
    # Check if the vector database was loaded successfully
    if main_faiss_db is None:
        print("Exiting due to vector database loading failure.")
        import sys
        sys.exit(1)
    
    # Get the Ethernet IPv4 address using the integrated function
    ip_address = get_ethernet_ipv4()
    if not ip_address:
        # Fallback to the server_ip from config if no Ethernet IPv4 address is found
        ip_address = server_ip
        print(f"Ethernet IPv4 address not found. Using fallback IP address from config: {ip_address}")
    else:
        print(f"Using detected Ethernet IPv4 address: {ip_address}")
    
    uvicorn.run(app, host=ip_address, port=5000)
