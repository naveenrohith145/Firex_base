from langchain_ollama import OllamaLLM
import os
import json
import datetime

# Define the path for storing generated contexts
CONTEXTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "generatedData", "contextjson")
# Create directory if it doesn't exist
os.makedirs(CONTEXTS_DIR, exist_ok=True)
CONTEXTS_JSON = os.path.join(CONTEXTS_DIR, "generated_contexts.json")

def save_context_to_json(query, generated_context):
    """
    Save the user query and generated context to a JSON file
    
    Args:
        query (str): The user's query for context generation
        generated_context (str): The generated context
    """
    try:
        # Create simple entry with query and context
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query,
            "context": generated_context
        }
        
        # Load existing data or create new structure
        if os.path.exists(CONTEXTS_JSON):
            with open(CONTEXTS_JSON, 'r') as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []
        
        # Append new entry
        data.append(entry)
        
        # Write back to file
        with open(CONTEXTS_JSON, 'w') as file:
            json.dump(data, file, indent=4)
            
        print(f"✅ Context saved to JSON file: {CONTEXTS_JSON}")
    except Exception as e:
        print(f"❌ Error saving context to JSON: {e}")

def generate_context(query, llm_ip="172.17.16.39"):
    """
    Generates context information based on a user query
    
    Args:
        query (str): The user's query for which context is needed
        llm_ip (str): IP address of the LLM server
    
    Returns:
        str: Generated context information
    """
    print(f"\n🔍 CONTEXT GENERATION STARTED 🔍")
    print(f"Query: {query}")
    
    try:
        # Initialize LLM
        llm = OllamaLLM(
            base_url=f"http://{llm_ip}:11434",
            model="mistral-nemo"
        )
        
        # Create prompt for context generation
        context_prompt = f"""
        Generate comprehensive context information for creating manual test cases based on the following query:
        
        Query: {query}
        
        In your response, include:
        1. Relevant requirements and specifications
        2. Expected user workflows and interactions
        3. Potential edge cases and boundary conditions
        4. Environment setup considerations
        5. Any important constraints or assumptions
        
        Focus on providing detailed information that would be useful for testing the described functionality.
        """
        
        print(f"⚙️ Generating context using LLM...")
        # Generate context using LLM
        response = llm.invoke(context_prompt)
        context = str(response)
        
        # Save the context to JSON for future reference
        save_context_to_json(query, context)
        
        print(f"✅ CONTEXT GENERATION COMPLETED SUCCESSFULLY ✅\n")
        return context
        
    except Exception as e:
        error_msg = f"Error generating context: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"❌ CONTEXT GENERATION FAILED ❌\n")
        return f"Error: {str(e)}"