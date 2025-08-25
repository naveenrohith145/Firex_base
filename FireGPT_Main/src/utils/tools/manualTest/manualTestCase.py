from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
import os
import json
import datetime

# Define the path for storing test cases
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
data_dir = os.path.join(base_dir, "generatedData", "manualtestcasesJSON")
# Create directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)
TEST_CASES_JSON = os.path.join(data_dir, "manual_test_cases.json")

def save_to_json(user_query, generated_test_cases):
    """
    Save the user query and generated test cases to a JSON file
    
    Args:
        user_query (str): The user's query about test case generation
        generated_test_cases (str): The generated test cases
    """
    try:
        # Create simple entry with just user query and test cases
        entry = {
            "user_query": user_query,
            "manual_test_case": generated_test_cases
        }
        
        # Load existing data or create new structure
        if os.path.exists(TEST_CASES_JSON):
            with open(TEST_CASES_JSON, 'r') as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []
        
        # Append new entry
        data.append(entry)
        
        # Write back to file
        with open(TEST_CASES_JSON, 'w') as file:
            json.dump(data, file, indent=4)
            
        print(f"✅ Test case saved to JSON file: {TEST_CASES_JSON}")
    except Exception as e:
        print(f"❌ Error saving test case to JSON: {e}")

def generate_manual_test_cases(user_query, vector_db_path, llm_ip="172.17.16.39", additional_context=None):
    """
    Generates manual test cases based on user query, context from vector database, and additional context
    
    Args:
        user_query (str): The user's query about test case generation
        vector_db_path (str): Path to the vector database for retrieving context
        llm_ip (str): IP address of the LLM server
        additional_context (str, optional): Additional context information to consider
    
    Returns:
        str: Generated test cases
    """
    print(f"\n🧪 MANUAL TEST CASE GENERATOR STARTED 🧪")
    print(f"User Query: {user_query}")
    print(f"Vector DB Path: {vector_db_path}")
    if additional_context:
        print(f"Additional Context: {len(additional_context)} characters")
    
    try:
        # Initialize LLM
        llm = OllamaLLM(
            base_url=f"http://{llm_ip}:11434",
            model="mistral-nemo"
        )
        
        # Initialize embeddings
        embed = OllamaEmbeddings(
            model="mxbai-embed-large",
            base_url=f"http://{llm_ip}:11434"
        )
        
        print(f"🔍 Loading vector database and retrieving context...")
        
        # Load vector database
        if os.path.exists(vector_db_path):
            vector_db = FAISS.load_local(
                folder_path=vector_db_path,
                index_name="index",
                embeddings=embed,
                allow_dangerous_deserialization=True
            )
            
            # Setup retriever
            retriever = vector_db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}  # Retrieve top 5 most relevant chunks
            )
            
            # Get relevant context
            docs = retriever.invoke(user_query)
            
            if docs:
                print(f"✅ Retrieved {len(docs)} context documents")
                
                # Extract context from documents
                db_context = "\n\n".join([doc.page_content for doc in docs])
                
                # Combine vector DB context with additional context if provided
                if additional_context:
                    combined_context = f"Context from documentation:\n{db_context}\n\nAdditional context:\n{additional_context}"
                else:
                    combined_context = f"Context from documentation:\n{db_context}"
                
                # Create prompt for test case generation
                test_case_prompt = f"""
                You are a quality assurance expert specializing in creating detailed manual test cases.
                
                {combined_context}
                
                Based on the user's request and the context provided, generate comprehensive manual test cases. 
                For each test case, include:
                1. Test Case ID
                2. Test Objective
                3. Preconditions
                4. Test Steps (numbered)
                5. Expected Results
                6. Pass/Fail Criteria
                
                User query: {user_query}
                """
                
                print(f"⚙️ Generating test cases using LLM...")
                # Generate test cases using LLM
                response = llm.invoke(test_case_prompt)
                
                # Format sources if available
                sources = set()
                for doc in docs:
                    if "source" in doc.metadata:
                        sources.add(os.path.basename(doc.metadata["source"]))
                
                result = str(response)
                
                if sources:
                    result += f"\n\nSources: {', '.join(sources)}"
                
                # Save to JSON file with just user query and test cases
                print(f"💾 Saving test cases to JSON storage...")
                save_to_json(user_query, result)
                
                print(f"✅ MANUAL TEST CASE GENERATOR COMPLETED SUCCESSFULLY ✅\n")
                return result
            else:
                print(f"⚠️ No relevant context found in vector database")
                
                # If we have additional context, we can still try to generate test cases
                if additional_context:
                    print(f"📝 Using only additional context to generate test cases...")
                    test_case_prompt = f"""
                    You are a quality assurance expert specializing in creating detailed manual test cases.
                    
                    Context information:
                    {additional_context}
                    
                    Based on the user's request and the context provided, generate comprehensive manual test cases.
                    For each test case, include:
                    1. Test Case ID
                    2. Test Objective
                    3. Preconditions
                    4. Test Steps (numbered)
                    5. Expected Results
                    6. Pass/Fail Criteria
                    
                    User query: {user_query}
                    """
                    
                    response = llm.invoke(test_case_prompt)
                    result = str(response)
                    save_to_json(user_query, result)
                    print(f"✅ MANUAL TEST CASE GENERATOR COMPLETED WITH ONLY ADDITIONAL CONTEXT ✅\n")
                    return result
                else:
                    message = "I couldn't find relevant information in the documentation to generate test cases. Please provide more specific requirements or ensure the documentation contains related information."
                    save_to_json(user_query, message)
                    print(f"❌ MANUAL TEST CASE GENERATOR COMPLETED WITH NO CONTEXT ❌\n")
                    return message
        else:
            error_msg = f"Error: Vector database not found at {vector_db_path}"
            print(f"❌ {error_msg}")
            save_to_json(user_query, error_msg)
            print(f"❌ MANUAL TEST CASE GENERATOR FAILED ❌\n")
            return error_msg
    except Exception as e:
        error_msg = f"Error generating test cases: {str(e)}"
        print(f"❌ {error_msg}")
        save_to_json(user_query, error_msg)
        print(f"❌ MANUAL TEST CASE GENERATOR FAILED ❌\n")
        return error_msg