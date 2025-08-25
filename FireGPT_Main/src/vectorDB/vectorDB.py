import os
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader, 
    CSVLoader,
    TextLoader,
    Docx2txtLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema.document import Document
import json
import time
import concurrent.futures
from tqdm import tqdm  # For progress bars (optional, install with pip if needed)
import hashlib

# Load configuration from config\config.json
with open(r"config\config.json", "r") as config_file:
    config = json.load(config_file)

# Load values from config\config.json
current_directory = config["current_directory"]
vector_db_path = config["vector_db_path"]
index_name = config["index_name"]
embedding_model = config["models"]["embedding_model"]
base_url = config["base_url"]

# Path for the document tracking database - now stored in data\vectorDB_documents
document_registry_path = os.path.join(current_directory, r"data\documentRegistry\document_registry.json")

# Ensure directory exists
os.makedirs(os.path.dirname(document_registry_path), exist_ok=True)

# Initialize OllamaEmbeddings
embed = OllamaEmbeddings(
    model=embedding_model,
    base_url=base_url
)

# Basic text extraction from doc files without dependencies
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
        return [Document(page_content=text, metadata={"source": file_path})]
    except Exception as e:
        print(f"Error extracting text from .doc file: {e}")
        return [Document(page_content="Error extracting text from document.", metadata={"source": file_path})]

# Enhanced loader function without external dependencies
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

# Enhanced function to extract maximum content from Excel or CSV files
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
        return [Document(page_content=f"Error processing file: {str(e)}", 
                        metadata={"source": file_path, "error": str(e)})]

# Function to load document registry or create if not exists
def load_document_registry():
    if os.path.exists(document_registry_path):
        try:
            with open(document_registry_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading document registry: {e}. Creating new registry.")
            return {}
    else:
        return {}

# Function to save document registry
def save_document_registry(registry):
    with open(document_registry_path, 'w') as f:
        json.dump(registry, f, indent=2)

# Function to calculate file hash for checking if content has changed
def get_file_hash(file_path):
    try:
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    except Exception as e:
        print(f"Error calculating hash for {file_path}: {e}")
        # Return timestamp as fallback
        return str(os.path.getmtime(file_path))

# Updated function to recursively find documents in a directory and its subdirectories
# and identify which need processing and which have been removed
def parse_documents_in_directory(root_directory, document_registry):
    try:
        all_document_files = []
        new_or_modified_files = []
        existing_rel_paths = set()
        
        for root, dirs, files in os.walk(root_directory):
            for file in files:
                if file.lower().endswith(('.pdf', '.docx', '.xls', '.xlsx', '.csv', '.png', '.jpg', '.jpeg', '.txt')):
                    # Store the full path and relative path for each document
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, root_directory)
                    existing_rel_paths.add(rel_path)  # Track found files
                    
                    # Get file metadata
                    file_hash = get_file_hash(full_path)
                    mod_time = os.path.getmtime(full_path)
                    
                    # Store all document paths
                    all_document_files.append((full_path, rel_path))
                    
                    # Check if file is new or modified
                    if rel_path not in document_registry:
                        new_or_modified_files.append((full_path, rel_path, file_hash, mod_time))
                        print(f"New document found: {rel_path}")
                    elif document_registry[rel_path].get('hash') != file_hash:
                        new_or_modified_files.append((full_path, rel_path, file_hash, mod_time))
                        print(f"Modified document found: {rel_path}")
        
        # Find removed files (in registry but not in file system)
        removed_files = []
        for rel_path in document_registry.keys():
            if rel_path not in existing_rel_paths:
                removed_files.append(rel_path)
                print(f"Removed document detected: {rel_path}")
        
        print(f"Found {len(all_document_files)} total documents in {root_directory}")
        print(f"Found {len(new_or_modified_files)} new or modified documents to process")
        print(f"Found {len(removed_files)} documents that have been removed")
        
        return all_document_files, new_or_modified_files, removed_files
    except Exception as e:
        print(f"Error accessing directory: {e}")
        return [], [], []

# Function to process a single document - for multithreading
def process_single_document(document_info):
    if len(document_info) == 2:  # Old format: (full_path, rel_path)
        full_path, rel_path = document_info
        file_hash = None
        mod_time = None
    else:  # New format: (full_path, rel_path, file_hash, mod_time)
        full_path, rel_path, file_hash, mod_time = document_info
        
    doc = os.path.basename(full_path)
    loader = get_loader(full_path, doc)
    
    if not loader:
        return []
        
    try:
        if doc.lower().endswith(('.xls', '.xlsx', '.csv')):
            documents = load_excel_or_csv(full_path, doc)
        else:
            documents = loader.load()

        if isinstance(documents, list):
            # Add relative path to metadata for better tracking
            for document in documents:
                if hasattr(document, 'metadata'):
                    document.metadata["rel_path"] = rel_path
            return documents
        else:
            if hasattr(documents, 'metadata'):
                documents.metadata["rel_path"] = rel_path
            return [documents]
    except Exception as e:
        print(f"Error loading document {full_path}: {e}")
        return []

# Updated function to upload documents with full paths using multithreading
def upload_documents(document_paths):
    docs = []
    total_files = len(document_paths)
    processed_count = 0
    
    if total_files == 0:
        print("No documents to process.")
        return docs
    
    # Determine optimal number of workers based on CPU cores
    max_workers = min(32, os.cpu_count() * 2)  # Use at most 2x CPU cores, capped at 32
    
    print(f"Processing {total_files} documents using {max_workers} parallel workers...")
    
    # Use ThreadPoolExecutor for I/O-bound tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all processing tasks
        future_to_doc = {executor.submit(process_single_document, doc_path): doc_path for doc_path in document_paths}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_doc):
            doc_path = future_to_doc[future]
            processed_count += 1
            
            if processed_count % 20 == 0 or processed_count == total_files:
                print(f"Progress: {processed_count}/{total_files} documents processed ({processed_count/total_files*100:.1f}%)")
            
            try:
                documents = future.result()
                docs.extend(documents)
            except Exception as e:
                print(f"Error processing {doc_path[0]}: {e}")
    
    print(f"Loaded {len(docs)} documents.")
    return docs

# Improved document splitter with better context preservation
def split_document(documents, chunk_size=1000, chunk_overlap=300):
    # If no documents to split, return empty list
    if not documents:
        print("No documents to split.")
        return []
        
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

# Function to handle database update when documents are removed
def rebuild_database_without_removed_docs(removed_rel_paths, document_registry, folder_path, index_name):
    """Rebuild the vector database excluding removed documents"""
    print(f"Rebuilding vector database to remove {len(removed_rel_paths)} deleted documents...")
    
    # Keep track of documents that should remain in the database
    active_documents = {}
    for rel_path, data in document_registry.items():
        if rel_path not in removed_rel_paths:
            active_documents[rel_path] = data
    
    if not active_documents:
        print("No active documents remain. Creating empty database.")
        # No documents left, create empty database (or you could delete the files)
        if os.path.exists(os.path.join(folder_path, f"{index_name}.faiss")):
            os.remove(os.path.join(folder_path, f"{index_name}.faiss"))
        if os.path.exists(os.path.join(folder_path, f"{index_name}.pkl")):
            os.remove(os.path.join(folder_path, f"{index_name}.pkl"))
        return None
    
    # Process all active documents again to rebuild database
    docs_to_process = [(doc_data["full_path"], rel_path) for rel_path, doc_data in active_documents.items()]
    
    # Process the documents
    documents = upload_documents(docs_to_process)
    chunked_docs = split_document(documents)
    
    # Create brand new database with only active documents
    vector_db = FAISS.from_documents(chunked_docs, embed)
    vector_db.save_local(folder_path=folder_path, index_name=index_name)
    
    print(f"Vector database rebuilt with {len(active_documents)} remaining documents")
    return vector_db

# Updated function to store documents in a vector database
# with support for merging new documents into existing database
def store_vector_db(documents, embeddings, folder_path, index_name, removed_rel_paths=None):
    # If we have removed documents, we need to rebuild the database
    if removed_rel_paths and len(removed_rel_paths) > 0:
        return rebuild_database_without_removed_docs(
            removed_rel_paths,
            load_document_registry(),
            folder_path, 
            index_name
        )
    
    # If no documents to store, return existing DB or None
    if not documents:
        # Try to load existing DB
        if os.path.exists(os.path.join(folder_path, f"{index_name}.faiss")):
            print(f"No new documents to store. Loading existing vector database.")
            try:
                return FAISS.load_local(folder_path, embeddings, index_name, allow_dangerous_deserialization=True)
            except Exception as e:
                print(f"Error loading existing vector database: {e}")
                return None
        else:
            print("No documents to store and no existing database found.")
            return None
    
    # Check if vector database already exists
    if os.path.exists(os.path.join(folder_path, f"{index_name}.faiss")):
        try:
            print(f"Updating existing vector database with new documents.")
            # Load existing vector database with security flag
            vector_db = FAISS.load_local(folder_path, embeddings, index_name, allow_dangerous_deserialization=True)
            
            # Add new documents to the existing database
            vector_db.add_documents(documents)
            
            # Save the updated database
            vector_db.save_local(folder_path=folder_path, index_name=index_name)
            print(f"Vector database updated with {len(documents)} new document chunks.")
            
        except Exception as e:
            print(f"Error updating vector database: {e}. Creating new database instead.")
            vector_db = FAISS.from_documents(documents, embeddings)
            vector_db.save_local(folder_path=folder_path, index_name=index_name)
    else:
        # Create new vector database if it doesn't exist
        print(f"Creating new vector database with {len(documents)} document chunks.")
        vector_db = FAISS.from_documents(documents, embeddings)
        vector_db.save_local(folder_path=folder_path, index_name=index_name)
    
    print(f"Vector database saved to {folder_path} as {index_name}.")
    return vector_db

# Updated function to handle document processing with recursive directory traversal,
# incremental updates, and document removal
def document_handling(directory=None):
    if directory is None:
        directory = current_directory
    
    # Load document registry
    document_registry = load_document_registry()
    
    start_discovery = time.time()
    # Get all document files, identify new/modified ones, and detect removed ones
    all_document_paths, new_document_paths, removed_rel_paths = parse_documents_in_directory(directory, document_registry)
    end_discovery = time.time()
    print(f"Document discovery completed in {end_discovery - start_discovery:.2f} seconds")
    
    # Upload only new or modified documents
    start_processing = time.time()
    new_documents = upload_documents([doc_info[:2] for doc_info in new_document_paths])
    end_processing = time.time()
    print(f"Document processing completed in {end_processing - start_processing:.2f} seconds")
    
    # Process the new documents
    start_splitting = time.time()
    chunked_new_docs = split_document(new_documents)
    end_splitting = time.time()
    print(f"Document splitting completed in {end_splitting - start_splitting:.2f} seconds")
    
    # Update the vector database with new documents and handle removed documents
    start_vectorizing = time.time()
    faiss_db = store_vector_db(chunked_new_docs, embed, vector_db_path, index_name, removed_rel_paths)
    end_vectorizing = time.time()
    print(f"Vector database update completed in {end_vectorizing - start_vectorizing:.2f} seconds")
    
    # Update document registry: add new/modified docs and remove deleted ones
    for full_path, rel_path, file_hash, mod_time in new_document_paths:
        document_registry[rel_path] = {
            "hash": file_hash,
            "last_modified": mod_time,
            "last_processed": time.time(),
            "full_path": full_path
        }
    
    # Remove deleted documents from registry
    for rel_path in removed_rel_paths:
        if rel_path in document_registry:
            del document_registry[rel_path]
            print(f"Removed {rel_path} from document registry")
    
    # Save updated document registry
    save_document_registry(document_registry)
    print(f"Document registry updated: {len(new_document_paths)} new/modified, {len(removed_rel_paths)} removed.")
    
    return faiss_db

# Function to recursively process images in directory and subdirectories
def process_images_in_directory(directory):
    image_count = 0
    for root, _, files in os.walk(directory):
        image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_count += len(image_files)
        if image_files:
            print(f"Found {len(image_files)} image files in {root}")
    print(f"Total images found: {image_count}")
    return image_count

if __name__ == "__main__":
    start_time = time.time()
    print(f"Creating/updating vector database from documents in {current_directory} and its subdirectories...")
    document_handling()
    end_time = time.time()
    print(f"Vector database process completed in {end_time - start_time:.2f} seconds")