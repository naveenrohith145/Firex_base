from langchain.tools import BaseTool
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from src.utils.tools.diagram import displayDiagram as displayDiagram
from typing import Optional
import os
from src.utils.tools.manualTest import manualTestCase as manualTestCase
from  src.utils.tools.context import contextTool as contextTool

# Initialize the LLM and embeddings
llm_IP = "172.17.16.39"

embed = OllamaEmbeddings(
    model="mxbai-embed-large",
    base_url=f"http://{llm_IP}:11434"
)

llm = OllamaLLM(
    base_url=f"http://{llm_IP}:11434",
    model="mistral-nemo"
)

# Define output path relative to the current directory

output_path = os.path.join(os.getcwd(), "data", "images")
print(f"Output path for images: {output_path}")
print("=="*50)
# Ensure the directory exists
os.makedirs(output_path, exist_ok=True)


class PlannerTool(BaseTool):

    name: str = "Planner"
    description: str = "Creates the plan on how to respond with user query and determines tools/agents required"

    def _run(self, query: str) -> str:
        try:
            return llm.invoke(f"Create a step-by-step plan for the following query and specify the tools/agents required: {query}")
        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(self, query: str) -> str:
        return self._run(query)

class GraphvizTool(BaseTool):

    name: str = "GraphvizTool"
    description: str = "Generates a graph image from Graphviz DOT code."

    def _run(self, dot_code: str, output_path: Optional[str] = "graph_output") -> str:
        try:
            message, base64_image = displayDiagram.display_graphviz_graph(dot_code, output_path)
            
            if base64_image:
                return {
                    "message": message,
                    "image_data": base64_image,
                    "image_path": f"{output_path}.png",
                    "format": "png"
                }
            else:
                return message
        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(self, dot_code: str, output_path: Optional[str] = "graph_output") -> str:
        return self._run(dot_code, output_path)

class MermaidTool(BaseTool):

    name: str = "MermaidTool"
    description: str = "Generates a diagram from Mermaid code."

    def _run(self, mermaid_code: str, output_path: Optional[str] = "mermaid_output") -> str:
        
        try:
            message, base64_image = displayDiagram.display_mermaid_diagram(mermaid_code, output_path)
            if base64_image:
                return {
                    "message": message,
                    "image_data": base64_image,
                    "image_path": f"{output_path}.png",
                    "format": "png"
                }
            else:
                return message
        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(self, mermaid_code: str, output_path: Optional[str] = "mermaid_output") -> str:
        return self._run(mermaid_code, output_path)

class ProtocolVerificationTool(BaseTool):

    name: str = "ProtocolVerificationTool"
    description: str = "Verifies if the user query adheres to the protocols and policies specified in protocol.txt."

    def _run(self, query: str, protocol_file: str = "protocol.txt") -> str:
        
        try:
            if not os.path.exists(protocol_file):
                return f"Error: Protocol file '{protocol_file}' not found."

            with open(protocol_file, "r") as file:
                protocols = file.read()

            prompt = (
                f"Given the following protocols and policies:\n\n{protocols}\n\n"
                f"Does the following query adhere to these protocols? Respond only with 'Yes' or 'No'\n\n"
                f"Query: {query}"
            )
            response = llm.invoke(prompt)
            return response
        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(self, query: str, protocol_file: str = "protocol.txt") -> str:
        return self._run(query, protocol_file)

class ManualTestCaseTool(BaseTool):

    name: str = "ManualTestCaseTool"
    description: str = "Generates manual test cases based on user requirements and documentation."

    def _run(self, query: str, vector_db_path: Optional[str] = r"vectorDatabase", additional_context: Optional[str] = None) -> str:
        try:
            print("\n=== CALLING MANUAL TEST CASE TOOL ===")
            print(f"Query: {query}")
            print(f"Vector DB path: {vector_db_path}")
            
            if additional_context:
                print(f"Additional context provided: {len(additional_context)} characters")
            
            result = manualTestCase.generate_manual_test_cases(query, vector_db_path, llm_IP, additional_context)
            
            print("=== MANUAL TEST CASE TOOL EXECUTION COMPLETED ===\n")
            return result
        except Exception as e:
            error_msg = f"Error in ManualTestCaseTool: {str(e)}"
            print(f"❌ {error_msg}")
            print("=== MANUAL TEST CASE TOOL EXECUTION FAILED ===\n")
            return f"Error: {str(e)}"

    async def _arun(self, query: str, vector_db_path: Optional[str] = r"vectorDatabase", additional_context: Optional[str] = None) -> str:
        return self._run(query, vector_db_path, additional_context)

class ContextTool(BaseTool):

    name: str = "ContextTool"
    description: str = "Generates context information for test case generation."

    def _run(self, query: str, llm_ip: Optional[str] = "172.17.16.39") -> str:
        try:
            print("\n=== CALLING CONTEXT TOOL ===")
            print(f"Query: {query}")
            
            context = contextTool.generate_context(query, llm_ip)
            
            print("=== CONTEXT TOOL EXECUTION COMPLETED ===\n")
            return context
        except Exception as e:
            error_msg = f"Error in ContextTool: {str(e)}"
            print(f"❌ {error_msg}")
            print("=== CONTEXT TOOL EXECUTION FAILED ===\n")
            return f"Error: {str(e)}"

    async def _arun(self, query: str, llm_ip: Optional[str] = "172.17.16.39") -> str:
        return self._run(query, llm_ip)