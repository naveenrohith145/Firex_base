from graphviz import Source
import os
import base64

def display_graphviz_graph(dot_code: str, output_path: str = None):
    """
    Takes Graphviz DOT code as input and displays the graph.
    Returns a tuple of (message, base64_image)
    """
    try:
        src = Source(dot_code)
        # Render the graph
        filename = f"{output_path}"
        src.render(filename, cleanup=True, format='png', view=False)
        
        # Read the generated image and convert to base64
        image_path = f"{output_path}.png"
        with open(image_path, 'rb') as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        message = f"Graph image has been successfully generated and saved to {output_path}.png"
        print("Graph has been rendered.")
        return message, base64_image
    except Exception as e:
        print("An error occurred while rendering the graph:", e)
        return f"Error: {str(e)}", None

def _display_mermaid_diagram(mermaid_code: str, output_path: str = "mermaid_output"):
    """
    Takes Mermaid code as input and generates a diagram.
    """
    try:
        # Save the Mermaid code to a temporary file
        temp_file = f"{output_path}.mmd"
        with open(temp_file, "w") as file:
            file.write(mermaid_code)

        # Use the `mmdc` (Mermaid CLI) to generate the diagram
        os.system(f"mmdc -i {temp_file} -o {output_path}.png")
        
        # Read the generated image and convert to base64
        image_path = f"{output_path}.png"
        with open(image_path, 'rb') as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Clean up the temporary file
        os.remove(temp_file)

        return f"Mermaid diagram has been successfully generated and saved to {output_path}.png", base64_image
    except Exception as e:
        return f"Error: {str(e)}", None
    
def display_mermaid_diagram(mermaid_code: str, output_path: str = "mermaid_output"):
    """
    Takes Mermaid code as input and generates an HTML file and a PNG image to render the diagram.
    """
    try:
        # First try to generate a PNG using the _display_mermaid_diagram function
        message, base64_image = _display_mermaid_diagram(mermaid_code, output_path)
        if base64_image:
            return message, base64_image
            
        # If PNG generation fails, fall back to HTML
        # Create an HTML file with embedded Mermaid code
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
            <script>
                mermaid.initialize({{ startOnLoad: true }});
            </script>
        </head>
        <body>
            <div class="mermaid">
                {mermaid_code}
            </div>
        </body>
        </html>
        """
        # Save the HTML content to a file
        html_file = f"{output_path}.html"
        with open(html_file, "w") as file:
            file.write(html_content)

        return f"Mermaid diagram has been successfully generated and saved to {html_file}. Open it in a browser to view the diagram.", None
    except Exception as e:
        return f"Error: {str(e)}", None