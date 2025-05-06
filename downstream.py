import asyncio
from postprocessor import DocumentPostprocessor

def downstream(result):
    # Initialize postprocessor
    postprocessor = DocumentPostprocessor()
    
    try:
        # Postprocess fields
        formatted_fields = postprocessor.postprocess_fields(result["fields"])
        formatted_result["fields"] = formatted_fields
        
        return formatted_result
        
    except Exception as e:
        print(f"Error during postprocessing: {str(e)}")