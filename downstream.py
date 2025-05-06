import asyncio
from pipeline.postprocessor import DocumentPostprocessor

def downstream(result):
    # Initialize postprocessor
    postprocessor = DocumentPostprocessor()
    
    try:
        # Postprocess fields
        formatted_fields = postprocessor.postprocess_fields(result)
        formatted_fields["fields"] = formatted_fields
        
        return formatted_fields
        
    except Exception as e:
        print(f"Error during postprocessing: {str(e)}")