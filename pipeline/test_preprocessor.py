import asyncio
from preprocessing import DocumentPreprocessor
from benchmark.models.qwen import QwenModel

async def test_preprocessing():
    # Initialize Qwen model
    model = QwenModel()
    
    # Initialize preprocessor
    preprocessor = DocumentPreprocessor(model)
    
    # Test with a sample document
    document_path = "documents/sample_document.jpg"  # Replace with actual document path
    
    try:
        result = await preprocessor.preprocess_document(document_path)
        print("\nPreprocessing Results:")
        print(f"Document Type: {result['document_type']}")
        print(f"Detected Language: {result['detected_language']}")
        print("\nExtracted Text (Markdown):")
        print(result['extracted_text'])
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_preprocessing()) 