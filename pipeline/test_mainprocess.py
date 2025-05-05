import asyncio
from main_process import DocumentProcessor
from benchmark.models.qwen import QwenModel

async def test_processing():
    # Initialize Qwen model
    model = QwenModel()
    
    # Initialize processor
    processor = DocumentProcessor(model)
    
    # Test with a sample document
    document_path = "documents/sample_document.jpg"  # Replace with actual document path
    
    try:
        result = await processor.process_document(document_path)
        print("\nProcessing Results:")
        print(f"Document Type: {result['document_type']}")
        print(f"Language: {result['language']}")
        print(f"Confidence Score: {result['confidence_score']}")
        print("\nExtracted Fields:")
        for field, value in result['fields'].items():
            print(f"{field}: {value}")
    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_processing()) 