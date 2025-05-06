import asyncio
from pipeline.main_process import DocumentProcessor
from downstream import downstream
from benchmark.models.qwen import QwenModel
from evaluation import final_evaluation
async def document_extractor():
    # Initialize Qwen model
    model = QwenModel()
    
    # Initialize processor
    processor = DocumentProcessor(model)
    
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

        formatted_result = downstream(result)

        print(formatted_result)

        evaluation_result, rule_compliance_result = final_evaluation(document_path, result['fields'], result['document_type'], formatted_result)

        print(evaluation_result)
        print(rule_compliance_result)

    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    asyncio.run(document_extractor()) 