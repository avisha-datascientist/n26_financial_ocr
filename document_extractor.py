import asyncio
from pipeline.main_process import DocumentProcessor
from downstream import downstream
from benchmark.models.qwen import QwenModel
from evaluation import evaluation
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

        evaluation_result = evaluation(document_path, result['fields'], result['document_type'])

        rule_compliance_result = check_rule_compliance(formatted_result)

        # Save results
        output_path = "/Users/avishabhiryani/Documents/private/N26_GenAI_Take_Home_Assignment/results"
        postprocessor.save_results(formatted_result, output_path)
        print(f"\nResults saved to {output_path}")
    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    asyncio.run(document_extractor()) 