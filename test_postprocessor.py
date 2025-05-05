import asyncio
from postprocessor import DocumentPostprocessor

async def test_postprocessing():
    # Initialize postprocessor
    postprocessor = DocumentPostprocessor()
    
    # Sample results for testing
    sample_results = {
        "document_type": "invoice",
        "language": "en",
        "confidence_score": 0.95,
        "fields": {
            "invoice_date": "2023-12-31",
            "due_date": "31/12/2023",
            "amount": "€1,234.56",
            "customer_name": "John Doe",
            "customer_address": "123 Main St, City, Country",
            "invoice_number": "INV-2023-001",
            "additional_info": "Payment terms: 30 days"
        }
    }
    
    try:
        # Postprocess fields
        formatted_fields = postprocessor.postprocess_fields(
            sample_results["fields"],
            sample_results["language"]
        )
        sample_results["fields"] = formatted_fields
        
        # Present results
        print("\nFormatted Results:")
        print(postprocessor.present_results(sample_results))
        
        # Save results
        output_path = "results/sample_results.txt"
        postprocessor.save_results(sample_results, output_path)
        print(f"\nResults saved to {output_path}")
        
    except Exception as e:
        print(f"Error during postprocessing: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_postprocessing()) 