import openai
import json

def evaluation(document_path, extracted_fields, document_type):

    with open(document_path, "rb") as f:
        file_bytes = f.read()

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a document evaluation assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": "Here is a document and some extracted fields. Rate completeness and accuracy:\n"
                                        "Extracted fields: {extracted_fields} \n"
                                        "Category: {document_type}"},
                {"type": "file", "file": {"name": document_path, "data": file_bytes}}
            ]}
        ],
        temperature=0.2
    )

    return response.choices[0].message["content"]

def check_rule_compliance(formatted_result: dict):
    prompt = f"""
        You are a data format compliance checker.

        Below are extracted fields from a document. Your task is to validate if the values comply with formatting rules.

        ### Extracted Fields (JSON):
        {json.dumps(formatted_result, indent=2)}

        ### Formatting Rules:
        1. Dates must follow the format: "dd.mm.yyyy" (e.g., 05.05.2024)
        2. Names must be in the format: "first_name last_name" (e.g., John Doe)
        3. Addresses must be: "street_name street_number, city zipcode, country" (e.g., Main St 42, Zurich 8000, Switzerland)
        4. Monetary amounts must be in the format: "amount currency_symbol", using a full stop as decimal separator (e.g., 1234.56 $)
        5. All other fields must match the original document exactly (assume they were correct if format not otherwise specified)

        ### Your Task:
        - Check each field’s value for rule compliance.
        - Score overall compliance out of 100.
        - Report any fields that violate the rules and why.

        Respond in the following JSON format:
        {{
        "compliance_score": <int>,
        "violations": [
            {{
            "field": "<field_name>",
            "reason": "<what is wrong>"
            }}
        ]
        }}
        """

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return json.loads(response.choices[0].message["content"])


def final_evaluation(document_path, extracted_fields, document_type, formatted_result):
    evaluation_result = evaluation(document_path, extracted_fields, document_type)
    rule_compliance_result = check_rule_compliance(formatted_result)

    return evaluation_result, rule_compliance_result