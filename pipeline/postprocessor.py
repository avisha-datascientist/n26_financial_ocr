from typing import Dict, Any, List
import json
from datetime import datetime
import re
from pathlib import Path

class DocumentPostprocessor:
    def __init__(self):
        pass
        
        
    def _format_date(self, date_str: str) -> str:
        """Format date string to dd.mm.yyyy format.
        
        Args:
            date_str: Date string to format
            
        Returns:
            Formatted date string in dd.mm.yyyy format
        """
        if not date_str:
            return date_str
            
        # Common date formats to try
        date_formats = [
            '%Y-%m-%d',    # 2023-12-31
            '%d/%m/%Y',    # 31/12/2023
            '%m/%d/%Y',    # 12/31/2023
            '%d.%m.%Y',    # 31.12.2023
            '%Y.%m.%d',    # 2023.12.31
            '%d %B %Y',    # 31 December 2023
            '%B %d, %Y',   # December 31, 2023
        ]
        
        # Try to parse the date using various formats
        for fmt in date_formats:
            try:
                date = datetime.strptime(date_str, fmt)
                # Convert to dd.mm.yyyy format
                return date.strftime('%d.%m.%Y')
            except ValueError:
                continue
        return date_str  # Return original if parsing fails
        
    def _format_amount(self, amount_str: str) -> str:
        """Format monetary amounts to 'amount currency_symbol' format.
        
        Args:
            amount_str: String containing amount and possibly currency symbol
            
        Returns:
            Formatted amount string
        """
        try:
            # Remove any existing currency symbols and whitespace
            amount_str = amount_str.strip()
            
            # Extract currency symbol if present
            currency_symbol = None
            for symbol in ['€', '$', '£', 'CHF']:
                if symbol in amount_str:
                    currency_symbol = symbol
                    amount_str = amount_str.replace(symbol, '').strip()
                    break
            
            # If no currency symbol found, default to €
            if not currency_symbol:
                currency_symbol = '€'
            
            # Remove any thousand separators and replace decimal separator with dot
            amount_str = amount_str.replace('.', '').replace(',', '.')
            
            # Convert to float and back to string to normalize
            amount = float(amount_str)
            
            # Format as "amount currency_symbol"
            return f"{amount} {currency_symbol}"
            
        except Exception as e:
            print(f"Error formatting amount {amount_str}: {str(e)}")
            return amount_str  # Return original string if formatting fails
        
    def _format_name(self, name_str: str) -> str:
        """Format name string preserving original capitalization.
        
        Args:
            name_str: Name string to format
            
        Returns:
            Formatted name string
        """
        if not name_str:
            return name_str
        return name_str.strip()
        
    def _format_address(self, address_str: str) -> str:
        """Format address string preserving original formatting.
        
        Args:
            address_str: Address string to format
            
        Returns:
            Formatted address string
        """
        if not address_str:
            return address_str
        return address_str.strip()
        
    def postprocess_fields(self, fields: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Postprocess extracted fields with proper formatting.
        
        Args:
            fields: Dictionary of extracted fields
            language: Language code for formatting rules
            
        Returns:
            Dictionary of formatted fields
        """
        formatted_fields = {}
        
        for field, value in fields.items():
            if value is None:
                formatted_fields[field] = None
                continue
                
            # Apply appropriate formatting based on field type
            if any(keyword in field.lower() for keyword in ['document_date', 'statement_period', 'effective_date', 'payment_due_date']):
                formatted_fields[field] = self._format_date(value, language)
            elif any(keyword in field.lower() for keyword in ['portfolio_value', 'opening_balance', 'closing_balance', 'available_balance', 'garnishment_amount', 'credit_limit', 'minimum_payment', 'previous_balance', 'new_balance']):
                formatted_fields[field] = self._format_amount(value)
            elif any(keyword in field.lower() for keyword in ['customer_name', 'debtor_name', 'creditor_name']):
                formatted_fields[field] = self._format_name(value)
            elif any(keyword in field.lower() for keyword in ['institution_address', 'adresse', 'dirección', 'indirizzo']):
                formatted_fields[field] = self._format_address(value)
            else:
                formatted_fields[field] = value
                
        return formatted_fields
        
    def present_results(self, results: Dict[str, Any]) -> str:
        """Present processing results in a readable format.
        
        Args:
            results: Dictionary containing processing results
            
        Returns:
            Formatted string presentation of results
        """
        output = []
        output.append("=" * 50)
        output.append("Document Processing Results")
        output.append("=" * 50)
        output.append(f"Document Type: {results['document_type']}")
        output.append(f"Language: {results['language']}")
        output.append(f"Confidence Score: {results['confidence_score']:.2f}")
        output.append("\nExtracted Fields:")
        output.append("-" * 50)
        
        for field, value in results['fields'].items():
            if value is None:
                output.append(f"{field}: Not found")
            else:
                output.append(f"{field}: {value}")
                
        output.append("=" * 50)
        return "\n".join(output)
        
    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Save processing results to a file.
        
        Args:
            results: Dictionary containing processing results
            output_path: Path to save the results
        """
        # Save formatted presentation
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.present_results(results))
            
        # Save raw JSON data
        json_path = Path(output_path).with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False) 