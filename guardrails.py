import re
from datetime import datetime

class DocumentValidator:
    def __init__(self, confidence_threshold: float = 0.50):
        self.confidence_threshold = confidence_threshold

    def validate_date(self, date_str: str) -> bool:
        # Common date format validation patterns
        date_formats = ["%d-%b-%Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d %B %Y"]
        for fmt in date_formats:
            try:
                datetime.strptime(date_str.strip(), fmt)
                return True
            except ValueError:
                continue
        return False

    def validate_amount(self, amount_str: str) -> bool:
        # Check if string contains a valid numeric float/integer pattern
        cleaned = re.sub(r'[^\d.]', '', amount_str)
        try:
            float(cleaned)
            return len(cleaned) > 0
        except ValueError:
            return False

    def validate_extraction(self, extracted_data: dict) -> dict:
        validated_results = {}
        flagged_for_review = False
        review_reasons = []

        for field, details in extracted_data.items():
            value = details.get("value", "Not found")
            confidence = details.get("confidence", 0.0)

            status = "PASSED"
            note = ""

            # Rule 1: Check model confidence threshold
            if confidence < self.confidence_threshold or value == "Not found":
                status = "FLAGGED"
                note = "Low confidence or missing value."
                flagged_for_review = True

            # Rule 2: Field-specific semantic regex validation
            elif field == "date" and not self.validate_date(value):
                status = "FLAGGED"
                note = "Failed date format validation rule."
                flagged_for_review = True
                review_reasons.append(f"Invalid date format: {value}")

            elif field == "total_amount" and not self.validate_amount(value):
                status = "FLAGGED"
                note = "Failed numeric amount validation rule."
                flagged_for_review = True
                review_reasons.append(f"Invalid monetary amount: {value}")

            validated_results[field] = {
                "value": value,
                "confidence": confidence,
                "status": status,
                "validation_note": note
            }

        return {
            "fields": validated_results,
            "requires_human_review": flagged_for_review,
            "review_reasons": review_reasons
        }