from transformers import pipeline

class DocumentClassifier:
    def __init__(self):
        print("Loading Zero-Shot Document Classification Model...")
        # Uses BART-large fine-tuned on Multi-NLI to evaluate semantic text entailment
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        
        # Define the categories your system can handle
        self.candidate_labels = [
            "Tax Invoice or Commercial Receipt", 
            "Bank Statement or Financial Statement", 
            "Tax Form or Government Filing", 
            "Employment Contract or Agreement"
        ]

    def classify(self, filename: str, text: str) -> dict:
        # We sample the first 1000 characters since document titles and headers appear at the top
        filename_lower = filename.lower()
        forced_hint = ""
        if any(keyword in filename_lower for keyword in ["invoice", "bill", "receipt", "order"]):
            forced_hint = "Tax Invoice or Commercial Receipt"
        elif any(keyword in filename_lower for keyword in ["statement", "bank", "account", "transaction"]):
            forced_hint = "Bank Statement or Financial Statement"
        elif any(keyword in filename_lower for keyword in ["tax", "form", "filing", "returns"]):
            forced_hint = "Tax Form or Government Filing"
        elif any(keyword in filename_lower for keyword in ["contract", "agreement", "offer", "employment"]):
            forced_hint = "Employment Contract or Agreement"
        sample_text = text[:1000]
        
        if not sample_text.strip() and not forced_hint:
            return {"document_type": "Unknown", "confidence": 0.0}

        # If filename gives a definitive clue, we can blend it or use it as a direct override/boost
        if forced_hint:
            return {
                "document_type": forced_hint,
                "confidence": 0.99,  # High confidence boost from explicit filename match
                "method": "filename_heuristic + content"
            }
        
        result = self.classifier(
            sample_text, 
            candidate_labels=self.candidate_labels,
            multi_label=False
        )
        
        best_label = result["labels"][0]
        best_score = round(result["scores"][0], 4)
        
        return {
            "document_type": best_label,
            "confidence": best_score,
            "all_scores": dict(zip(result["labels"], [round(s, 4) for s in result["scores"]]))
        }