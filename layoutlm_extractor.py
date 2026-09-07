from transformers import pipeline

class LayoutLMExtractor:
    def __init__(self):
        print("Loading LayoutLM Multi-Modal AI...")
        self.qa_model = pipeline(
            "document-question-answering", 
            model="impira/layoutlm-document-qa"
        )
        
        self.queries = {
            "invoice_number": "What is the invoice number?",
            "date": "What is the date?",
            "total_amount": "What is the total amount?"
        }

    def convert_paddle_to_layoutlm(self, ocr_data: list, image_width: int, image_height: int) -> list:
        word_boxes = []
        
        for block in ocr_data:
            text = block["text"]
            box = block["box"]
            
            xmin = int(min(point[0] for point in box))
            ymin = int(min(point[1] for point in box))
            xmax = int(max(point[0] for point in box))
            ymax = int(max(point[1] for point in box))
            
            # NORMALIZE TO 0-1000 SCALE (Required by LayoutLM)
            xmin = min(max(int((xmin / image_width) * 1000), 0), 1000)
            ymin = min(max(int((ymin / image_height) * 1000), 0), 1000)
            xmax = min(max(int((xmax / image_width) * 1000), 0), 1000)
            ymax = min(max(int((ymax / image_height) * 1000), 0), 1000)
            
            word_boxes.append([text, [xmin, ymin, xmax, ymax]])
            
        return word_boxes

    def extract_fields(self, image, ocr_data: list) -> dict:
        results = {}
        
        # Get the actual width and height of the PIL Image
        img_width, img_height = image.size
        
        # Convert and normalize the boxes
        word_boxes = self.convert_paddle_to_layoutlm(ocr_data, img_width, img_height)
        
        for field, question in self.queries.items():
            response = self.qa_model(
                image=image,
                word_boxes=word_boxes,
                question=question
            )
            
            if isinstance(response, list):
                response = response[0] if response else {}
                
            results[field] = {
                "value": response.get("answer", "Not found"),
                "confidence": round(response.get("score", 0.0), 4)
            }
            
        return results