# ocr/paddle_ocr.py
from paddleocr import PaddleOCR
import numpy as np
from PIL import Image

class DocumentOCR:
    def __init__(self, lang: str = 'en'):
        # use_angle_cls=True helps if the document is scanned upside down or sideways
        self.ocr_engine = PaddleOCR(use_angle_cls=True, lang=lang)

    def extract_text(self, image: Image.Image) -> list[dict]:
        """
        Takes a PIL image and returns a list of dictionaries containing:
        - text: The extracted string
        - box: Bounding box coordinates [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
        - confidence: OCR confidence score (0 to 1)
        """
        # Convert PIL Image to numpy array (PaddleOCR requires this format)
        img_array = np.array(image.convert('RGB'))
        
        # Run OCR
        result = self.ocr_engine.ocr(img_array)
        
        extracted_data = []
        
        # PaddleOCR returns a nested list. result[0] contains the data for the first (and only) image passed.
        if result and result[0]:
            for line in result[0]:
                box = line[0]            # Coordinates
                text = line[1][0]        # Text string
                confidence = line[1][1]  # Confidence score
                
                extracted_data.append({
                    "text": text,
                    "box": box,
                    "confidence": float(confidence)
                })
                
        return extracted_data