import pymupdf 
from PIL import Image
import io

def pdf_to_images(file_path: str, dpi: int = 300) -> list[Image.Image]:
    """
    Converts a PDF file into a list of PIL Images.
    Uses 300 DPI for high-quality OCR results.
    """
    try:
        doc = pymupdf.open(file_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Matrix to scale up the resolution based on DPI
            zoom = dpi / 72.0 
            mat = pymupdf.Matrix(zoom, zoom)
            
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            image = Image.open(io.BytesIO(img_data))
            images.append(image)
            
        return images
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return []