from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from PIL import Image, ImageDraw, ImageFont
import os
from loaders import DocumentLoader

def create_image_pdf(filename="image_sample.pdf"):
    # 1. Create an image with text
    img = Image.new('RGB', (800, 600), color='white')
    d = ImageDraw.Draw(img)
    
    # Try to load a default font, otherwise use default
    try:
        font = ImageFont.truetype("Arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()
        
    d.text((50, 50), "This is a scanned document test.", fill='black', font=font)
    d.text((50, 100), "OCR should extract this text.", fill='black', font=font)
    
    img_path = "temp_text_image.png"
    img.save(img_path)
    
    # 2. Create PDF and draw image on it (no text layer)
    c = canvas.Canvas(filename, pagesize=letter)
    c.drawImage(img_path, 0, 0, width=600, height=800)
    c.save()
    
    # Cleanup
    if os.path.exists(img_path):
        os.remove(img_path)
        
    print(f"Created image-only PDF: {filename}")

def test_ocr():
    filename = "image_sample.pdf"
    create_image_pdf(filename)
    
    loader = DocumentLoader()
    print("Loading PDF (expecting OCR trigger)...")
    docs = loader.load_file(filename)
    
    if not docs:
        print("❌ Failed to load document")
        return
        
    print(f"Loaded {len(docs)} pages")
    print(f"Metadata: {docs[0].metadata}")
    print(f"Content: {docs[0].page_content.strip()}")
    
    if docs[0].metadata.get('ocr'):
        print("✅ OCR was triggered")
    else:
        print("❌ OCR was NOT triggered")
        
    if "scanned document test" in docs[0].page_content:
        print("✅ Text extraction successful")
    else:
        print("❌ Text extraction failed")

if __name__ == "__main__":
    test_ocr()
