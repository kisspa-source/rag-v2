import docx
from pptx import Presentation
from openpyxl import Workbook
import os

def create_sample_docx(filename="data/samples/sample.docx"):
    doc = docx.Document()
    doc.add_heading('Test Document', 0)
    doc.add_paragraph('This is a test paragraph for Word loader.')
    doc.add_paragraph('Another paragraph with some more text.')
    doc.save(filename)
    print(f"Created {filename}")

def create_sample_pptx(filename="data/samples/sample.pptx"):
    prs = Presentation()
    slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    title.text = "Hello, World!"
    subtitle.text = "This is a test presentation."
    prs.save(filename)
    print(f"Created {filename}")

def create_sample_xlsx(filename="data/samples/sample.xlsx"):
    wb = Workbook()
    ws = wb.active
    ws.title = "TestSheet"
    ws.append(["Name", "Age", "City"])
    ws.append(["Alice", 30, "New York"])
    ws.append(["Bob", 25, "Los Angeles"])
    wb.save(filename)
    print(f"Created {filename}")

if __name__ == "__main__":
    create_sample_docx()
    create_sample_pptx()
    create_sample_xlsx()
