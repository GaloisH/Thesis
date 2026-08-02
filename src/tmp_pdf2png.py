"""Temporary: merge 3-page PDF into single PNG."""
import fitz  # PyMuPDF
import numpy as np
from PIL import Image

pdf_path = r"C:\Users\13765\Downloads\IPA_胡锦添.pdf"
output_path = r"D:\python_code\projects\thesis\src\merged_output.png"

doc = fitz.open(pdf_path)
images = []
for page in doc:
    pix = page.get_pixmap(dpi=200)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    images.append(np.array(img))
doc.close()

# Stack 3 pages vertically
merged = np.vstack(images)
Image.fromarray(merged).save(output_path)
print(f"Saved: {output_path}")
