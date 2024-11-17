from PIL import Image
import pytesseract
import os

# Set the path to the Tesseract executable if necessary (especially on Windows)
# Uncomment and set the path if needed
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_image(image_path):
    if not os.path.isfile(image_path):
        return f"File not found: {image_path}"
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        # remove groups of whitespaces and newline characters
        text = " ".join(text.split())
        return text if text else "No text found in the image."
    except pytesseract.pytesseract.TesseractError as e:
        return f"Tesseract OCR error: {e}"
    except Exception as e:
        return f"An error occurred: {e}"

# Test the function with an image file
image_path = "Improvements/multimodality/image_to_text/c.png"
text = extract_text_from_image(image_path)
print("Extracted Text:", text)
