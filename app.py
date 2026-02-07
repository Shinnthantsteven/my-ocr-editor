import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR
from pdf2image import convert_from_bytes
import io

# Initialize OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

st.title("🤖 Automatic Certificate Fixer")
st.write("Upload your document and it will automatically find the number to fix.")

uploaded_file = st.file_uploader("Upload Image or PDF", type=['png', 'jpg', 'jpeg', 'pdf'])

if uploaded_file:
    # 1. Convert PDF/Image to CV2 format
    if uploaded_file.type == "application/pdf":
        pages = convert_from_bytes(uploaded_file.read())
        img_pil = pages[0].convert("RGB")
    else:
        img_pil = Image.open(uploaded_file).convert("RGB")
    
    cv_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    # 2. Input for new digits
    target_text = st.text_input("Which old number should I find? (e.g., 00253)")
    new_text = st.text_input("What is the NEW number?")

    if st.button("Fix Automatically") and target_text and new_text:
        with st.spinner("Finding and healing background..."):
            # 3. Run OCR to find coordinates
            results = ocr.ocr(cv_img, cls=True)
            
            mask = np.zeros(cv_img.shape[:2], np.uint8)
            found = False
            coords = None

            for line in results[0]:
                detected_text = line[1][0]
                if target_text in detected_text:
                    found = True
                    # Get box coordinates
                    box = np.array(line[0]).astype(np.int32)
                    cv2.fillPoly(mask, [box], 255)
                    coords = box[0] # Top-left corner
            
            if found:
                # 4. Seamless Background Healing (Inpainting)
                # This removes the old text and recreates the paper texture
                cleaned_cv = cv2.inpaint(cv_img, mask, 3, cv2.INPAINT_TELEA)
                
                # Convert back to PIL for high-quality text rendering
                final_img = Image.fromarray(cv2.cvtColor(cleaned_cv, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(final_img)
                
                # 5. Add New Text (Using a bold font logic)
                # We place it exactly where the old text was
                f_size = int((box[2][1] - box[0][1]) * 0.9) # Auto-size based on old box
                font = ImageFont.load_default() # For absolute perfection, upload a .ttf font
                
                draw.text((coords[0], coords[1]), new_text, fill=(0,0,0), font=font)
                
                st.success("Fixed Successfully!")
                st.image(final_img, caption="Modified Document")
                
                # Download Button
                buf = io.BytesIO()
                final_img.save(buf, format="PNG")
                st.download_button("Download Fixed File", buf.getvalue(), "fixed_document.png")
            else:
                st.error(f"Could not find '{target_text}' in the document.")
