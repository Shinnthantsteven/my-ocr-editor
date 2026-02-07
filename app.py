import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR

# 1. Start the OCR Engine
ocr = PaddleOCR(use_angle_cls=True, lang='en')

st.title("🎯 Perfect Digit Fixer")
uploaded_file = st.file_uploader("Upload your document image", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    # Load image
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)
    
    st.image(img, caption="Original Document", use_column_width=True)
    
    # User Input
    old_text = st.text_input("What digits do you want to REMOVE?")
    new_text = st.text_input("What are the NEW digits?")
    
    if st.button("Fix Document"):
        # 2. Find the coordinates of the old text
        results = ocr.ocr(img_array, cls=True)
        
        found = False
        for line in results[0]:
            text_detected = line[1][0]
            if old_text in text_detected:
                # Get the box coordinates
                box = line[0] # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                
                # 3. CLEAN (Inpaint): Cover the old text with the background color
                # We pick the color from the pixel right next to the box
                bg_color = img.getpixel((box[0][0] - 5, box[0][1]))
                
                draw = ImageDraw.Draw(img)
                draw.polygon([tuple(p) for p in box], fill=bg_color)
                
                # 4. ADD NEW TEXT: (In a real app, we would match font here)
                # For now, we use a standard clean font
                draw.text((box[0][0], box[0][1]), new_text, fill=(0,0,0))
                found = True
        
        if found:
            st.success("Fixed!")
            st.image(img, caption="Modified Document", use_column_width=True)
        else:
            st.error("Could not find those digits in the document.")
