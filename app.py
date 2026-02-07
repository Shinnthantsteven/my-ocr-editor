import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
import easyocr
from pdf2image import convert_from_bytes
import io

# Load OCR engine
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'])

reader = load_reader()

st.set_page_config(layout="wide")
st.title("🛡️ Direct Visual Document Surgeon")

uploaded_file = st.file_uploader("Upload PDF or Image", type=['pdf', 'png', 'jpg', 'jpeg'])

if uploaded_file:
    # High-res conversion (300 DPI) for pro results
    if uploaded_file.type == "application/pdf":
        try:
            images = convert_from_bytes(uploaded_file.read(), dpi=300)
            img_pil = images[0].convert("RGB")
        except Exception as e:
            st.error("Poppler is still installing. Please wait 1 minute and refresh.")
            st.stop()
    else:
        img_pil = Image.open(uploaded_file).convert("RGB")
    
    cv_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # Automatically find everything on the page
    results = reader.readtext(cv_img)
    
    # Visual Layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Original Document")
        st.image(img_pil, use_container_width=True)

    with col2:
        st.subheader("Direct Fix Controls")
        # Let user choose by looking at what was found
        options = [f"Text: '{res[1]}' at Position: {res[0][0]}" for res in results]
        target_selection = st.selectbox("Pick the exact text to fix:", options)
        
        new_text = st.text_input("Enter the CORRECT replacement:")
        
        if st.button("🚀 Apply Surgical Fix") and new_text:
            idx = options.index(target_selection)
            target_box = np.array(results[idx][0]).astype(np.int32)
            
            # --- THE SURGERY (No Whiteout) ---
            # 1. Heal Background (Inpainting)
            mask = np.zeros(cv_img.shape[:2], np.uint8)
            cv2.fillPoly(mask, [target_box], 255)
            healed_cv = cv2.inpaint(cv_img, mask, 3, cv2.INPAINT_NS)
            
            # 2. Add New Text in same spot
            final_img = Image.fromarray(cv2.cvtColor(healed_cv, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(final_img)
            draw.text((target_box[0][0], target_box[0][1]), new_text, fill=(0,0,0))
            
            st.success("Fixed! Download below.")
            st.image(final_img, caption="Edited Result", use_container_width=True)
            
            buf = io.BytesIO()
            final_img.save(buf, format="PNG")
            st.download_button("📥 Download Result", buf.getvalue(), "fixed.png")
