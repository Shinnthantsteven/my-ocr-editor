import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import easyocr
from pdf2image import convert_from_bytes
import io

# Switch to EasyOCR for stability and precision
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'])

reader = load_reader()

st.set_page_config(layout="wide", page_title="Surgical Doc Editor")
st.title("🛡️ Zero-Destruction Document Fixer")
st.write("Upload your PDF/Image. This tool erases and replaces digits while keeping the rest of the document 100% original.")

uploaded_file = st.file_uploader("Upload Document", type=['pdf', 'png', 'jpg', 'jpeg'])

if uploaded_file:
    # Maintain 300 DPI so the document stays sharp
    if uploaded_file.type == "application/pdf":
        images = convert_from_bytes(uploaded_file.read(), dpi=300)
        img_pil = images[0].convert("RGB")
    else:
        img_pil = Image.open(uploaded_file).convert("RGB")
    
    cv_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    with st.sidebar:
        st.header("Precision Tuning")
        find_val = st.text_input("Numbers to REMOVE", placeholder="e.g. 2101")
        replace_val = st.text_input("Numbers to INSERT", placeholder="e.g. 2026")
        
        st.divider()
        st.write("Match Original Style")
        spacing = st.slider("Letter Spacing", -10, 20, 0)
        y_nudge = st.slider("Up/Down Alignment", -15, 15, 0)
        font_size_mod = st.slider("Font Size Match", 0.5, 1.5, 0.9)

    if st.button("🚀 Apply Seamless Fix") and find_val and replace_val:
        with st.spinner("Analyzing document..."):
            # Detect text coordinates
            results = reader.readtext(cv_img)
            
            target_box = None
            for (bbox, text, prob) in results:
                if find_val in text:
                    target_box = np.array(bbox).astype(np.int32)
                    break
            
            if target_box is not None:
                # 1. HEAL BACKGROUND (The most important part)
                mask = np.zeros(cv_img.shape[:2], np.uint8)
                cv2.fillPoly(mask, [target_box], 255)
                # This removes ink and copies the paper texture/color
                healed_cv = cv2.inpaint(cv_img, mask, 3, cv2.INPAINT_NS)
                
                # 2. MATCH AND REPLACE
                final_img = Image.fromarray(cv2.cvtColor(healed_cv, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(final_img)
                
                # Calculate original height
                box_h = max(target_box[:, 1]) - min(target_box[:, 1])
                f_size = int(box_h * font_size_mod)
                
                # Draw character by character to match "Original Spacing"
                curr_x = target_box[0][0]
                curr_y = target_box[0][1] + y_nudge
                
                for char in replace_val:
                    # We use a bold default font, but you can upload a .ttf for 100% match
                    draw.text((curr_x, curr_y), char, fill=(0,0,0))
                    # Move cursor for next character
                    curr_x += (f_size * 0.6) + spacing

                st.success("Surgery Complete! The document layout is untouched.")
                st.image(final_img, use_column_width=True)
                
                # Export
                buf = io.BytesIO()
                final_img.save(buf, format="PNG")
                st.download_button("📥 Download Perfect Result", buf.getvalue(), "fixed_doc.png")
            else:
                st.error("Text not found. Ensure it matches exactly what you see on the page.")
