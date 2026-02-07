import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import easyocr
from pdf2image import convert_from_bytes
import io

# Stable engine for surgical precision
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'])

reader = load_reader()

st.set_page_config(layout="wide", page_title="Precision Doc Editor")
st.title("📑 Surgical Document Digit Fixer")

uploaded_file = st.file_uploader("Upload Image or PDF", type=['pdf', 'png', 'jpg', 'jpeg'])

if uploaded_file:
    # Maintain high resolution (300 DPI) to keep document crisp
    if uploaded_file.type == "application/pdf":
        images = convert_from_bytes(uploaded_file.read(), dpi=300)
        img_pil = images[0].convert("RGB")
    else:
        img_pil = Image.open(uploaded_file).convert("RGB")
    
    cv_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    with st.sidebar:
        st.header("Precision Tuning")
        old_val = st.text_input("Numbers to DELETE", placeholder="e.g. 56")
        new_val = st.text_input("Numbers to INSERT", placeholder="e.g. 65")
        st.divider()
        st.write("Match Original Font")
        spacing = st.slider("Digit Spacing", -10, 20, 0)
        y_nudge = st.slider("Vertical Alignment", -15, 15, 0)
        f_size_mod = st.slider("Font Size Match", 0.5, 1.5, 0.95)

    if st.button("🚀 Apply Seamless Fix") and old_val and new_val:
        with st.spinner("Finding coordinates and healing background..."):
            results = reader.readtext(cv_img)
            
            target_box = None
            for (bbox, text, prob) in results:
                if old_val in text:
                    target_box = np.array(bbox).astype(np.int32)
                    break
            
            if target_box is not None:
                # 1. HEAL: Remove old ink and restore paper texture
                mask = np.zeros(cv_img.shape[:2], np.uint8)
                cv2.fillPoly(mask, [target_box], 255)
                healed_cv = cv2.inpaint(cv_img, mask, 3, cv2.INPAINT_NS)
                
                # 2. REPLACE: Add new digits character-by-character
                final_img = Image.fromarray(cv2.cvtColor(healed_cv, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(final_img)
                
                box_h = max(target_box[:, 1]) - min(target_box[:, 1])
                f_size = int(box_h * f_size_mod)
                
                curr_x = target_box[0][0]
                curr_y = target_box[0][1] + y_nudge
                
                for char in new_val:
                    # Renders text as pixels on the image
                    draw.text((curr_x, curr_y), char, fill=(0,0,0))
                    # Manual tracking/spacing to match original
                    curr_x += (f_size * 0.6) + spacing

                st.success("Surgery Complete! The rest of the document is 100% original.")
                st.image(final_img, use_column_width=True)
                
                buf = io.BytesIO()
                final_img.save(buf, format="PNG")
                st.download_button("📥 Download Result", buf.getvalue(), "fixed_doc.png")
            else:
                st.error("Text not found. Check your spelling.")
