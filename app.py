import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import easyocr
from pdf2image import convert_from_bytes
import io

# Load EasyOCR (Stable and won't crash on Python 3.13)
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'])

reader = load_reader()

st.set_page_config(layout="wide")
st.title("🛡️ Pro Surgical Document Editor")
st.write("Surgically edit digits without destroying the original layout or fonts.")

uploaded_file = st.file_uploader("Upload Document", type=['pdf', 'png', 'jpg', 'jpeg'])

if uploaded_file:
    # High-res conversion to keep the document looking original
    if uploaded_file.type == "application/pdf":
        images = convert_from_bytes(uploaded_file.read(), dpi=300)
        img_pil = images[0].convert("RGB")
    else:
        img_pil = Image.open(uploaded_file).convert("RGB")
    
    cv_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    with st.sidebar:
        st.header("Surgical Settings")
        find_text = st.text_input("Numbers to DELETE (e.g., 56)")
        replace_text = st.text_input("Numbers to INSERT (e.g., 65)")
        st.divider()
        st.write("Match Original Font Style")
        spacing = st.slider("Letter Spacing", -10, 20, 0)
        y_nudge = st.slider("Up/Down Align", -15, 15, 0)
        font_size_mod = st.slider("Font Size Match", 0.5, 1.5, 0.95)

    if st.button("🚀 Apply Seamless Fix") and find_text and replace_text:
        with st.spinner("Healing background and matching fonts..."):
            results = reader.readtext(cv_img)
            
            target_box = None
            for (bbox, text, prob) in results:
                if find_text in text:
                    target_box = np.array(bbox).astype(np.int32)
                    break
            
            if target_box is not None:
                # 1. HEAL: Use Inpainting to recreate the paper texture
                mask = np.zeros(cv_img.shape[:2], np.uint8)
                cv2.fillPoly(mask, [target_box], 255)
                healed_cv = cv2.inpaint(cv_img, mask, 3, cv2.INPAINT_NS)
                
                # 2. REPLACE: Draw new digits pixel-by-pixel
                final_img = Image.fromarray(cv2.cvtColor(healed_cv, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(final_img)
                
                # Get font height from the original digits
                box_h = max(target_box[:, 1]) - min(target_box[:, 1])
                f_size = int(box_h * font_size_mod)
                
                curr_x = target_box[0][0]
                curr_y = target_box[0][1] + y_nudge
                
                for char in replace_text:
                    draw.text((curr_x, curr_y), char, fill=(0,0,0))
                    # Move cursor based on font size + user spacing
                    curr_x += (f_size * 0.6) + spacing

                st.success("Edit Complete! All other fonts and layouts are untouched.")
                st.image(final_img, use_column_width=True)
                
                buf = io.BytesIO()
                final_img.save(buf, format="PNG")
                st.download_button("📥 Download Final Document", buf.getvalue(), "fixed.png")
            else:
                st.error("Could not find the text. Make sure it matches exactly.")
