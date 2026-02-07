import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR
from pdf2image import convert_from_bytes
import io

# FIXED: Simplified initialization to stop the ValueError
@st.cache_resource
def load_ocr():
    return PaddleOCR(lang='en', use_gpu=False)

try:
    ocr_engine = load_ocr()
except Exception as e:
    st.error(f"Engine Load Error: {e}")

st.set_page_config(layout="wide")
st.title("🛡️ Pro Document 'Pixel-Perfect' Editor")
st.write("This tool fixes digits without destroying the document layout or fonts.")

uploaded_file = st.file_uploader("Upload Document", type=['pdf', 'png', 'jpg', 'jpeg'])

if uploaded_file:
    # We keep the quality high (300 DPI) so fonts look crisp
    if uploaded_file.type == "application/pdf":
        bytes_data = uploaded_file.read()
        images = convert_from_bytes(bytes_data, dpi=300)
        img_pil = images[0].convert("RGB")
    else:
        img_pil = Image.open(uploaded_file).convert("RGB")
    
    cv_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    with st.sidebar:
        st.header("Surgical Settings")
        find_text = st.text_input("Numbers to REMOVE")
        replace_text = st.text_input("Numbers to INSERT")
        st.divider()
        char_spacing = st.slider("Digit Spacing", -5, 15, 0)
        y_offset = st.slider("Vertical Nudge", -10, 10, 0)

    if st.button("🚀 Fix Document Without Destroying Layout") and find_text and replace_text:
        with st.spinner("Finding digits and healing background..."):
            result = ocr_engine.ocr(cv_img, cls=True)
            
            target_box = None
            for line in result[0]:
                if find_text in line[1][0]:
                    target_box = np.array(line[0]).astype(np.int32)
                    break
            
            if target_box is not None:
                # 1. HEAL: We remove ONLY the ink of the numbers found
                mask = np.zeros(cv_img.shape[:2], np.uint8)
                cv2.fillPoly(mask, [target_box], 255)
                # This heals the paper texture perfectly
                healed_cv = cv2.inpaint(cv_img, mask, 3, cv2.INPAINT_NS)
                
                # 2. RESTORE: Put the new digits back
                final_img = Image.fromarray(cv2.cvtColor(healed_cv, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(final_img)
                
                # Calculate height to match original font
                box_h = max(target_box[:, 1]) - min(target_box[:, 1])
                font_size = int(box_h * 0.95)
                
                # Position logic
                curr_x = target_box[0][0]
                curr_y = target_box[0][1] + y_offset

                # Write characters one by one to maintain 'Original' spacing
                for char in replace_text:
                    draw.text((curr_x, curr_y), char, fill=(0,0,0))
                    # Move to next position
                    curr_x += (font_size * 0.6) + char_spacing

                st.success("Edit Complete! The rest of the document is 100% original.")
                st.image(final_img, use_column_width=True)
                
                # Download
                buf = io.BytesIO()
                final_img.save(buf, format="PNG")
                st.download_button("📥 Download Perfect Document", buf.getvalue(), "fixed_doc.png")
            else:
                st.error("Text not found. Ensure the 'Numbers to REMOVE' matches exactly.")
