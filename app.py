import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR
from pdf2image import convert_from_bytes
import io

# Load the engine
@st.cache_resource
def load_ocr():
    return PaddleOCR(use_angle_cls=False, show_log=False, lang='en')

ocr_engine = load_ocr()

st.set_page_config(layout="wide", page_title="Professional Document Editor")
st.title("🖋️ Professional Document Edit Tool")

uploaded_file = st.file_uploader("Upload Image-Based PDF", type=['pdf', 'png', 'jpg', 'jpeg'])

if uploaded_file:
    # PDF to Image High-Res
    if uploaded_file.type == "application/pdf":
        images = convert_from_bytes(uploaded_file.read(), dpi=300)
        img_pil = images[0].convert("RGB")
    else:
        img_pil = Image.open(uploaded_file).convert("RGB")
    
    cv_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    with st.sidebar:
        st.header("Precision Controls")
        old_val = st.text_input("Text to REMOVE", placeholder="e.g. 56")
        new_val = st.text_input("Text to INSERT", placeholder="e.g. 65")
        
        st.divider()
        st.write("Fine-Tune Appearance")
        f_scale = st.slider("Font Width Match", 0.5, 2.0, 1.0)
        char_spacing = st.slider("Character Spacing", -10, 20, 0)
        y_move = st.slider("Vertical Align", -20, 20, 0)

    if st.button("Apply Seamless Fix") and old_val and new_val:
        with st.spinner("Analyzing font and healing background..."):
            result = ocr_engine.ocr(cv_img, cls=True)
            
            target_data = None
            for line in result[0]:
                if old_val in line[1][0]:
                    target_data = line
                    break
            
            if target_data:
                # 1. COORDINATES
                box = np.array(target_data[0]).astype(np.int32)
                
                # 2. SEAMLESS HEALING (Inpainting)
                # Removes old ink and restores paper texture
                mask = np.zeros(cv_img.shape[:2], np.uint8)
                cv2.fillPoly(mask, [box], 255)
                healed_cv = cv2.inpaint(cv_img, mask, 3, cv2.INPAINT_NS)
                
                # 3. TEXT RECONSTRUCTION
                final_img = Image.fromarray(cv2.cvtColor(healed_cv, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(final_img)
                
                # Calculate font size from original box height
                box_h = max(box[:, 1]) - min(box[:, 1])
                font_size = int(box_h * 0.9)
                
                # Use a bold standard font
                try:
                    font = ImageFont.truetype("LiberationSans-Bold.ttf", font_size)
                except:
                    font = ImageFont.load_default()

                # Draw characters one by one to control spacing (Tracking)
                curr_x = box[0][0]
                curr_y = box[0][1] + y_move
                
                for char in new_val:
                    draw.text((curr_x, curr_y), char, fill=(0,0,0), font=font)
                    # Move X based on character width + manual spacing
                    char_w = draw.textbbox((0, 0), char, font=font)[2]
                    curr_x += (char_w * f_scale) + char_spacing

                st.subheader("Result (Original vs Modified)")
                st.image(final_img)
                
                # Export
                output = io.BytesIO()
                final_img.save(output, format="PNG")
                st.download_button("📥 Download Final Document", output.getvalue(), "fixed_doc.png")
            else:
                st.error("Could not locate that text in the document.")
