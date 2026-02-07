import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
import easyocr
from pdf2image import convert_from_bytes
import io

@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'])

reader = load_reader()

st.set_page_config(layout="wide")
st.title("🛡️ Pro Surgical Document Editor")

uploaded_file = st.file_uploader("Upload Document", type=['pdf', 'png', 'jpg', 'jpeg'])

if uploaded_file:
    try:
        if uploaded_file.type == "application/pdf":
            # high-res conversion for professional results
            images = convert_from_bytes(uploaded_file.read(), dpi=300)
            img_pil = images[0].convert("RGB")
        else:
            img_pil = Image.open(uploaded_file).convert("RGB")
        
        cv_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        with st.sidebar:
            st.header("Precision Edit")
            find_val = st.text_input("Numbers to REMOVE")
            replace_val = st.text_input("Numbers to INSERT")
            st.divider()
            spacing = st.slider("Digit Spacing", -10, 20, 0)
            y_nudge = st.slider("Vertical Align", -15, 15, 0)

        if st.button("🚀 Apply Seamless Fix") and find_val and replace_val:
            results = reader.readtext(cv_img)
            target_box = next((np.array(bbox).astype(np.int32) for (bbox, text, prob) in results if find_val in text), None)
            
            if target_box is not None:
                mask = np.zeros(cv_img.shape[:2], np.uint8)
                cv2.fillPoly(mask, [target_box], 255)
                # Seamless healing of the paper texture
                healed_cv = cv2.inpaint(cv_img, mask, 3, cv2.INPAINT_NS)
                
                final_img = Image.fromarray(cv2.cvtColor(healed_cv, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(final_img)
                box_h = max(target_box[:, 1]) - min(target_box[:, 1])
                
                curr_x, curr_y = target_box[0][0], target_box[0][1] + y_nudge
                for char in replace_val:
                    draw.text((curr_x, curr_y), char, fill=(0,0,0))
                    curr_x += (box_h * 0.6) + spacing

                st.success("Fixed! Document layout remains original.")
                st.image(final_img, use_column_width=True)
                
                buf = io.BytesIO()
                final_img.save(buf, format="PNG")
                st.download_button("📥 Download", buf.getvalue(), "fixed.png")
            else:
                st.error("Text not found in document.")
                
    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.info("If seeing 'poppler' error, please wait 1-2 minutes for the system to install packages.txt")
