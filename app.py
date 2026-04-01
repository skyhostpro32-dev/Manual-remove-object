import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="AI Object Remover", layout="centered")

st.title("🧠 AI Object Remover (Brush Tool)")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Resize for stability
    image = image.resize((600, 400))
    img_np = np.array(image)

    st.write("✏️ Brush over object (align with image below)")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Image")

    with col2:
        canvas = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=25,
            stroke_color="red",
            height=400,
            width=600,
            drawing_mode="freedraw",
            key="canvas",
        )

    if st.button("Remove Object"):
        if canvas.image_data is not None:

            # Extract mask
            mask = canvas.image_data[:, :, 3]
            mask = (mask > 0).astype("uint8") * 255

            # Smooth mask
            kernel = np.ones((7, 7), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)

            with st.spinner("Processing..."):
                result = cv2.inpaint(img_np, mask, 3, cv2.INPAINT_TELEA)

            result_img = Image.fromarray(result)

            st.image(result_img, caption="Result")

            # Download
            buf = io.BytesIO()
            result_img.save(buf, format="PNG")

            st.download_button(
                "Download",
                data=buf.getvalue(),
                file_name="removed.png",
                mime="image/png"
            )
        else:
            st.warning("Brush over object first")
