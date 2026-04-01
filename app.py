import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="AI Object Remover", layout="centered")

st.title("🧠 AI Object Remover (Manual - Stable)")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.image(image, caption="Original Image", use_column_width=True)

    st.write("✏️ Draw over the object you want to remove")

    # ✅ SAFE image conversion (prevents canvas crash)
    image_rgba = image.convert("RGBA")
    img_safe = Image.fromarray(np.array(image_rgba))

    canvas = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=20,
        stroke_color="red",
        background_image=img_safe,
        update_streamlit=True,
        height=img_safe.height,
        width=img_safe.width,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Remove Selected Object"):
        if canvas.image_data is not None:

            # 🧠 Create mask from drawing
            mask = canvas.image_data[:, :, 3]
            mask = (mask > 0).astype("uint8") * 255

            # Smooth edges
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

            with st.spinner("Removing object..."):
                result = cv2.inpaint(img_np, mask, 3, cv2.INPAINT_TELEA)

            result_img = Image.fromarray(result)

            st.image(result_img, caption="Result", use_column_width=True)

            # 📥 Download
            buf = io.BytesIO()
            result_img.save(buf, format="PNG")

            st.download_button(
                "Download Image",
                data=buf.getvalue(),
                file_name="removed.png",
                mime="image/png"
            )
        else:
            st.warning("Please draw on the image first.")
