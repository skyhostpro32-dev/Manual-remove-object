import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io

st.set_page_config(page_title="AI Object Remover", layout="centered")

st.title("🧠 AI Object Remover (Click Mask - No Canvas)")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.image(image, caption="Click points on object")

    # Store clicked points
    if "points" not in st.session_state:
        st.session_state.points = []

    # Click input
    click = st.number_input("Add X coordinate", min_value=0, max_value=image.width)
    click_y = st.number_input("Add Y coordinate", min_value=0, max_value=image.height)

    if st.button("Add Point"):
        st.session_state.points.append((int(click), int(click_y)))

    st.write("Selected Points:", st.session_state.points)

    if st.button("Remove Object"):
        if len(st.session_state.points) == 0:
            st.warning("Add at least one point")
        else:
            mask = np.zeros((image.height, image.width), dtype=np.uint8)

            # Create circular mask around points
            for (x, y) in st.session_state.points:
                cv2.circle(mask, (x, y), 30, 255, -1)

            # Smooth mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

            with st.spinner("Removing object..."):
                result = cv2.inpaint(img_np, mask, 3, cv2.INPAINT_TELEA)

            result_img = Image.fromarray(result)

            st.image(result_img, caption="Result")

            # Download
            buf = io.BytesIO()
            result_img.save(buf, format="PNG")

            st.download_button(
                "Download Image",
                data=buf.getvalue(),
                file_name="removed.png",
                mime="image/png"
            )
