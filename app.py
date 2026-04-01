import streamlit as st
from PIL import Image
import numpy as np
import cv2
from rembg import remove
import io
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="AI Object Remover", layout="centered")

st.title("🧠 AI Object Remover (Auto + Manual)")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.image(image, caption="Original Image", use_column_width=True)

    st.write("✏️ (Optional) Draw over object to remove manually")

    # ✅ FIX: safe image conversion for canvas
    img_safe = Image.fromarray(np.array(image))

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

    if st.button("Remove Object / Person"):
        with st.spinner("Processing..."):

            # -------------------------------
            # 🔹 AUTO MASK (person detection)
            # -------------------------------
            auto_mask = None
            try:
                output = remove(image)
                output_np = np.array(output)

                if output_np.shape[2] == 4:
                    auto_mask = output_np[:, :, 3]
                else:
                    auto_mask = cv2.cvtColor(output_np, cv2.COLOR_RGB2GRAY)

            except:
                auto_mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)

            # -------------------------------
            # 🔹 MANUAL MASK (drawn)
            # -------------------------------
            manual_mask = None
            if canvas.image_data is not None:
                manual_mask = canvas.image_data[:, :, 3]
                manual_mask = (manual_mask > 0).astype("uint8") * 255

            # -------------------------------
            # 🔹 COMBINE MASKS
            # -------------------------------
            if manual_mask is not None:
                if auto_mask is not None:
                    mask = cv2.bitwise_or(auto_mask, manual_mask)
                else:
                    mask = manual_mask
            else:
                mask = auto_mask

            # If nothing selected → skip
            if mask is None:
                st.warning("No object detected or selected.")
                st.stop()

            # -------------------------------
            # 🔹 CLEAN MASK
            # -------------------------------
            _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

            # -------------------------------
            # 🔹 INPAINT
            # -------------------------------
            result = cv2.inpaint(img_np, mask, 3, cv2.INPAINT_TELEA)
            result_img = Image.fromarray(result)

        st.image(result_img, caption="Result", use_column_width=True)

        # Download
        buf = io.BytesIO()
        result_img.save(buf, format="PNG")

        st.download_button(
            "Download Image",
            data=buf.getvalue(),
            file_name="result.png",
            mime="image/png"
        )
