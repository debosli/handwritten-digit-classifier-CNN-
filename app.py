import streamlit as st
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

model = tf.keras.models.load_model("mnist_cnn.keras")

st.set_page_config(
    page_title="Handwritten Digit Classifier",
    page_icon="Icon/edit.png",
    layout="centered"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    h1 { color: #00d4aa; }
    .stProgress > div > div { background-color: #00d4aa; }
    div[data-testid="metric-container"] {
        background-color: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 10px;
        padding: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("Handwritten Digit Classifier")
st.write("Gambar angka 0-9 di canvas, prediksi akan muncul otomatis.")

col1, col2 = st.columns([1, 1.4])

with col1:
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )

with col2:
    if canvas_result.image_data is not None and canvas_result.image_data[:, :, 0].max() > 10:
        img = canvas_result.image_data
        img = img[:, :, 0]
        img = np.expand_dims(img, axis=-1)
        img = tf.image.resize(img, [28, 28])
        img = img / 255.0
        img = tf.reshape(img, (1, 28, 28, 1))
        
        prediction = model.predict(img, verbose=0)
        digit = np.argmax(prediction)
        confidence = prediction[0][digit] * 100
        
        # Top 3 kandidat
        top3_idx = np.argsort(prediction[0])[::-1][:3]
        
        st.markdown(f"### Predicted digit: {digit}")
        
        # Top 3 boxes
        c1, c2, c3 = st.columns(3)
        for i, col in enumerate([c1, c2, c3]):
            with col:
                idx = top3_idx[i]
                pct = prediction[0][idx] * 100
                st.metric(label=f"#{i+1}", value=str(idx), delta=f"{pct:.1f}%")
        
# Probability bar chart semua digit 0-9
        st.write("Probability distribution:")
        bars_html = ""
        for i in range(10):
            prob = prediction[0][i] * 100
            color = "#00d4aa" if i == digit else "#2d3250"
            bars_html += f"""
            <div style="display:flex; align-items:center; margin-bottom:6px; gap:8px;">
                <span style="width:16px; font-size:13px; color:#aaa;">{i}</span>
                <div style="flex:1; background:#1e2130; border-radius:4px; height:10px;">
                    <div style="width:{prob:.1f}%; background:{color}; height:10px; border-radius:4px;"></div>
                </div>
                <span style="width:45px; font-size:12px; color:#aaa; text-align:right;">{prob:.1f}%</span>
            </div>
            """
        st.markdown(bars_html, unsafe_allow_html=True)  # ← indent 8 spasi, di dalam with col2