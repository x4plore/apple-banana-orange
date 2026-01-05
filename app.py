import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# CONFIG
# =========================
IMG_SIZE = (128, 128)

# ⚠️ MUST MATCH train_ds.class_names ORDER
CLASS_NAMES = [
    'freshapples', 
    'freshbanana', 
    'freshoranges', 
    'rottenapples', 
    'rottenbanana', 
    'rottenoranges', 
    'unripeapple', 
    'unripebanana', 
    'unripeorange'
]

CONFIDENCE_THRESHOLD = 0.85

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Fruit Ripeness Classifier",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .ripe {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        color: #155724;
    }
    .rotten {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        color: #721c24;
    }
    .unripe {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        color: #856404;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🍎 Fruit Ripeness Classifier</div>', unsafe_allow_html=True)
st.markdown("**Analyze the ripeness state of your fruit using AI**")
st.divider()

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("final_model_v2.h5")

model = load_model()

# =========================
# HELPER FUNCTIONS
# =========================
def get_ripeness_state(class_name):
    """Extract ripeness state from class name"""
    if class_name.startswith("fresh"):
        return "Ripe"
    elif class_name.startswith("rotten"):
        return "Rotten"
    elif class_name.startswith("unripe"):
        return "Unripe"
    return class_name

def get_ripeness_emoji(state):
    """Return emoji for ripeness state"""
    if state == "Ripe":
        return "✅"
    elif state == "Rotten":
        return "❌"
    elif state == "Unripe":
        return "⏳"
    return "❓"

def get_ripeness_color(state):
    """Return color class for ripeness state"""
    if state == "Ripe":
        return "ripe"
    elif state == "Rotten":
        return "rotten"
    elif state == "Unripe":
        return "unripe"
    return ""

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    img_array = np.array(image).astype(np.float32)
    img_array /= 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# =========================
# FILE UPLOAD SECTION
# =========================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📸 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a fruit image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

with col2:
    st.subheader("ℹ️ How it works")
    st.info(
        "Upload a clear image of your fruit. The AI will analyze it and "
        "determine whether it's **ripe**, **rotten**, or **unripe**."
    )
    
    st.subheader("🎯 Supported Fruits")
    st.write("This classifier works best with:")
    fruits_cols = st.columns(3)
    with fruits_cols[0]:
        st.markdown("🍎 **Apples**")
    with fruits_cols[1]:
        st.markdown("🍌 **Bananas**")
    with fruits_cols[2]:
        st.markdown("🍊 **Oranges**")
    
    st.subheader("📊 Ripeness States")
    state_info = {
        "✅ Ripe": "Fruit is ready to eat",
        "❌ Rotten": "Fruit has spoiled",
        "⏳ Unripe": "Fruit needs more time"
    }
    for state, desc in state_info.items():
        st.write(f"{state}: {desc}")

# =========================
# PREDICTION SECTION
# =========================
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col_img, col_results = st.columns([1, 1], gap="large")
    
    with col_img:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col_results:
        if st.button("🔍 Classify", use_container_width=True, type="primary"):
            with st.spinner("Analyzing image..."):
                input_data = preprocess_image(image)
                predictions = model.predict(input_data, verbose=0)[0]

                # Top prediction
                pred_index = int(np.argmax(predictions))
                confidence = float(predictions[pred_index])
                pred_class = CLASS_NAMES[pred_index]
                ripeness_state = get_ripeness_state(pred_class)
                emoji = get_ripeness_emoji(ripeness_state)
                color_class = get_ripeness_color(ripeness_state)

            # Main result
            st.markdown(
                f'<div class="prediction-box {color_class}">'
                f'<h3>{emoji} {ripeness_state}</h3>'
                f'<p style="font-size: 1.2em; margin: 10px 0;"><b>Confidence: {confidence:.1%}</b></p>'
                f'</div>',
                unsafe_allow_html=True
            )
            
            st.progress(confidence)

            # Low confidence warning
            if confidence < CONFIDENCE_THRESHOLD:
                st.warning(
                    f"⚠️ **Low Confidence Alert**\n\n"
                    f"The model is only {confidence:.1%} confident. "
                    f"Please use your own judgment or inspect the fruit manually."
                )

# =========================
# FOOTER
# =========================
st.divider()
st.caption("🤖 Powered by TensorFlow | Analyzes 3 ripeness states across 3 fruits")

