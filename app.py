import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import json

st.set_page_config(page_title="EcoSort AI", page_icon="♻️", layout="wide")

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "ecosort_mobilenetv2.keras"
CLASS_PATH = ROOT / "models" / "class_names.json"

st.title("♻️ EcoSort AI")
st.subheader("Intelligent Waste Classification using Transfer Learning")
st.write("Upload a waste image to classify it into a waste category and view disposal guidance.")

CLASSES_FALLBACK = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
GUIDANCE = {
    "cardboard": ("Dry recyclable", "Keep clean and dry; flatten boxes before placing in paper/cardboard recycling."),
    "glass": ("Recyclable where accepted", "Rinse containers and follow local glass collection rules. Do not handle broken glass without protection."),
    "metal": ("Recyclable", "Empty and rinse cans; place them in the local metal recycling stream."),
    "paper": ("Dry recyclable", "Keep paper clean and dry. Greasy or food-soiled paper may not be accepted."),
    "plastic": ("Check resin/local rules", "Empty and rinse the item. Follow local recycling rules because not every plastic type is accepted."),
    "trash": ("General waste", "If the item cannot be reused, composted, or accepted for recycling, follow local general-waste rules.")
}

@st.cache_resource
def load_model():
    try:
        import tensorflow as tf
        if not MODEL_PATH.exists() or not CLASS_PATH.exists():
            return None, None, "Model files are not available yet. Train the model first using train_model.py."
        model = tf.keras.models.load_model(MODEL_PATH)
        classes = json.loads(CLASS_PATH.read_text(encoding="utf-8"))
        return model, classes, None
    except Exception as e:
        return None, None, f"Could not load the model: {e}"

with st.sidebar:
    st.header("About this project")
    st.write("Model: MobileNetV2 transfer learning")
    st.write("Classes: cardboard, glass, metal, paper, plastic, trash")
    st.caption("This is an educational prototype. Follow local waste rules; image predictions may be wrong.")

uploaded = st.file_uploader("Choose a waste image", type=["jpg", "jpeg", "png", "webp"])
if uploaded:
    try:
        image = Image.open(uploaded).convert("RGB")
        left, right = st.columns([1, 1])
        with left:
            st.image(image, caption="Uploaded image", use_container_width=True)
        model, classes, error = load_model()
        with right:
            if error:
                st.warning(error)
                st.info("Follow README.md to download the dataset and train the model.")
            else:
                with st.spinner("Classifying image..."):
                    img = image.resize((224, 224))
                    arr = np.asarray(img, dtype=np.float32)
                    arr = np.expand_dims(arr, axis=0)
                    # MobileNetV2 preprocessing expects pixel values in [-1, 1].
                    import tensorflow as tf
                    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
                    probs = model.predict(arr, verbose=0)[0]
                    best = int(np.argmax(probs))
                    label = classes[best]
                    confidence = float(probs[best]) * 100
                st.metric("Predicted category", label.replace("_", " ").title())
                st.metric("Model confidence", f"{confidence:.1f}%")
                if confidence < 55:
                    st.warning("Low confidence. Try a clearer image with one item centered against a plain background.")
                kind, advice = GUIDANCE.get(label.lower(), ("Check local rules", "Verify the item with your local waste collection guidance."))
                st.markdown(f"**Disposal guidance: {kind}**")
                st.write(advice)
                st.markdown("**Class probabilities**")
                for name, score in sorted(zip(classes, probs), key=lambda x: x[1], reverse=True):
                    st.progress(float(score), text=f"{name.title()}: {score*100:.1f}%")
    except Exception as e:
        st.error(f"Could not process this image: {e}")
else:
    st.info("Upload a JPG, JPEG, PNG, or WEBP image to begin.")
    st.markdown("### How it works")
    st.write("Image → resize/preprocess → MobileNetV2 model → predicted waste category → disposal guidance.")
