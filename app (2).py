import glob
import json
import random
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

IMG_SIZE = (224, 224)
CONF_THRESHOLD = 0.60
BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
EMOJI = {"rock": "✊", "paper": "✋", "scissors": "✌️"}

st.set_page_config(page_title="Rock Paper Scissors AI", page_icon="✊", layout="centered")


@st.cache_resource
def load_model():
    model_files = sorted(glob.glob("*.keras"))   # works even if the file was renamed, e.g. rps_model_small.keras
    if not model_files:
        st.error("No .keras model file found in the repository. Upload rps_model.keras to GitHub.")
        st.stop()
    model = tf.keras.models.load_model(model_files[0])
    with open("class_names.json") as f:
        class_names = json.load(f)
    return model, class_names


def prepare_image(pil_img):
    """Same preprocessing as the notebook: fix rotation, pad to a square, resize to 224x224."""
    img = ImageOps.exif_transpose(pil_img).convert("RGB")
    w, h = img.size
    s = max(w, h)
    square = Image.new("RGB", (s, s), (255, 255, 255))
    square.paste(img, ((s - w) // 2, (s - h) // 2))
    return square.resize(IMG_SIZE, Image.BILINEAR)


def predict_gesture(model, class_names, pil_img):
    x = np.expand_dims(np.array(prepare_image(pil_img), dtype=np.float32), axis=0)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx]), probs


def play_round(user_move):
    computer_move = random.choice(list(BEATS))
    if user_move == computer_move:
        return computer_move, "Draw"
    if BEATS[user_move] == computer_move:
        return computer_move, "You win!"
    return computer_move, "Computer wins!"


# ---------------- UI ----------------
if "score" not in st.session_state:
    st.session_state.score = {"You": 0, "Computer": 0, "Draw": 0}
    st.session_state.last_file = None

model, class_names = load_model()

st.title("✊ ✋ ✌️ Rock Paper Scissors AI")
st.write("Show your hand to the camera or upload a photo. The AI recognises your move, "
         "the computer picks its own, and the winner is decided.")
st.info("Tip: plain background, one hand, whole hand visible in the photo.")

source = st.radio("Image source", ["Use camera", "Upload a photo"], horizontal=True)
file = (st.camera_input("Show your move") if source == "Use camera"
        else st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"]))

if file is not None:
    pil_img = Image.open(file)
    move, conf, probs = predict_gesture(model, class_names, pil_img)

    col1, col2 = st.columns(2)
    col1.image(ImageOps.exif_transpose(pil_img), caption="Your photo", use_container_width=True)

    with col2:
        st.subheader("AI sees")
        for name, p in sorted(zip(class_names, probs), key=lambda t: -t[1]):
            st.progress(float(p), text=f"{EMOJI[name]} {name}: {p:.0%}")

    if conf < CONF_THRESHOLD:
        st.warning(f"Not sure ({conf:.0%}). Please try another photo with a clearer hand.")
    else:
        file_id = getattr(file, "file_id", None) or file.name
        if st.session_state.last_file != file_id:   # play only once per new photo
            computer, result = play_round(move)
            st.session_state.last_file = file_id
            st.session_state.last_round = (move, computer, result)
            key = {"You win!": "You", "Computer wins!": "Computer", "Draw": "Draw"}[result]
            st.session_state.score[key] += 1

        you, computer, result = st.session_state.last_round
        st.markdown(f"### You: {EMOJI[you]} {you} &nbsp; vs &nbsp; Computer: {EMOJI[computer]} {computer}")
        if result == "You win!":
            st.success("🎉 You win!")
        elif result == "Computer wins!":
            st.error("🤖 Computer wins!")
        else:
            st.info("🤝 Draw")

st.divider()
s = st.session_state.score
c1, c2, c3 = st.columns(3)
c1.metric("You", s["You"])
c2.metric("Computer", s["Computer"])
c3.metric("Draws", s["Draw"])
if st.button("Reset score"):
    st.session_state.score = {"You": 0, "Computer": 0, "Draw": 0}
    st.session_state.last_file = None
    st.rerun()

st.caption("Model: MobileNetV2 (transfer learning + fine-tuning) trained on the Kaggle Rock-Paper-Scissors dataset "
           "with background replacement and rotation augmentation.")
