from pathlib import Path
import joblib
import streamlit as st

from utils import email_security_features

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "phishing_model.joblib"

st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Phishing Email Detection")
st.write("Machine learning model for classifying an email as **Phishing** or **Safe**.")

if not MODEL_PATH.exists():
    st.error("Model not found. First run: python train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)

email_text = st.text_area(
    "Paste email content",
    height=260,
    placeholder="Paste the subject and body of an email here..."
)

if st.button("Analyze Email", type="primary"):
    if not email_text.strip():
        st.warning("Please paste an email first.")
    else:
        prediction = int(model.predict([email_text])[0])
        probability = model.predict_proba([email_text])[0]

        phishing_probability = probability[1] * 100
        safe_probability = probability[0] * 100
        features = email_security_features(email_text)

        if prediction == 1:
            st.error(f"⚠️ Prediction: PHISHING")
        else:
            st.success(f"✅ Prediction: SAFE")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Safe probability", f"{safe_probability:.2f}%")
        with col2:
            st.metric("Phishing probability", f"{phishing_probability:.2f}%")

        st.subheader("Detected email features")
        st.write({
            "URL count": int(features[0]),
            "Suspicious URL count": int(features[1]),
            "IP-address URL count": int(features[2]),
            "URL shortener count": int(features[3]),
            "@ in URL count": int(features[4]),
            "HTTP URL count": int(features[5]),
            "HTTPS URL count": int(features[6]),
            "Phishing keyword count": int(features[8]),
        })

st.caption("Educational project. A prediction is not a guarantee that an email is safe.")
