import streamlit as st
import pandas as pd
import pickle
import joblib

# Load model & label encoder

model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")


with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Load feature data to get columns
df = pd.read_csv("disease_data.csv")
features = df.drop(columns=["Disease"]).columns.tolist()

# Streamlit UI
st.title("🧠 MedGuardian")
st.markdown("### AI-Powered Early Disease Detection System")

# Symptom & Gender Reference
with st.expander("📋 View Symptom & Gender Reference"):
    st.markdown("""
    **Gender**:
    - **0** → Male
    - **1** → Female

    **Symptoms** (0 = No, 1 = Yes):
    - **fever** → Patient has fever
    - **cough** → Persistent cough
    - **fatigue** → Feeling unusually tired
    - **headache** → Head pain or pressure
    - **nausea** → Feeling of wanting to vomit
    """)

user_input = []
st.markdown("### Fill the details:")

for feature in features:
    if df[feature].dtype == 'object':
        val = st.selectbox(f"{feature}:", df[feature].unique())
    else:
        val = st.slider(f"{feature}:", int(df[feature].min()), int(df[feature].max()))
    user_input.append(val)

if st.button("🔍 Predict Disease"):
    input_df = pd.DataFrame([user_input], columns=features)
    pred = model.predict(input_df)
    disease = le.inverse_transform(pred)[0]
    st.success(f"🩺 Disease Predicted: **{disease}**")


