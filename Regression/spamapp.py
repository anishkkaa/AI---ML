import  streamlit as st
import pandas as pd
import joblib

#page congiguration
st.set_page_config(
    page_title="Spam Detection App",
    page_icon="📧",
    layout="wide",

)
#Load the model and vectorizer    
@st.cache_resource
def load_model():
    model_data = joblib.load('spam_detector.joblib')
    return model_data['model'], model_data['vectorizer']

try:
    model, vectorizer = load_model()

    #main UI
    st.title("📧 Spam Message Detector")
    st.write()
    #text input
    message = st.text_area("Enter the message to classify:", height=100)
    if st.button("Check Message", type="primary"):
        if message:
            #preprocess and predict
            message= message.lower()
            message_vectorized = vectorizer.transform([message])
            prediction = model.predict(message_vectorized)
            probablity = model.predict_proba(message_vectorized)

            #show result
            st.write("---")
            col1, col2 = st.columns(2)
            with col1:
                if prediction[0] == "spam":
                    st.error(" ✉️This message is likely SPAM!")
                else:
                    st.success("📨This message is likely HAM (not spam)!")

            with col2:
                spam_prob = probablity[0][1]if model.classes_[1] == "spam" else probablity[0][0]
                st.metric(
                    "Spam Probablity",
                    f"{spam_prob*100:.2f} %"
                )
                #show confidence bars
                st.write("Confidence Scores:")
                st.progress(1 - spam_prob, text="Spam")
                st.progress(spam_prob, text="Ham")
        else:
            st.warning("Please enter a message to check")
    
        # example messages
        with st.expander("💡Example Messages"):
            st.write("""
                **Spam Examples:**
                - "Congratulations! You've won a $1000 Walmart gift card. Click here to claim your prize."
                - "Urgent! Your account has been compromised. Please verify your information immediately."
                - "Get paid to work from home. No experience required. Sign up now!"
    
                **Ham Examples:**
                - "Hey, are we still on for lunch tomorrow?"
                - "Don't forget to bring the documents for the meeting."
                - "Can you send me the report by the end of the day?"
                - "Happy birthday! Wishing you a wonderful day filled with joy and surprises."
            """)
except Exception as e:
    st.error(f"""Error loading model: {e}"
            please ensure:
            1. THe 'spam_detector.joblib' file is in the correct directory.
            2. The saved file contains both the model and vectorizer.body
            3. All required packages are installed."""
            )
    #footer
    st.markdown("---")
    st.markdown("Developed using Streamlit")