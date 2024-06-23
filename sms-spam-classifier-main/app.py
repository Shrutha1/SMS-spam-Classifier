import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Initialize Porter Stemmer
ps = PorterStemmer()

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Custom CSS to add a background image and style for messages
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://cdn.wallpapersafari.com/77/64/KwuO8p.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    .success-message {
        color: black !important;
    }
    .error-message {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app title
st.title("📧 Email/SMS Spam Classifier")

# Introduction and instructions
st.markdown("""
Welcome to the Email/SMS Spam Classifier! This tool uses machine learning to identify whether a given message is spam or not. 
Simply enter the message you want to check in the text area below and click 'Predict'.
""")

# Input area for the message
input_sms = st.text_area("Type your message here...", key="input_message", value="", height=150)

# Predict button
if st.button('Predict'):
    with st.spinner('Analyzing the message...'):
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. Predict
        result = model.predict(vector_input)[0]
        # 4. Display result
        if result == 1:
            st.error("🚫 Spam")
            # Show danger image pop-up for spam
            st.markdown(
                f"""
                <style>
                #popup-spam {{
                    display: block;
                    position: fixed;
                    z-index: 1000;
                    left: 50%;
                    top: 50%;
                    width: 80%;
                    max-width: 400px;
                    transform: translate(-50%, -50%);
                    background-color: white;
                    border: 2px solid red;
                    border-radius: 10px;
                    padding: 20px;
                    text-align: center;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                }}
                #popup-spam img {{
                    width: 100px;
                    height: 100px;
                }}
                #popup-spam p {{
                    color: black; /* Change text color to black */
                }}
                </style>
                <div id="popup-spam">
                    <img src="https://cdn.pixabay.com/photo/2022/02/04/14/55/skull-6992912_960_720.jpg" alt="Danger">
                    <p><strong>Warning:</strong> This message is classified as spam!</p>
                    <p>{input_sms}</p>
                </div>
                <script>
                setTimeout(function() {{
                    var popup = document.getElementById('popup-spam');
                    popup.style.display = 'none';
                }}, 5000); // Adjusted to 5 seconds
                </script>
                """,
                unsafe_allow_html=True
            )
        else:
            st.success("✅ Not Spam")
            # Show success image pop-up for not spam
            st.markdown(
                f"""
                <style>
                #popup-notspam {{
                    display: block;
                    position: fixed;
                    z-index: 1000;
                    left: 50%;
                    top: 50%;
                    width: 80%;
                    max-width: 400px;
                    transform: translate(-50%, -50%);
                    background-color: white;
                    border: 2px solid green;
                    border-radius: 10px;
                    padding: 20px;
                    text-align: center;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                }}
                #popup-notspam img {{
                    width: 100px;
                    height: 100px;
                }}
                #popup-notspam p {{
                    color: black; /* Change text color to black */
                }}
                </style>
                <div id="popup-notspam">
                    <img src="https://cdn-icons-png.freepik.com/512/9827/9827073.png" alt="Success">
                    <p><strong>Success:</strong> This message is not classified as spam!</p>
                    <p>{input_sms}</p>
                </div>
                <script>
                setTimeout(function() {{
                    var popup = document.getElementById('popup-notspam');
                    popup.style.display = 'none';
                }}, 5000); // Adjusted to 5 seconds
                </script>
                """,
                unsafe_allow_html=True
            )

# Add a back button
st.markdown('<br>', unsafe_allow_html=True)  # Add some space
if st.button('Back'):
    # Clear previous message and reset UI
    input_sms = ""

# Footer
st.markdown("""
---
Developed by Detector  
Powered by Streamlit
""")
