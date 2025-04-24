import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load and display data
@st.cache_data
def load_data():
    return pd.read_csv("sms_spam_dataset.csv")

# Train a simple spam classifier
@st.cache_resource
def train_model(data):
    X_train, X_test, y_train, y_test = train_test_split(
        data["message"], data["label"], test_size=0.2, random_state=42
    )
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])
    pipeline.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipeline.predict(X_test))
    return pipeline, acc

def main():
    st.set_page_config(page_title="Spam Detector", layout="centered")
    st.title("📩 SMS Spam Detector")

    data = load_data()
    model, accuracy = train_model(data)

    st.subheader("Type an SMS message:")
    user_input = st.text_area("Enter your message here", height=150)

    if st.button("Check for Spam"):
        if user_input.strip():
            prediction = model.predict([user_input])[0]
            st.success(f"This message is **{prediction.upper()}**")
        else:
            st.warning("Please enter a message.")

    st.write(f"Model Accuracy: **{accuracy * 100:.2f}%**")

    with st.expander("📊 View Sample Data"):
        st.dataframe(data.sample(5))

if __name__ == "__main__":
    main()
