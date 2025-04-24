import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_csv(url, sep='\t', names=["label", "message"])
    return df

# Train model
@st.cache_resource
def train_model(data):
    X_train, X_test, y_train, y_test = train_test_split(
        data["message"], data["label"], test_size=0.2, random_state=42)
    
    model = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc

# UI Layout
def main():
    st.title("📧 Spam Detection App")
    st.write("A simple machine learning app to classify SMS as spam or ham.")
    
    df = load_data()
    model, accuracy = train_model(df)

    st.subheader("Enter a message:")
    user_input = st.text_area("Type here...", height=150)
    
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        prediction = model.predict([user_input])[0]
        st.success(f"Prediction: **{prediction.upper()}**")
    
    st.subheader("Model Performance")
    st.write(f"Accuracy on test data: **{accuracy * 100:.2f}%**")
    
    with st.expander("See Dataset Sample"):
        st.dataframe(df.sample(10))

if __name__ == "__main__":
    main()
