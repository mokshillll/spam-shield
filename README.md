# spam-shield
# SMS Spam Detection App

A simple machine learning web app to classify SMS messages as **spam** or **ham** (not spam), built using **Python**, **scikit-learn**, and **Streamlit**.

---

 Features

- Input SMS text and get real-time spam prediction
- Trained using a Naive Bayes classifier on public SMS dataset
- View model accuracy and sample data
- Lightweight and fast — perfect for learning or small projects

---

 How It Works
- The model is a Naive Bayes classifier trained on the SMS dataset.
- The text is converted into numerical features using CountVectorizer.
- It predicts whether a message is spam based on word patterns.

---

Backend (backend.py)
- Loads the dataset (sms_spam_dataset.csv)
- Trains a Naive Bayes classifier using text features
- Provides a function to predict if a given message is spam
- Returns the model’s accuracy

---

Frontend (frontend.py)
- Built with Streamlit for a fast, interactive UI
- Lets the user input a message
- Calls the backend to predict the label (spam or ham)
- Displays prediction and model accuracy
- Shows a sample of the original dataset

---

 Example
- Input: "Congratulations! You've won a $1000 Walmart gift card. Click here to claim."
- Output: SPAM


