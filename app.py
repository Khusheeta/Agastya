from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Load your preprocessed data and trained model
df = pd.read_csv("Mental_Health_FAQ.csv")
tfidf_vectorizer = TfidfVectorizer()
tfidf_train = tfidf_vectorizer.fit_transform(df["Questions"])
le = LabelEncoder()
df["Answers_Code"] = le.fit_transform(df["Answers"])
mn = MultinomialNB()
mn.fit(tfidf_train, df["Answers_Code"])

# Initialize an empty list to store chat history
chat_history = []

# Function to process user input and generate chatbot response
def get_bot_response(user_input):
    test = [user_input]
    testing = tfidf_vectorizer.transform(test)
    response_code = mn.predict(testing)[0]
    answer_list = df["Answers"].unique().tolist()
    answer_index = df["Answers_Code"].unique().tolist().index(response_code)
    bot_response = answer_list[answer_index]
    return bot_response

# Route for home page
@app.route('/')
def home():
    return render_template('index.html', chat_history=chat_history)

# Route to handle chatbot requests
@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    # Get bot response
    bot_response = get_bot_response(user_input)
    # Add user and bot messages to chat history
    chat_history.append({'user': True, 'message': user_input})
    chat_history.append({'user': False, 'message': bot_response})
    return render_template('index.html', chat_history=chat_history)

if __name__ == '__main__':
    app.run(debug=True)
