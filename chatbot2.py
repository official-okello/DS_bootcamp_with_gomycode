import streamlit as st
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import string

# SpaCy model for preprocessing
nlp = spacy.load("en_core_web_sm")

# BERT model for sentence embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load and preprocess the text file
try:
    with open('../Text/american_football.txt', 'r', encoding='utf-8', errors='ignore') as f:
        data = f.read().replace('\n', ' ')
except FileNotFoundError:
    st.error("Error: File not found. Ensure '../Text/american_football.txt' is correctly placed.")

# Tokenizing text into sentences
sentences = nltk.sent_tokenize(data)

# Text preprocessing
def preprocess(sentence):
    doc = nlp((sentence).lower())
    words = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_digit]
    return " ".join(words)

# Preprocessing all sentences
processed_sentences = [preprocess(sentence) for sentence in sentences]

# Converting sentences to embeddings
sentence_embeddings = model.encode(processed_sentences)

# BERT-based sentence similarity function
def get_most_relevant_sentence(query):
    query_processed = preprocess(query)
    query_embedding = model.encode([query_processed])
    similarities = cosine_similarity(query_embedding, sentence_embeddings)
    best_match = sentences[similarities.argmax()]
    return best_match

# Chatbot function
def chatbot(question):
    return get_most_relevant_sentence(question)

# Streamlit application with chat interface
def main():
    st.title("Chatbot")
    st.write("Hello! I'm JB. Ask me anything about the topic in the text file.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.chat_input("Type your question...")

    if user_input:
        response = chatbot(user_input)
        st.session_state.messages.append(("You", user_input))
        st.session_state.messages.append(("Bot", response))

    for sender, msg in st.session_state.messages:
        st.chat_message(sender).write(msg)

if __name__ == "__main__":
    main()