# Import Libraries
import streamlit as st
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import speech_recognition as sr

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Load BERT-based sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to Initialize the Speech Recognizer with the chosen API and language
def transcribe_speech(language="en-US"):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        st.info("Speak now...")
        audio_text = r.listen(source)

    try:
        return r.recognize_google(audio_text, language=language)
    except sr.UnknownValueError:
        return "No speech detected. Try speaking clearly."
    except sr.RequestError:
        return "Network error. Check your internet connection."

# Load and preprocess the text file
try:
    with open('../Text/american_football.txt', 'r', encoding='utf-8', errors='ignore') as f:
        data = f.read().replace('\n', ' ')
except FileNotFoundError:
    st.warning("Text file not found! Using default example text instead.")
    data = "American football is a popular sport played with an oval-shaped ball."

# Tokenizing text into sentences
sentences = nltk.sent_tokenize(data)

# Text preprocessing
def preprocess(sentence):
    doc = nlp(sentence.lower())
    words = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_digit]
    return " ".join(words)

# Preprocessing all sentences
processed_sentences = [preprocess(sentence) for sentence in sentences]

# Convert sentences to embeddings
sentence_embeddings = model.encode(processed_sentences)

# BERT-based sentence similarity function
def get_most_relevant_sentence(query):
    query_processed = preprocess(query)
    query_embedding = model.encode([query_processed])
    similarities = cosine_similarity(query_embedding, sentence_embeddings)
    
    best_match_idx = similarities.argmax()
    best_match_score = similarities[0][best_match_idx]

    # Introducing a similarity threshold
    if best_match_score > 0.5: 
        return sentences[best_match_idx]
    else:
        return "I couldn't find a strong match. Could you rephrase?"

# Chatbot function
def chatbot(question):
    return get_most_relevant_sentence(question)

# Streamlit application with chat interface
def main():
    st.title("Chatbot with Speech Recognition")
    st.write("Hello! I'm JB. You can ask me anything about the topic in the text file.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Dropdown for language selection
    language = st.selectbox("Choose Language", ["en-US", "fr-FR", "es-ES", "de-DE"])

    user_input = st.chat_input("Type your question...")

    # Process typed input
    if user_input:
        response = chatbot(user_input)
        st.session_state.messages.append(("You", user_input))
        st.session_state.messages.append(("Bot", response))

    # Allow speech-based queries
    if st.button("Use Speech"):
        spoken_query = transcribe_speech(language)
        response = chatbot(spoken_query)
        st.session_state.messages.append(("You (Spoken)", spoken_query))
        st.session_state.messages.append(("Bot", response))

    # Display chat messages
    for sender, msg in st.session_state.messages:
        st.chat_message(sender).write(msg)

if __name__ == "__main__":
    main()