# Import Libraries
import streamlit as st
import speech_recognition as sr

# Function to Initialize the Speech Recognizer with the chosen API and language
def transcribe_speech(api_choice, language):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        st.info("Speak now...")
        audio_text = r.listen(source)

    try:
        if api_choice == "Google":
            return r.recognize_google(audio_text, language=language)
        elif api_choice == "Sphinx":
            return r.recognize_sphinx(audio_text, language=language)
        else:
            return "Selected API not supported."
    except sr.UnknownValueError:
        return "No speech detected. Try speaking clearly."
    except sr.RequestError:
        return "Network error. Check your internet connection."

# Streamlit App
def main():
    st.title("Enhanced Speech Recognition App")
    st.write("Click on the microphone to start speaking:")

    # Dropdown to choose speech recognition API
    api_choice = st.selectbox("Choose Speech Recognition API", ["Google", "Sphinx"])
    
    # Dropdown to choose the language
    language = st.selectbox("Choose Language", ["en-US", "fr-FR", "es-ES", "de-DE"])
    
    # Initialize session state for pause/resume functionality
    if "paused" not in st.session_state:
        st.session_state.paused = False

    # Buttons for pausing and resuming
    if st.button("Pause"):
        st.session_state.paused = True

    if st.button("Resume"):
        st.session_state.paused = False

    # Start Recording
    if st.button("Start Recording") and not st.session_state.paused:
        text = transcribe_speech(api_choice, language)
        st.write("Transcription:", text)

        # Allow user to save transcription
        if st.button("Save Transcription"):
            with open("transcription.txt", "w") as f:
                f.write(text)
            st.success("Transcription saved successfully!")

if __name__ == "__main__":
    main()