import pyttsx3


def save_speech(speech: str, out_path: str):
    # Initialize the engine
    engine = pyttsx3.init()

    # Set the voice
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)  # You can change the index number to select a different voice

    # Convert text to speech and save to file
    engine.save_to_file(speech, out_path)

    # Play the speech
    engine.runAndWait()
    return out_path
