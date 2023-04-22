import pyttsx3
# Initialize the engine
engine = pyttsx3.init()
# Set the voice
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # You can change the index number to select a different voice


def save_speech(speech: str, out_path: str)->str:

    # Convert text to speech and save to file
    engine.save_to_file(speech, out_path)

    # Play the speech
    engine.runAndWait()
    return out_path
def say_speech(speech:str,ind=1)->None:
    engine.setProperty('voice', voices[ind].id)  # You can change the index number to select a different voice
    # Convert text to speech and save to file
    engine.say(speech)

    # Play the speech
    engine.runAndWait()