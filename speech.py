import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_speech(text: str):
    speech_file_path = "speech.mp3"
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="onyx",
        input=text,
        instructions="Responda de forma empolgada como se você fosse um robo futurista"
    ) as response:
        response.stream_to_file(speech_file_path)
    return speech_file_path
