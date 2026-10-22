import os
import tempfile
from gtts import gTTS
from groq import Groq


from pathlib import Path


def transcribe_audio(audio_bytes: bytes, filename: str = "recording.webm") -> str:
    """Convert user speech audio bytes into text using Groq Whisper."""
    if not audio_bytes:
        raise ValueError("Audio buffer is empty.")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured in environment variables."
        )

    client = Groq(api_key=api_key)

    suffix = Path(filename).suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio.write(audio_bytes)
        temp_path = temp_audio.name

    try:
        with open(temp_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(temp_path, file.read()),
                model="whisper-large-v3-turbo",
                response_format="text",
            )
        return str(transcription).strip()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _text_to_speech_groq(text: str, api_key: str) -> bytes:
    """
    Fast path: Groq's own TTS endpoint (playai-tts), running on the same
    infrastructure as the rest of this app's Groq calls — this is what
    replaces gTTS's slow round-trip to a separate third-party service.
    """

    client = Groq(api_key=api_key)
    voice = os.getenv("GROQ_TTS_VOICE", "Fritz-PlayAI")

    response = client.audio.speech.create(
        model="playai-tts",
        voice=voice,
        input=text,
        response_format="mp3",
    )

    # The Groq SDK's speech response exposes raw bytes via .read() /
    # .content depending on SDK version — try both rather than assume.
    if hasattr(response, "read"):
        return response.read()

    if hasattr(response, "content"):
        return response.content

    raise RuntimeError("Unrecognized response shape from Groq TTS.")


def _text_to_speech_gtts(text: str) -> bytes:
    """Fallback path: gTTS, used only if Groq TTS is unavailable."""

    tts = gTTS(text=text, lang="en")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_path = temp_file.name

    try:
        tts.save(temp_path)
        with open(temp_path, "rb") as file:
            return file.read()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def text_to_speech(text: str) -> bytes:
    """
    Convert text response into audio bytes.

    Tries Groq's TTS endpoint first (fast — same infra as the rest of
    this app's Groq calls). Falls back to gTTS only if the Groq call
    fails, so Listen keeps working even if Groq TTS isn't enabled for
    this account or has a transient error, rather than breaking outright.
    """

    if not text or not text.strip():
        raise ValueError("Text content is empty.")

    clean_text = text.strip()
    api_key = os.getenv("GROQ_API_KEY")

    if api_key:
        try:
            return _text_to_speech_groq(clean_text, api_key)
        except Exception:
            # Falls through to gTTS below — a slow response beats a
            # broken one, and a transient Groq TTS issue shouldn't take
            # down the whole feature.
            pass

    return _text_to_speech_gtts(clean_text)