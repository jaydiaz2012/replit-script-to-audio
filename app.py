"""
Script-to-Audio Master - app.py
Requirements:
  pip install streamlit openai gtts pydub

Notes:
- Expects OPENAI_API_KEY in environment (Replit Secrets: OPENAI_API_KEY).
- pydub uses ffmpeg for some operations. On Replit add ffmpeg to replit.nix or your environment if needed.
- This implementation uses gTTS for TTS (no cloud credentials required). If you prefer higher-quality voices
  (ElevenLabs, Google TTS, or OpenAI TTS), swap the text_to_speech() implementation and add credentials.
"""

import os
import tempfile
import time
from typing import Tuple, List

import streamlit as st

# AI and TTS libs
try:
    import openai
except Exception:
    openai = None

try:
    from gtts import gTTS
except Exception:
    gTTS = None

from pydub import AudioSegment  # used to normalize / concatenate if needed

# ---------------------------
# Configuration and helpers
# ---------------------------

st.set_page_config(page_title="Script-to-Audio Master", layout="wide")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if OPENAI_API_KEY and openai:
    # Support both new and classic openai clients defensively
    try:
        # classic openai python lib usage
        openai.api_key = OPENAI_API_KEY
    except Exception:
        # if using `from openai import OpenAI` variant, user library may differ;
        # we'll still call openai.ChatCompletion below when available.
        pass

# Default values
DEFAULT_MODEL = os.environ.get("SCRIPT_TO_AUDIO_MODEL", "gpt-4o-mini")  # fallback model name
MAX_TOKENS = 1200

# ---------------------------
# Utility functions
# ---------------------------

def call_openai_chat(prompt: str, system: str = None, model: str = DEFAULT_MODEL, temperature: float = 0.8) -> str:
    """
    Call the OpenAI chat-completion endpoint. Best-effort support for typical openai python SDKs.
    Returns the assistant text or raises an informative error.
    """
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY not found. Please set OPENAI_API_KEY in environment (Replit Secrets).")
        raise RuntimeError("Missing OPENAI_API_KEY")

    if not openai:
        st.error("The OpenAI python client library is not installed or failed to import.")
        raise RuntimeError("openai library missing")

    # Prepare message list
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # Try different calling styles depending on package version
    try:
        # Classic OpenAI library:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=MAX_TOKENS,
            n=1,
        )
        # Extract text
        text = resp["choices"][0]["message"]["content"]
        return text.strip()
    except Exception as e1:
        # Try new OpenAI client packaging (openai.OpenAI)
        try:
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=MAX_TOKENS)
            text = resp.choices[0].message.content
            return text.strip()
        except Exception as e2:
            st.error("OpenAI request failed. See console for details.")
            st.exception(e2)
            raise

def split_text_for_tts(text: str, max_chars: int = 4500) -> List[str]:
    """
    Split long text into chunks for gTTS (which can have limits). Splits on sentence boundaries roughly.
    """
    if len(text) <= max_chars:
        return [text]

    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    cur = ""
    for s in sentences:
        if len(cur) + len(s) + 1 <= max_chars:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return chunks

def text_to_speech_gtts(text: str, lang: str = "en", slow: bool = False, tld: str = "com") -> Tuple[str, bytes]:
    """
    Convert text to mp3 bytes using gTTS.
    Returns (tmp_mp3_path, mp3_bytes)
    """
    if gTTS is None:
        raise RuntimeError("gTTS is not installed. Please pip install gtts.")

    chunks = split_text_for_tts(text)
    segments = []
    # Create a temp file for each chunk and concatenate using pydub
    for i, chunk in enumerate(chunks):
        tts = gTTS(text=chunk, lang=lang, slow=slow, tld=tld)
        tmp = tempfile.NamedTemporaryFile(suffix=f"_{i}.mp3", delete=False)
        tmp_path = tmp.name
        tmp.close()
        tts.save(tmp_path)
        segments.append(AudioSegment.from_file(tmp_path, format="mp3"))

    if not segments:
        raise RuntimeError("No audio segments created.")

    # Concatenate
    combined = segments[0]
    for seg in segments[1:]:
        combined += seg

    # Optional: normalize volume
    normalized = match_target_amplitude(combined, -14.0)

    out_tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    out_path = out_tmp.name
    out_tmp.close()
    normalized.export(out_path, format="mp3")
    # Read bytes
    with open(out_path, "rb") as f:
        mp3_bytes = f.read()

    return out_path, mp3_bytes

def match_target_amplitude(sound: AudioSegment, target_dBFS: float) -> AudioSegment:
    """
    Normalize audio to target dBFS.
    """
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

# ---------------------------
# Streamlit UI
# ---------------------------

st.title("🎙️ Script-to-Audio Master")
st.caption("Generate story scripts with AI and convert them to audio storybooks (MP3).")

# Sidebar - parameters
with st.sidebar:
    st.header("Story Parameters")
    user_prompt = st.text_area("Story seed / prompt", value="A heartwarming fantasy adventure about a child who discovers a secret library in an old lighthouse.", height=150)
    genre = st.selectbox("Genre", ["Fantasy", "Adventure", "Sci-Fi", "Mystery", "Drama", "Comedy", "Horror (PG-13)"])
    tone = st.selectbox("Tone", ["Warm / calm", "Dramatic", "Whimsical", "Suspenseful", "Playful", "Educational"])
    target_age = st.selectbox("Target age group", ["Children (3-7)", "Kids (8-12)", "Teenagers (13-17)", "Adults"])
    story_length = st.selectbox("Approx length", ["Short (~300-500 words)", "Medium (~600-1000 words)", "Long (~1200-2000 words)"])
    language = st.selectbox("Story language", ["en", "tr", "es", "fr", "de", "it", "pt"])
    voice_style = st.selectbox("Voice style", ["Neutral", "Warm & Narrator", "Energetic", "Soft / Whisper", "Deep & Resonant"])
    tts_speed = st.checkbox("Slow narration (tts)", value=False)
    st.markdown("---")
    st.write("Model & API")
    model_choice = st.text_input("OpenAI model", value=DEFAULT_MODEL)
    temperature = st.slider("Creativity (temperature)", 0.0, 1.2, 0.8)
    st.write("🔒 Make sure OPENAI_API_KEY is set in environment (Replit Secrets).")

# Main actions
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Generated Story Script")
    script_holder = st.empty()

    gen_button = st.button("Generate story script", type="primary")

    if gen_button:
        # Build a structured prompt for consistent output
        len_map = {
            "Short (~300-500 words)": "short (about 350-500 words)",
            "Medium (~600-1000 words)": "medium length (about 700-1000 words)",
            "Long (~1200-2000 words)": "long (about 1200-1600 words)"
        }
        chosen_len = len_map.get(story_length, "medium length")

        system_msg = (
            "You are a professional children's story writer and narrator. "
            "Produce a well-structured story with a beginning, middle, and end. "
            "Use vivid descriptions, clear dialogue tags when appropriate, and respect the given tone, genre, and target age. "
            "Return only the story text (no extra metadata)."
        )

        prompt = (
            f"Write a {chosen_len} {genre.lower()} story in {language} with the following tone: {tone}. "
            f"The target audience is {target_age}. Use voice style: {voice_style}. "
            f"Seed / idea: {user_prompt}\n\n"
            "Please include scene breaks as blank lines where appropriate."
        )

        with st.spinner("Calling AI model to generate the script..."):
            try:
                story_text = call_openai_chat(prompt=prompt, system=system_msg, model=model_choice, temperature=temperature)
                script_holder.code(story_text, language="text")
                st.success("Story generated ✅")
                # Save current generated to session state for later use
                st.session_state["last_generated_story"] = story_text
            except Exception as e:
                st.error("Failed to generate story. Check the OpenAI API key and model settings.")
                st.exception(e)

    # Show existing session story if any
    if "last_generated_story" in st.session_state and not gen_button:
        script = st.session_state["last_generated_story"]
        script_holder.code(script, language="text")

    st.markdown("---")
    st.write("Save / Export")
    if "last_generated_story" in st.session_state:
        st.download_button("Download script (.txt)", data=st.session_state["last_generated_story"], file_name="story.txt", mime="text/plain")

with col2:
    st.subheader("Audio / TTS")
    convert_button = st.button("Convert latest script to audio (MP3)")

    if convert_button:
        if "last_generated_story" not in st.session_state:
            st.warning("Please generate a story script first.")
        else:
            story_text = st.session_state["last_generated_story"]
            with st.spinner("Converting text to speech..."):
                try:
                    # Map language selection to gTTS supported codes; we used language selectbox values already
                    lang_code = language
                    # gTTS TLD selection heuristic for voice style (not a real voice param, but gives some variety)
                    tld = "com"
                    if voice_style in ["Warm & Narrator", "Deep & Resonant"]:
                        tld = "co.uk"
                    elif voice_style == "Energetic":
                        tld = "com.au"
                    # Convert
                    mp3_path, mp3_bytes = text_to_speech_gtts(text=story_text, lang=lang_code, slow=tts_speed, tld=tld)
                    # Store in session
                    st.session_state["last_audio_path"] = mp3_path
                    st.session_state["last_audio_bytes"] = mp3_bytes
                    st.success("Audio created ✅")
                except Exception as e:
                    st.error("Text-to-speech failed. See details below.")
                    st.exception(e)

    if "last_audio_bytes" in st.session_state:
        st.audio(st.session_state["last_audio_bytes"], format="audio/mp3")
        st.download_button("Download audio (.mp3)", data=st.session_state["last_audio_bytes"], file_name="story.mp3", mime="audio/mpeg")
    else:
        st.info("No audio generated yet. Generate a script then click 'Convert latest script to audio'.")

st.markdown("---")
st.caption("Tip: For higher-quality, expressive TTS voices (voice casting, SSML, emotional speaking), integrate an external TTS provider like ElevenLabs, Google Cloud Text-to-Speech, or OpenAI TTS and replace the text_to_speech_gtts() implementation.")

# Optional: show debugging info only if dev mode
if st.checkbox("Show debug info"):
    st.subheader("Debug")
    st.write("OPENAI_API_KEY present:", bool(OPENAI_API_KEY))
    st.write("gTTS available:", gTTS is not None)
    st.write("openai module:", "present" if openai else "missing")
    if "last_audio_path" in st.session_state:
        st.write("Last audio path:", st.session_state["last_audio_path"])

