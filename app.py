"""
Script-to-Audio Master  –  Streamlit App
----------------------------------------
• Generates story scripts with OpenAI GPT models
• Converts them to MP3 audio using gTTS or OpenAI TTS
• Fully compatible with Python 3.13+
• Supports manual API key entry
"""

import os
import sys
import tempfile
from typing import Tuple, List

import streamlit as st

# --- Patch for Python 3.13+ (audioop removed) -------------------------
try:
    import audioop  # noqa
except ImportError:
    sys.modules["audioop"] = None  # pydub fallback shim

# --- AI & TTS libraries -----------------------------------------------
try:
    import openai
except Exception:
    openai = None

try:
    from gtts import gTTS
except Exception:
    gTTS = None

from pydub import AudioSegment

# ---------------------------------------------------------------------
st.set_page_config(page_title="Script-to-Audio Master", layout="wide")

# --- API key handling -------------------------------------------------
if "OPENAI_API_KEY" not in os.environ:
    st.sidebar.warning("No OpenAI API Key found in environment.")
    input_key = st.sidebar.text_input(
        "Enter your OpenAI API Key",
        type="password",
        help="Get yours at https://platform.openai.com/account/api-keys"
    )
    if input_key:
        os.environ["OPENAI_API_KEY"] = input_key
        st.sidebar.success("✅ API key loaded for this session.")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

DEFAULT_MODEL = os.environ.get("SCRIPT_TO_AUDIO_MODEL", "gpt-4o-mini-tts")
MAX_TOKENS = 1200

if OPENAI_API_KEY and openai:
    try:
        openai.api_key = OPENAI_API_KEY
    except Exception:
        pass


# --------------------------- Utility functions ------------------------

def call_openai_chat(prompt: str, system: str = None,
                     model: str = DEFAULT_MODEL, temperature: float = 0.8) -> str:
    if not OPENAI_API_KEY:
        st.error("Please enter a valid OpenAI API key in the sidebar.")
        raise RuntimeError("Missing OPENAI_API_KEY")
    if not openai:
        st.error("openai library not installed.")
        raise RuntimeError("openai missing")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        resp = openai.ChatCompletion.create(
            model=model, messages=messages,
            temperature=temperature, max_tokens=MAX_TOKENS)
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e1:
        try:
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=MAX_TOKENS)
            return resp.choices[0].message.content.strip()
        except Exception as e2:
            st.exception(e2)
            raise e2


def split_text_for_tts(text: str, max_chars: int = 4500) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, cur = [], ""
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


def match_target_amplitude(sound: AudioSegment, target_dBFS: float) -> AudioSegment:
    try:
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)
    except Exception:
        return sound


def text_to_speech_gtts(text: str, lang: str = "en",
                        slow: bool = False, tld: str = "com") -> Tuple[str, bytes]:
    """Convert text to MP3 using Google gTTS."""
    if gTTS is None:
        raise RuntimeError("gTTS not installed. pip install gtts")

    chunks = split_text_for_tts(text)
    segments = []
    for i, chunk in enumerate(chunks):
        tts = gTTS(text=chunk, lang=lang, slow=slow, tld=tld)
        tmp = tempfile.NamedTemporaryFile(suffix=f"_{i}.mp3", delete=False)
        tmp.close()
        tts.save(tmp.name)
        segments.append(AudioSegment.from_file(tmp.name, format="mp3"))

    combined = segments[0]
    for seg in segments[1:]:
        combined += seg

    normalized = match_target_amplitude(combined, -14.0)
    out_tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    out_tmp.close()
    normalized.export(out_tmp.name, format="mp3")
    with open(out_tmp.name, "rb") as f:
        mp3_bytes = f.read()
    return out_tmp.name, mp3_bytes


def text_to_speech_openai(text: str, voice: str = "alloy") -> Tuple[str, bytes]:
    """Convert text to MP3 using OpenAI TTS API."""
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OpenAI API key for TTS.")
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        # Stream the response to save directly to a file
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text
        ) as response:
            tmp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            response.stream_to_file(tmp_file.name)
            tmp_file.close()
            with open(tmp_file.name, "rb") as f:
                mp3_bytes = f.read()
            return tmp_file.name, mp3_bytes
    except Exception as e:
        st.error("OpenAI TTS failed. Check API key or model availability.")
        st.exception(e)
        raise


# --------------------------- Streamlit UI ------------------------------

st.title("🎙️ Script-to-Audio Master")
st.caption("Generate AI stories and turn them into audio storybooks!")

with st.sidebar:
    st.header("Story Parameters")
    user_prompt = st.text_area(
        "Story seed / prompt",
        "A heartwarming fantasy adventure about a child who discovers a secret library in an old lighthouse.",
        height=150)
    genre = st.selectbox("Genre",
                         ["Fantasy", "Adventure", "Sci-Fi", "Mystery",
                          "Drama", "Comedy", "Horror (PG-13)"])
    tone = st.selectbox("Tone",
                        ["Warm / calm", "Dramatic", "Whimsical",
                         "Suspenseful", "Playful", "Educational"])
    target_age = st.selectbox("Target age group",
                              ["Children (3-7)", "Kids (8-12)",
                               "Teenagers (13-17)", "Adults"])
    story_length = st.selectbox("Length",
                                ["Short (~400 w)", "Medium (~800 w)", "Long (~1500 w)"])
    language = st.selectbox("Language", ["en", "tr", "es", "fr", "de", "it", "pt"])
    voice_style = st.selectbox("Voice style",
                              ["Neutral", "Warm & Narrator", "Energetic",
                                "Soft / Whisper", "Deep & Resonant"])
    tts_engine = st.radio("Select TTS Engine", ["OpenAI TTS", "Google gTTS"], index=0)
    tts_speed = st.checkbox("Slow narration (for gTTS only)", value=False)
    model_choice = st.text_input("OpenAI model", value=DEFAULT_MODEL)
    temperature = st.slider("Creativity", 0.0, 1.2, 0.8)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Generated Story Script")
    script_holder = st.empty()
    gen_button = st.button("Generate story")

    if gen_button:
        sys_msg = (
            "You are a professional story writer. "
            "Write a vivid story with clear beginning, middle and end. "
            "Use the given tone, genre and age target. "
            "Return only the story text."
        )
        prompt = (f"Write a {story_length} {genre.lower()} story in {language}. "
                  f"Tone: {tone}. Audience: {target_age}. "
                  f"Voice style: {voice_style}. Idea: {user_prompt}")
        with st.spinner("Generating story..."):
            try:
                story = call_openai_chat(prompt, system=sys_msg,
                                         model=model_choice, temperature=temperature)
                st.session_state["story"] = story
                script_holder.code(story, language="text")
                st.success("Story generated ✅")
            except Exception as e:
                st.exception(e)

    if "story" in st.session_state and not gen_button:
        script_holder.code(st.session_state["story"], language="text")

    if "story" in st.session_state:
        st.download_button("Download script (.txt)",
                           data=st.session_state["story"],
                           file_name="story.txt", mime="text/plain")

with col2:
    st.subheader("Audio / TTS")
    if st.button("Convert to audio (MP3)"):
        if "story" not in st.session_state:
            st.warning("Generate a story first.")
        else:
            story = st.session_state["story"]
            with st.spinner(f"Converting to speech using {tts_engine}..."):
                try:
                    if tts_engine == "OpenAI TTS":
                        # Select voice based on style
                        voice_map = {
                            "Neutral": "alloy",
                            "Warm & Narrator": "echo",
                            "Energetic": "coral",
                            "Soft / Whisper": "onyx",
                            "Deep & Resonant": "fable"
                        }
                        voice_choice = voice_map.get(voice_style, "alloy")
                        path, mp3_bytes = text_to_speech_openai(story, voice=voice_choice)
                    else:
                        tld = "co.uk" if voice_style in ["Warm & Narrator", "Deep & Resonant"] \
                            else "com.au" if voice_style == "Energetic" else "com"
                        path, mp3_bytes = text_to_speech_gtts(
                            story, lang=language, slow=tts_speed, tld=tld)

                    st.session_state["audio"] = mp3_bytes
                    st.audio(mp3_bytes, format="audio/mp3")
                    st.download_button("Download audio (.mp3)",
                                       data=mp3_bytes,
                                       file_name="story.mp3", mime="audio/mpeg")
                    st.success("Audio created ✅")
                except Exception as e:
                    st.exception(e)

st.markdown("---")
st.caption("Tip: Choose between OpenAI’s high-quality TTS or Google gTTS for multilingual voices.")
