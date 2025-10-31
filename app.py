"""
Script-to-Audio Master – Streamlit App
--------------------------------------
• Generates story scripts using OpenAI GPT models
• Converts them into MP3 audio via gTTS or OpenAI TTS
• Python 3.13 compatible with manual API key entry
• Includes “Inspire me” random prompt generator
"""

import os, sys, tempfile, random
from typing import Tuple, List
import streamlit as st

# --- Patch for Python 3.13+ (audioop removed) -------------------------
try:
    import audioop  # noqa
except ImportError:
    sys.modules["audioop"] = None

# --- AI & TTS libs ----------------------------------------------------
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
        "Enter your OpenAI API Key", type="password",
        help="Get yours at https://platform.openai.com/account/api-keys")
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

# --------------------------- Utilities -------------------------------
def call_openai_chat(prompt: str, system: str = None,
                     model: str = DEFAULT_MODEL, temperature: float = 0.8) -> str:
    if not OPENAI_API_KEY:
        st.error("Please enter a valid OpenAI API key.")
        raise RuntimeError("Missing key")
    if not openai:
        raise RuntimeError("openai library missing")

    messages = [{"role": "system", "content": system}] if system else []
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
            st.exception(e2); raise

def split_text_for_tts(text: str, max_chars: int = 4500) -> List[str]:
    if len(text) <= max_chars: return [text]
    import re
    sents = re.split(r'(?<=[.!?])\s+', text)
    out, cur = [], ""
    for s in sents:
        if len(cur)+len(s)+1 <= max_chars: cur = (cur+" "+s).strip()
        else: out.append(cur); cur = s
    if cur: out.append(cur)
    return out

def match_target_amplitude(sound: AudioSegment, target_dBFS: float) -> AudioSegment:
    try: return sound.apply_gain(target_dBFS - sound.dBFS)
    except Exception: return sound

def text_to_speech_gtts(text: str, lang="en", slow=False, tld="com") -> Tuple[str, bytes]:
    if gTTS is None: raise RuntimeError("gTTS not installed.")
    chunks = split_text_for_tts(text)
    segs = []
    for i,c in enumerate(chunks):
        tts = gTTS(text=c, lang=lang, slow=slow, tld=tld)
        tmp = tempfile.NamedTemporaryFile(suffix=f"_{i}.mp3", delete=False)
        tmp.close(); tts.save(tmp.name)
        segs.append(AudioSegment.from_file(tmp.name, format="mp3"))
    audio = segs[0]
    for s in segs[1:]: audio += s
    audio = match_target_amplitude(audio, -14.0)
    out = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    out.close(); audio.export(out.name, format="mp3")
    with open(out.name,"rb") as f: return out.name, f.read()

def text_to_speech_openai(text: str, voice="alloy") -> Tuple[str, bytes]:
    if not OPENAI_API_KEY: raise RuntimeError("Missing OpenAI key for TTS.")
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts", voice=voice, input=text) as r:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        r.stream_to_file(tmp.name); tmp.close()
        with open(tmp.name,"rb") as f: return tmp.name, f.read()

# --------------------------- Streamlit UI -----------------------------
st.title("🎙️ Script-to-Audio Master")
st.caption("Turn your ideas into AI-written, narrated storybooks!")

with st.sidebar:
    st.header("Story Configuration")

    # --- User prompt + random inspiration ---
    c1, c2 = st.columns([3,1])
    with c1:
        user_prompt = st.text_area(
            "✏️ Enter your story idea:",
            placeholder="Type anything: 'A robot who learns to paint sunsets.'",
            height=140)
    with c2:
        if st.button("🎲 Inspire me"):
            sample_prompts = [
                "A time-travelling baker who uses bread to change history.",
                "An astronaut cat lost on a candy planet.",
                "A shy dragon who opens a tea shop in a busy city.",
                "A mysterious letter that rewrites reality every midnight.",
                "A talking violin searching for its lost melody."
            ]
            user_prompt = random.choice(sample_prompts)
            st.session_state["prompt"] = user_prompt
            st.success(f"✨ Inspiration: {user_prompt}")

    genre = st.selectbox("Genre", ["Fantasy","Adventure","Sci-Fi","Mystery",
                                   "Drama","Comedy","Horror (PG-13)"])
    tone = st.selectbox("Tone", ["Warm / calm","Dramatic","Whimsical",
                                 "Suspenseful","Playful","Educational"])
    target_age = st.selectbox("Target age group",
        ["Children (3-7)","Kids (8-12)","Teenagers (13-17)","Adults"])
    story_length = st.selectbox("Length",
        ["Short (~400 w)","Medium (~800 w)","Long (~1500 w)"])
    language = st.selectbox("Language", ["en","tr","es","fr","de","it","pt"])
    voice_style = st.selectbox("Voice style",
        ["Neutral","Warm & Narrator","Energetic","Soft / Whisper","Deep & Resonant"])
    tts_engine = st.radio("TTS Engine", ["OpenAI TTS","Google gTTS"], index=0)
    tts_speed = st.checkbox("Slow narration (gTTS only)", value=False)
    model_choice = st.text_input("OpenAI model", value=DEFAULT_MODEL)
    temperature = st.slider("Creativity", 0.0, 1.2, 0.8)

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Generated Story Script")
    script_holder = st.empty()
    if st.button("Generate story"):
        user_prompt = user_prompt or st.session_state.get("prompt","")
        if not user_prompt.strip():
            st.warning("Please enter or generate a prompt first.")
        else:
            sys_msg = ("You are a professional storyteller. "
                       "Write a vivid story with a clear beginning, middle, and end. "
                       "Respect tone, genre, and age target. Return only the story text.")
            prompt = (f"Write a {story_length} {genre.lower()} story in {language}. "
                      f"Tone: {tone}. Audience: {target_age}. Voice style: {voice_style}. "
                      f"Prompt: {user_prompt}")
            with st.spinner("Generating story..."):
                story = call_openai_chat(prompt, system=sys_msg,
                                         model=model_choice, temperature=temperature)
                st.session_state["story"] = story
                script_holder.code(story, language="text")
                st.success("Story generated ✅")

    if "story" in st.session_state:
        script_holder.code(st.session_state["story"], language="text")
        st.download_button("Download script (.txt)",
            data=st.session_state["story"], file_name="story.txt", mime="text/plain")

with col2:
    st.subheader("Audio / TTS")
    if st.button("Convert to audio (MP3)"):
        if "story" not in st.session_state:
            st.warning("Generate a story first.")
        else:
            story = st.session_state["story"]
            with st.spinner(f"Converting via {tts_engine}..."):
                if tts_engine=="OpenAI TTS":
                    voice_map={"Neutral":"alloy","Warm & Narrator":"fable",
                               "Energetic":"nova","Soft / Whisper":"coral",
                               "Deep & Resonant":"echo"}
                    v=voice_map.get(voice_style,"alloy")
                    _,mp3=text_to_speech_openai(story,voice=v)
                else:
                    tld=("co.uk" if voice_style in ["Warm & Narrator","Deep & Resonant"]
                         else "com.au" if voice_style=="Energetic" else "com")
                    _,mp3=text_to_speech_gtts(story,lang=language,slow=tts_speed,tld=tld)
                st.audio(mp3, format="audio/mp3")
                st.download_button("Download audio (.mp3)", data=mp3,
                    file_name="story.mp3", mime="audio/mpeg")
                st.success("Audio created ✅")

st.markdown("---")
st.caption("💡 Tip: Click *🎲 Inspire me* if you need a creative idea to start!")
