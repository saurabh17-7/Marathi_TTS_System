# OCR → Marathi Text → Speech (USER-FRIENDLY + PROSODY) – FIXED VERSION

import customtkinter as ctk
from gtts import gTTS
import pygame, os, re, tempfile
import soundfile as sf
import numpy as np
from scipy import signal
import time, random
from threading import Thread
from tkinter import filedialog
from PIL import Image
import pytesseract
import pdfplumber
import cv2

pygame.mixer.init()

# -------------------- OCR CLEANING --------------------
def clean_marathi_ocr_text(text):
    corrections = {
        "मड": "माझी",
        "वे ": "चे ",
        "पी ": "ची ",
        "शाळेपी": "शाळेची",
        "शाळेवे": "शाळेचे",
        "अर्यत": "सुंदर",
        "|": "",
        "॥": "",
        "“": "",
        "”": "",
        "  ": " "
    }
    for k, v in corrections.items():
        text = text.replace(k, v)
    return text.strip()


# -------------------- Prosody --------------------
class ProsodyModifier:
    def apply(self, audio, level, emotion, voice=None, sr=22050):
        speed = 0.9 + (level / 100) * 0.3
        intensity = 0.9 + (level / 100) * 0.8
        volume = 0.85 + (level / 100) * 0.5
        rng = np.random.default_rng()

        # small natural jitter to avoid robotic monotone
        jitter = float(np.clip(rng.normal(1.0, 0.02), 0.95, 1.05))
        speed *= jitter
        intensity *= float(np.clip(rng.normal(1.0, 0.03), 0.9, 1.1))

        if emotion == "neutral":
            speed *= 0.95
            intensity *= 0.9
            volume *= 0.9
        elif emotion == "happy":
            speed *= 1.2
        elif emotion == "sad":
            speed *= 0.85
            intensity *= 0.7
        elif emotion == "angry":
            speed *= 1.15
            intensity *= 1.4
            volume *= 1.4
        elif emotion == "emotional":
            # slower, more dynamic, slightly reduced volume
            speed *= 0.9
            intensity *= 1.25
            volume *= 0.85
        elif emotion == "shocked":
            # quicker, louder, higher intensity
            speed *= 1.25
            intensity *= 1.3
            volume *= 1.2

        if voice:
            speed *= voice["speed"]
            intensity *= voice["intensity"]
            volume *= voice["volume"]

        # apply duration change
        new_len = max(1, int(len(audio) / speed))
        audio = signal.resample(audio, new_len)

        # intensity scaling (dynamic range)
        mean = np.mean(audio)
        audio = mean + (audio - mean) * intensity

        sr = 22050 if 'sr' not in globals() else globals().get('sr', 22050)

        # emotion specific spectral and modulation tweaks
        if emotion == "happy":
            # light vibrato and slight high-frequency lift
            vib_amp = 0.006
            vib_freq = 5.5
            t = np.arange(len(audio)) / sr
            audio = audio * (1 + vib_amp * np.sin(2 * np.pi * vib_freq * t))
            # gentle high frequency boost via FFT
            spec = np.fft.rfft(audio)
            freqs = np.fft.rfftfreq(len(audio), 1/sr)
            spec[freqs > 3000] *= 1.05
            audio = np.fft.irfft(spec)
        elif emotion == "sad":
            # reduce highs, add slight pause-like breathing (attenuation)
            spec = np.fft.rfft(audio)
            freqs = np.fft.rfftfreq(len(audio), 1/sr)
            spec[freqs > 3000] *= 0.8
            audio = np.fft.irfft(spec)
            # gentle tremolo
            trem_amp = 0.02
            trem_freq = 3.0
            t = np.arange(len(audio)) / sr
            audio = audio * (1 - trem_amp * (1 - np.cos(2 * np.pi * trem_freq * t)) / 2)
        elif emotion == "angry":
            # boost highs and add aggressive energy + slight clipping
            spec = np.fft.rfft(audio)
            freqs = np.fft.rfftfreq(len(audio), 1/sr)
            spec[freqs > 1500] *= 1.2
            audio = np.fft.irfft(spec)
            # small fast modulation
            vib_amp = 0.01
            vib_freq = 6.5
            t = np.arange(len(audio)) / sr
            audio = audio * (1 + vib_amp * np.sin(2 * np.pi * vib_freq * t))
            audio = np.tanh(audio * 1.6)
        elif emotion == "emotional":
            # breath before phrase + richer mid frequencies
            spec = np.fft.rfft(audio)
            freqs = np.fft.rfftfreq(len(audio), 1/sr)
            spec[(freqs > 400) & (freqs < 3500)] *= 1.08
            audio = np.fft.irfft(spec)
            # add subtle vibrato
            vib_amp = 0.008
            vib_freq = 5.0
            t = np.arange(len(audio)) / sr
            audio = audio * (1 + vib_amp * np.sin(2 * np.pi * vib_freq * t))
        elif emotion == "shocked":
            # quick, slightly higher pitch feeling via brighter HF boost
            spec = np.fft.rfft(audio)
            freqs = np.fft.rfftfreq(len(audio), 1/sr)
            spec[freqs > 2000] *= 1.25
            audio = np.fft.irfft(spec)

        # final volume
        audio = audio * volume
        return audio


# -------------------- Emotion --------------------
EMOTION_KEYWORDS = {
    "happy": ["आनंद", "छान", "मस्त", "यश", "खुश"],
    "sad": ["दुःख", "वाईट", "कष्ट", "त्रास", "ताण"],
    "angry": ["राग", "चिड", "नको"],
    "emotional": ["भावना", "भावन", "हृदय"],
    "shocked": ["आश्चर्य", "थक्क", "हैराण"]
}

# Map common emojis to emotion labels. GUI buttons will insert these emojis.
EMOJI_TO_EMOTION = {
    "😊": "happy",
    "🙂": "happy",
    "😄": "happy",
    "😂": "happy",
    "😢": "sad",
    "😭": "emotional",
    "😥": "sad",
    "😡": "angry",
    "😠": "angry",
    "😲": "shocked",
    "😮": "shocked",
    "😧": "shocked",
    "🥲": "emotional"
}

def tokenize(text):
    return re.findall(r'\w+', text.lower())

def detect_emotion(text, story=False):
    if story:
        return "neutral"
    words = tokenize(text)
    # Check for emoji-based emotion first (highest priority)
    for emo, lbl in EMOJI_TO_EMOTION.items():
        if emo in text:
            return lbl
    scores = {k: 0 for k in EMOTION_KEYWORDS}
    for e, keys in EMOTION_KEYWORDS.items():
        for k in keys:
            if k in words:
                scores[e] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] else "neutral"


# -------------------- Voices --------------------
VOICES = {
    "narration": {"speed": 0.95, "intensity": 0.9, "volume": 0.9},
    "dialogue": {"speed": 1.05, "intensity": 1.0, "volume": 1.05},
}

def split_story(text):
    segments = []
    for l in text.split("\n"):
        l = l.strip()
        if not l:
            continue
        if l.startswith(("\"", "“")):
            segments.append((l.strip("“”\""), "dialogue"))
        else:
            segments.append((l, "narration"))
    return segments


# -------------------- OCR --------------------
def extract_text_from_image(path):
    img = Image.open(path)
    text = pytesseract.image_to_string(img, lang="mar")
    return clean_marathi_ocr_text(text)

def extract_text_from_document(path):
    text = ""
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    elif path.endswith(".pdf"):
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                text += p.extract_text() or ""
    return text.strip()

def extract_text_from_camera():
    cap = cv2.VideoCapture(0)
    text = ""
    while True:
        ret, frame = cap.read()
        cv2.imshow("SPACE = capture | ESC = exit", frame)
        k = cv2.waitKey(1)
        if k == 32:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            text = pytesseract.image_to_string(img, lang="mar")
            text = clean_marathi_ocr_text(text)
            break
        elif k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    return text.strip()


# -------------------- Dialects --------------------
class MarathiDialect:
    def __init__(self, rules):
        self.rules = rules

    def apply(self, text):
        for k, v in self.rules.items():
            text = text.replace(k, v)
        return text

DIALECTS = {
    "Standard": MarathiDialect({}),
    "Varhadi": MarathiDialect({"आहे": "आय", "नाही": "नाय", "मी": "म्ही"}),
    "Malvani": MarathiDialect({"आहे": "आसा", "नाही": "ना", "मला": "माका"}),
    "Ahirani": MarathiDialect({"आहे": "हाय", "नाही": "नाय"}),
    "Kokani": MarathiDialect({"आहे": "आसा", "नाही": "ना", "मला": "माका"})
}


# -------------------- TTS --------------------
class MarathiTTS:
    def __init__(self):
        self.prosody = ProsodyModifier()
        self.temp = None

    def _make_silence(self, seconds, sr):
        n = max(0, int(seconds * sr))
        return np.zeros(n, dtype=np.float32)

    def _make_breath(self, seconds, sr):
        # simple breath: filtered white noise with fade-in/out
        n = max(1, int(seconds * sr))
        noise = np.random.normal(0, 1.0, n).astype(np.float32)
        # envelope (quick inhale, slower release)
        env = np.linspace(0.0, 1.0, int(0.15 * n))
        sustain_len = max(0, n - 2 * len(env))
        if sustain_len > 0:
            env = np.concatenate([env, np.ones(sustain_len), np.linspace(1.0, 0.0, len(env))])
        else:
            env = np.linspace(0.0, 0.0, n)
        breath = noise * env * 0.06
        # low-pass via convolution with a small Hann window
        win = np.hanning(101)
        win = win / win.sum()
        breath = np.convolve(breath, win, mode="same")
        return breath

    def _make_crying_sound(self, duration=1.5, sr=22050):
        """Generate a realistic crying sound effect"""
        t = np.arange(int(duration * sr)) / sr
        
        # Base crying frequency - varies between 300-500 Hz with wobbling
        base_freq = 400
        freq_variation = 80 * np.sin(2 * np.pi * 1.2 * t)  # Wobble at 1.2 Hz
        freq = base_freq + freq_variation
        
        # Phase accumulation for frequency modulation
        phase = 2 * np.pi * np.cumsum(freq) / sr
        
        # Basic sine wave with frequency variation
        crying = np.sin(phase)
        
        # Add some harmonics for richness
        crying += 0.4 * np.sin(2 * phase)  # 2nd harmonic
        crying += 0.2 * np.sin(3 * phase)  # 3rd harmonic
        
        # Modulation (amplitude envelope - breathing-like)
        amplitude = 0.6 + 0.3 * np.sin(2 * np.pi * 0.8 * t)
        
        # Add slight tremolo (quivering effect)
        tremolo = 1 + 0.15 * np.sin(2 * np.pi * 4.5 * t)
        
        # Apply envelopes
        cry_sound = crying * amplitude * tremolo
        
        # Fade in and fade out
        fade_in = np.linspace(0, 1, int(0.2 * sr))
        fade_out = np.linspace(1, 0, int(0.3 * sr))
        
        cry_sound[:len(fade_in)] *= fade_in
        cry_sound[-len(fade_out):] *= fade_out
        
        # Normalize
        cry_sound = cry_sound / np.max(np.abs(cry_sound)) * 0.7
        
        return cry_sound.astype(np.float32)

    def play_crying_sound(self):
        """Play a realistic crying sound effect"""
        try:
            sr = 22050
            crying = self._make_crying_sound(duration=1.5, sr=sr)
            
            # Save to temporary file
            cry_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            sf.write(cry_temp, crying, sr)
            
            # Play the crying sound
            pygame.mixer.music.load(cry_temp)
            pygame.mixer.music.play()
            
            # Clean up after playback
            import atexit
            atexit.register(lambda: os.remove(cry_temp) if os.path.exists(cry_temp) else None)
        except Exception as e:
            print(f"Error playing crying sound: {e}")

    def generate(self, text, level, story, forced_emotion=None):
        segments = split_story(text)
        audios = []

        for seg, seg_type in segments:
            emotion = forced_emotion if forced_emotion else detect_emotion(seg, story)
            voice = VOICES["dialogue"] if seg_type == "dialogue" else VOICES["narration"]

            mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
            wav = mp3.replace(".mp3", ".wav")

            gTTS(seg, lang="mr").save(mp3)
            audio, sr = sf.read(mp3)
            # ensure float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # apply prosody with sample rate
            audio = self.prosody.apply(audio, level, emotion, voice, sr)

            # optionally add a breath before certain emotions
            if emotion in ("emotional", "shocked"):
                breath_dur = 0.28 if emotion == "emotional" else 0.16
                breath = self._make_breath(breath_dur, sr)
                audio = np.concatenate([breath, audio])

            audios.append(audio)

            # short pause after each segment, tuned per emotion
            pause_map = {
                "neutral": 0.12,
                "happy": 0.08,
                "sad": 0.28,
                "angry": 0.06,
                "emotional": 0.35,
                "shocked": 0.18
            }
            pause_sec = pause_map.get(emotion, 0.12)
            audios.append(self._make_silence(pause_sec, sr))

            os.remove(mp3)

        final = np.concatenate(audios)
        self.temp = wav
        sf.write(wav, final, sr)

    def play(self):
        pygame.mixer.music.load(self.temp)
        pygame.mixer.music.play()

    def stop(self):
        pygame.mixer.music.stop()


# -------------------- GUI --------------------
class OCRTTSApp:
    def __init__(self):
        self.tts = MarathiTTS()
        self.original_text = ""
        self.text = ""

        self.win = ctk.CTk()
        self.win.geometry("700x550")
        self.win.title("Marathi OCR TTS")

        frame = ctk.CTkFrame(self.win)
        frame.pack(expand=True, fill="both", padx=20, pady=20)

        ctk.CTkButton(frame, text="📁 Image / Document", command=self.select_input).pack(pady=5)
        ctk.CTkButton(frame, text="📷 Camera OCR", command=self.camera_input).pack(pady=5)

        self.box = ctk.CTkTextbox(frame, height=250)
        self.box.pack(fill="x", pady=10)
        self.box.configure(state="disabled")

        self.story = ctk.BooleanVar()
        ctk.CTkCheckBox(frame, text="Story Mode", variable=self.story).pack()

        self.slider = ctk.CTkSlider(frame, from_=0, to=100)
        self.slider.set(50)
        self.slider.pack(fill="x", pady=10)

        # Dialect selector for Marathi variants
        self.dialect = ctk.StringVar(value="Standard")
        ctk.CTkOptionMenu(frame, values=list(DIALECTS.keys()), variable=self.dialect, command=lambda _: self.on_dialect_change()).pack()

        # Emoji selector: clicking sets the voice emotion for all pasted text
        self.forced_emotion = None
        self.emoji_buttons = {}
        self.emoji_button_emojis = {}
        
        # Title for emotion section
        emotion_title = ctk.CTkLabel(frame, text="Select Emotion:", font=("Arial", 12, "bold"))
        emotion_title.pack(pady=(8, 4))
        
        emoji_frame = ctk.CTkFrame(frame, fg_color="transparent")
        emoji_frame.pack(pady=6, fill="x")
        
        emoji_defs = [
            ("😊", "happy", "#FFD700"),      # Gold
            ("😢", "sad", "#87CEEB"),        # Sky Blue
            ("😡", "angry", "#FF6B6B"),      # Red
            ("😭", "emotional", "#FF69B4"),  # Hot Pink
            ("😲", "shocked", "#FFB6C1")     # Light Pink
        ]
        
        for em, lbl, color in emoji_defs:
            # Container for emoji button and label
            emoji_container = ctk.CTkFrame(emoji_frame, fg_color="transparent")
            emoji_container.pack(side="left", padx=8, expand=True, fill="both")
            
            # Circular emoji button with color background
            btn = ctk.CTkButton(
                emoji_container, 
                text=em, 
                width=70,
                height=70,
                font=("Arial", 32),
                fg_color=color,
                hover_color=color,
                text_color="#FFFFFF",
                border_width=3,
                border_color="#FFFFFF",
                command=lambda l=lbl, e=em: self.set_emotion(l, e)
            )
            btn.pack(pady=(0, 8))
            self.emoji_buttons[lbl] = btn
            self.emoji_button_emojis[lbl] = em
            
            # Emoji name below circle
            label = ctk.CTkLabel(
                emoji_container, 
                text=lbl.capitalize(),
                font=("Arial", 11, "bold"),
                text_color="#FFFFFF"
            )
            label.pack()
        
        # Auto button - circular
        auto_container = ctk.CTkFrame(emoji_frame, fg_color="transparent")
        auto_container.pack(side="left", padx=8, expand=True, fill="both")
        auto_btn = ctk.CTkButton(
            auto_container, 
            text="🔄",
            width=70,
            height=70,
            font=("Arial", 32),
            fg_color="#808080",
            hover_color="#A9A9A9",
            text_color="#FFFFFF",
            border_width=3,
            border_color="#FFFFFF",
            command=lambda: self.set_emotion(None, None)
        )
        auto_btn.pack(pady=(0, 8))
        auto_label = ctk.CTkLabel(
            auto_container, 
            text="Auto",
            font=("Arial", 11, "bold"),
            text_color="#FFFFFF"
        )
        auto_label.pack()
        
        self.selection_label = ctk.CTkLabel(frame, text="Emotion: Auto", font=("Arial", 11, "bold"))
        self.selection_label.pack(pady=4)

        self.speak_btn = ctk.CTkButton(frame, text="▶ Speak", command=self.speak)
        self.speak_btn.pack(pady=5)
        self.stop_btn = ctk.CTkButton(frame, text="⏹ Stop", command=self.stop_playback)
        self.stop_btn.pack()

    def update_ui(self, text, original=False):
        if original:
            self.original_text = text
            # Apply dialect transform for display only
            dialect = self.dialect.get()
            if dialect != "Standard":
                display_text = DIALECTS[dialect].apply(self.original_text)
            else:
                display_text = self.original_text
        else:
            display_text = text
        self.text = display_text
        self.box.configure(state="normal")
        self.box.delete("1.0", "end")
        self.box.insert("end", display_text)
        self.box.configure(state="disabled")

    def on_dialect_change(self):
        # Refresh displayed text when dialect selection changes
        if self.original_text:
            self.update_ui(self.original_text, original=True)

    def set_emotion(self, emotion_label, emoji=None):
        # emotion_label: one of 'happy','sad','angry','emotional','shocked' or None for auto
        self.forced_emotion = emotion_label
        if emotion_label:
            self.selection_label.configure(text=f"Emotion: {emotion_label.capitalize()}")
        else:
            self.selection_label.configure(text="Emotion: Auto")
        # update button text to indicate selection (wrap selected emoji in brackets)
        for lbl, btn in self.emoji_buttons.items():
            base = self.emoji_button_emojis.get(lbl, lbl)
            if lbl == emotion_label:
                btn.configure(text=f"[{base}]")
            else:
                btn.configure(text=base)

    def select_input(self):
        path = filedialog.askopenfilename(
            parent=self.win,
            filetypes=[("Images", "*.png *.jpg *.jpeg"), ("Docs", "*.pdf *.txt")]
        )
        if not path:
            return

        def task():
            if path.endswith((".png", ".jpg", ".jpeg")):
                text = extract_text_from_image(path)
            else:
                text = extract_text_from_document(path)

            self.win.after(0, lambda: self.update_ui(text, original=True))

        Thread(target=task, daemon=True).start()

    def camera_input(self):
        Thread(
            target=lambda: self.win.after(
                0, lambda: self.update_ui(extract_text_from_camera(), original=True)
            ),
            daemon=True
        ).start()

    def speak(self):
        if not self.original_text:
            return
        # disable Speak while generating/playing
        self.speak_btn.configure(state="disabled")

        level = self.slider.get()
        story = self.story.get()
        dialect = self.dialect.get()

        text = (
            self.original_text
            if dialect == "Standard"
            else DIALECTS[dialect].apply(self.original_text)
        )

        self.update_ui(text)

        def gen_and_play():
            try:
                self.tts.generate(text, level, story, self.forced_emotion)
                self.tts.play()
                # wait for playback to finish
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
            finally:
                # re-enable speak button when done
                try:
                    self.speak_btn.configure(state="normal")
                except Exception:
                    pass

        Thread(target=gen_and_play, daemon=True).start()

    def stop_playback(self):
        # stop audio and re-enable speak button
        try:
            self.tts.stop()
        finally:
            try:
                self.speak_btn.configure(state="normal")
            except Exception:
                pass

    def run(self):
        self.win.mainloop()


if __name__ == "__main__":
    OCRTTSApp().run()
