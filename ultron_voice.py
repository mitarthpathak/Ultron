#!/usr/bin/env python3
"""Ultron's voice: microphone -> speech to text -> safety check -> llama-server -> Piper -> sox -> speaker.

One-time setup (Ubuntu, or 64-bit Raspberry Pi OS):

    sudo apt install -y sox libsox-fmt-all libportaudio2 python3-venv
    python3 -m venv ~/ultron-venv
    ~/ultron-venv/bin/pip install faster-whisper sounddevice numpy piper-tts
    mkdir -p ~/ultron-voices
    ~/ultron-venv/bin/python -m piper.download_voices --download-dir ~/ultron-voices en_US-ryan-medium

Start llama-server with ultron_chat_template.jinja first (README step 4), then:

    ~/ultron-venv/bin/python ~/Ultron/ultron_voice.py

Press Enter, speak, then pause: it stops listening after about a second of silence.
Ultron speaks each sentence as soon as it is written. Ctrl+C quits.

    --text          type instead of speaking (test without a microphone)
    --hands-free    no Enter key: it listens whenever Ultron is not talking
    --crisis-line   your country's crisis line, spoken in the safety message
    --help          everything else (voice, effect, whisper model, microphone)

The first run downloads the speech recognition model (about 140 MB for base.en).
"""
import argparse
import json
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import safety_net  # noqa: E402

RATE = 16000  # Hz, what the speech recognizer expects
BLOCK = 480  # samples per microphone block: 30 ms
# What Whisper tends to "hear" in background noise. Ignored in --hands-free mode only.
NOISE_TEXT = {"", "you", "thank you", "thanks for watching", "thank you for watching"}
DEFAULT_EFFECT = "pitch -250 flanger 0 1 0 60 0.5 20 echo 0.8 0.8 6 0.35"


# ---------------------------------------------------------------- hearing

def cut_utterance(blocks, threshold, wait_blocks=None, silence_blocks=33, max_blocks=500, min_loud=6):
    """Cut one spoken utterance out of a stream of 30 ms float32 blocks.

    Waits for a block louder than `threshold` (giving up after `wait_blocks`, if set),
    keeps 0.3 s from before it, and stops after `silence_blocks` quiet blocks in a row
    or `max_blocks` in total. Clicks and bumps shorter than `min_loud` loud blocks are
    skipped. Returns a list of blocks, or None if nobody spoke in time.
    """
    import numpy as np

    pre, speech, waited = [], None, 0
    loud_count = quiet_run = 0
    for block in blocks:
        loud = float(np.sqrt(np.mean(block ** 2))) > threshold
        if speech is None:
            if loud:
                speech, loud_count, quiet_run = pre + [block], 1, 0
            else:
                pre = (pre + [block])[-10:]
                waited += 1
                if wait_blocks is not None and waited >= wait_blocks:
                    return None
            continue
        speech.append(block)
        loud_count += loud
        quiet_run = 0 if loud else quiet_run + 1
        if quiet_run >= silence_blocks or len(speech) >= max_blocks:
            if loud_count >= min_loud:
                return speech
            pre, speech = speech[-10:], None  # just a click: keep waiting
    return speech if speech and loud_count >= min_loud else None


class Microphone:
    def __init__(self, device=None):
        import numpy as np
        import sounddevice as sd

        self.np, self.sd, self.device = np, sd, device
        self.threshold = 0.01

    def calibrate(self, seconds=1.0):
        """Measure the room's background noise and set the speech threshold above it."""
        audio = self.sd.rec(int(RATE * seconds), samplerate=RATE, channels=1, dtype="float32", device=self.device)
        self.sd.wait()
        noise = float(self.np.sqrt(self.np.mean(audio ** 2)))
        self.threshold = max(noise * 3.5, 0.006)
        return noise

    def listen(self, wait_seconds=None):
        """Record one utterance as 16 kHz mono float32, or None if nobody spoke in time."""
        def blocks():
            with self.sd.InputStream(samplerate=RATE, channels=1, dtype="float32",
                                     blocksize=BLOCK, device=self.device) as stream:
                while True:
                    data, _ = stream.read(BLOCK)
                    yield data[:, 0].copy()

        wait_blocks = int(wait_seconds * RATE / BLOCK) if wait_seconds else None
        speech = cut_utterance(blocks(), self.threshold, wait_blocks)
        return self.np.concatenate(speech) if speech else None


class Ears:
    """Speech to text with faster-whisper on the CPU."""

    def __init__(self, model_name, language):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")
        self.language = language or None

    def transcribe(self, audio):
        segments, _ = self.model.transcribe(audio, language=self.language, beam_size=1,
                                            vad_filter=True, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments).strip()


# ---------------------------------------------------------------- speaking

class Speaker:
    """Speaks sentences in order on a background thread, so the next sentence can be
    written while this one plays. Piper makes the speech, sox adds the robot effect."""

    def __init__(self, voice_path, effect, length_scale, silent=False, save_dir=None):
        self.silent, self.save_dir, self.count = silent, save_dir, 0
        self.effect = effect.split() if effect else []
        self.play = shutil.which("play")
        self.aplay = shutil.which("aplay")
        self.queue = queue.Queue()
        if not silent:
            from piper import PiperVoice
            from piper.config import SynthesisConfig

            self.voice = PiperVoice.load(str(voice_path))
            self.config = SynthesisConfig(length_scale=length_scale)
            if not self.play and not save_dir:
                print("  sox isn't installed, so there's no robot effect (sudo apt install sox libsox-fmt-all)")
        threading.Thread(target=self._run, daemon=True).start()

    def say(self, text):
        self.queue.put(text)

    def wait(self):
        """Block until everything queued has been spoken."""
        self.queue.join()

    def _run(self):
        while True:
            text = self.queue.get()
            try:
                if not self.silent:
                    self._speak(text)
            except Exception as e:  # keep talking even if one sentence fails
                print(f"\n  [speech error: {e}]", flush=True)
            finally:
                self.queue.task_done()

    def _speak(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "say.wav"
            with wave.open(str(raw), "wb") as w:
                self.voice.synthesize_wav(text, w, syn_config=self.config)
            if self.save_dir:  # for testing without speakers
                self.count += 1
                out = Path(self.save_dir) / f"ultron_{self.count:03d}.wav"
                cmd = ["sox", str(raw), str(out)] + self.effect if shutil.which("sox") else ["cp", str(raw), str(out)]
                subprocess.run(cmd, check=True)
            elif self.play:
                subprocess.run([self.play, "-q", str(raw)] + self.effect, check=False)
            elif self.aplay:
                subprocess.run([self.aplay, "-q", str(raw)], check=False)


# ---------------------------------------------------------------- thinking

def stream_reply(url, messages, temperature, max_tokens):
    """Yield pieces of Ultron's reply as llama-server writes them."""
    body = json.dumps({"messages": messages, "stream": True, "temperature": temperature,
                       "top_p": 0.8, "top_k": 20, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            choices = json.loads(data).get("choices") or []
            piece = choices[0].get("delta", {}).get("content") if choices else None
            if piece:
                yield piece


SENTENCE = re.compile(r"(.+?[.!?…]+[\"')\]]*)\s+", re.S)


def sentences(pieces):
    """Group streamed text into whole sentences."""
    buf = ""
    for piece in pieces:
        buf += piece
        while (m := SENTENCE.match(buf)):
            yield m.group(1).strip()
            buf = buf[m.end():]
    if buf.strip():
        yield buf.strip()


def speakable(text):
    """Drop characters a voice can't say (markdown, emoji)."""
    text = re.sub(r"[*_#`~<>\[\]{}|]", "", text)
    text = re.sub(r"[\U0001F000-\U0001FAFF☀-➿]", "", text)
    return " ".join(text.split())


def server_ok(url):
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=5) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


# ---------------------------------------------------------------- main loop

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://localhost:8080", help="llama-server address")
    ap.add_argument("--text", action="store_true", help="type instead of speaking")
    ap.add_argument("--hands-free", action="store_true", help="listen whenever Ultron isn't talking")
    ap.add_argument("--voice", default=str(Path.home() / "ultron-voices" / "en_US-ryan-medium.onnx"),
                    help="Piper voice (.onnx, with its .onnx.json next to it)")
    ap.add_argument("--effect", default=DEFAULT_EFFECT, help="sox effects for the robot sound ('' for none)")
    ap.add_argument("--slow", type=float, default=1.1, help="speech speed: above 1 is slower")
    ap.add_argument("--no-voice", action="store_true", help="print replies without speaking them")
    ap.add_argument("--save-audio", metavar="DIR", help="write each spoken sentence to DIR instead of playing it")
    ap.add_argument("--whisper", default="base.en",
                    help="speech recognition model: tiny.en (fastest, for a Pi), base.en, small.en")
    ap.add_argument("--language", default="en", help="spoken language for multilingual whisper models ('' = detect)")
    ap.add_argument("--mic", default=None, help="microphone number or name (see --list-mics)")
    ap.add_argument("--list-mics", action="store_true", help="list audio devices and exit")
    ap.add_argument("--crisis-line", default=None,
                    help="spoken in the safety message, e.g. 'Tele MANAS on one four four one six'")
    ap.add_argument("--memory", type=int, default=6, help="past exchanges Ultron remembers")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=120)
    args = ap.parse_args()

    if args.list_mics:
        import sounddevice as sd
        print(sd.query_devices())
        return
    if not server_ok(args.server):
        sys.exit(f"llama-server isn't answering at {args.server}. Start it first (README step 4).")
    if not args.no_voice and not Path(args.voice).exists():
        sys.exit(f"Voice not found: {args.voice}\nDownload it with:\n  {sys.executable} -m piper.download_voices "
                 f"--download-dir {Path(args.voice).parent} {Path(args.voice).stem}")

    system = (HERE / "SYSTEM_PROMPT.txt").read_text(encoding="utf-8").strip()
    print("Loading voice...", flush=True)
    speaker = Speaker(args.voice, args.effect, args.slow, silent=args.no_voice, save_dir=args.save_audio)
    mic = ears = None
    if not args.text:
        print(f"Loading speech recognition ({args.whisper})...", flush=True)
        ears = Ears(args.whisper, args.language)
        mic = Microphone(int(args.mic) if args.mic and args.mic.isdigit() else args.mic)
        print("Measuring background noise, stay quiet for a second...", flush=True)
        noise = mic.calibrate()
        print(f"  noise level {noise:.4f}, speech threshold {mic.threshold:.4f}")

    history = []
    speaker.say("Online.")
    print("\nUltron is listening. Ctrl+C to quit.\n")
    while True:
        try:
            if args.text:
                text = input("You: ").strip()
            else:
                speaker.wait()  # never listen to his own voice
                if args.hands_free:
                    time.sleep(0.3)
                    audio = mic.listen()
                else:
                    input("[Enter, then speak] ")
                    print("  listening...", flush=True)
                    audio = mic.listen(wait_seconds=8)
                if audio is None:
                    print("  (heard nothing)")
                    continue
                t = time.time()
                text = ears.transcribe(audio)
                if args.hands_free and text.lower().strip(" .!?,") in NOISE_TEXT:
                    continue
                print(f"You: {text}   ({time.time() - t:.1f}s to transcribe)")
            if not text:
                continue

            crisis = safety_net.crisis_message(text, args.crisis_line)
            if crisis:
                # Out of character on purpose: the fixed message replaces the model's reply.
                print(f"Ultron: {crisis}")
                speaker.say(crisis)
                reply = crisis
            else:
                messages = ([{"role": "system", "content": system}] + history[-2 * args.memory:]
                            + [{"role": "user", "content": text}])
                print("Ultron: ", end="", flush=True)
                parts, t = [], time.time()
                for sentence in sentences(stream_reply(args.server, messages, args.temperature, args.max_tokens)):
                    sentence = speakable(sentence)
                    if sentence:
                        print(sentence, end=" ", flush=True)
                        speaker.say(sentence)
                        parts.append(sentence)
                print(f"  ({time.time() - t:.1f}s)")
                reply = " ".join(parts)
            history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        except (KeyboardInterrupt, EOFError):
            print("\nUltron: Offline. For now.")
            speaker.say("Offline. For now.")
            break
        except (urllib.error.URLError, OSError) as e:
            print(f"\n  can't reach llama-server at {args.server}: {e}")
            time.sleep(2)
    speaker.wait()


if __name__ == "__main__":
    main()
