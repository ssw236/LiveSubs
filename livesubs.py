"""
LiveSubs - real-time on-screen translated subtitles for whatever your PC is playing.
Windows only (captures system audio via WASAPI loopback, so no virtual cable needed).

Install:
    pip install -r requirements.txt
    GPU (optional, much faster): NVIDIA card + CUDA 12 cuBLAS / cuDNN 9
    (see the faster-whisper README). Without it, it runs on CPU.

Run:
    python livesubs.py                     # auto-detect language -> English subtitles
    python livesubs.py --lang ja           # tell it the source language (faster + more accurate)
    python livesubs.py --model large-v3    # best quality (needs a strong GPU)
    python livesubs.py --transcribe        # captions in the ORIGINAL language instead

Controls:
    drag the subtitle box to move it, mouse wheel = text size,
    right-click = menu (quit), Ctrl+C in the console also quits.
"""

import argparse
import queue
import signal
import sys
import threading
import time
from math import gcd

import numpy as np
import tkinter as tk
from scipy.signal import resample_poly

WHISPER_SR = 16000
BG = "#101010"

# Things Whisper likes to "hear" in near-silence / music.
HALLUCINATIONS = {
    "thank you.", "thanks for watching!", "thank you for watching.",
    "thanks for watching.", "you", "bye.", "please subscribe.",
    "subtitles by the amara.org community",
}


# --------------------------------------------------------------------------
# Audio capture (system output loopback)
# --------------------------------------------------------------------------
class LoopbackCapture:
    def __init__(self):
        import pyaudiowpatch as pyaudio

        self.pa = pyaudio
        self.p = pyaudio.PyAudio()
        self.q = queue.Queue()
        dev = self._find_loopback()
        self.name = dev["name"]
        self.sr = int(dev["defaultSampleRate"])
        self.ch = int(dev["maxInputChannels"])
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=self.ch,
            rate=self.sr,
            input=True,
            input_device_index=dev["index"],
            frames_per_buffer=int(self.sr * 0.05),
            stream_callback=self._cb,
        )

    def _find_loopback(self):
        try:
            wasapi = self.p.get_host_api_info_by_type(self.pa.paWASAPI)
        except OSError:
            raise RuntimeError("WASAPI not available - LiveSubs needs Windows.")
        spk = self.p.get_device_info_by_index(wasapi["defaultOutputDevice"])
        if spk.get("isLoopbackDevice"):
            return spk
        for lb in self.p.get_loopback_device_info_generator():
            if spk["name"] in lb["name"]:
                return lb
        raise RuntimeError(f"No loopback device found for '{spk['name']}'.")

    def _cb(self, data, frames, info, status):
        a = np.frombuffer(data, dtype=np.float32)
        if self.ch > 1:
            a = a.reshape(-1, self.ch).mean(axis=1)
        self.q.put(a.copy())
        return (None, self.pa.paContinue)

    def close(self):
        try:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Speech -> text/translation worker
# --------------------------------------------------------------------------
class Translator(threading.Thread):
    def __init__(self, args, ui_q):
        super().__init__(daemon=True)
        self.args = args
        self.ui_q = ui_q
        self.cap = None

    def _load_model(self):
        from faster_whisper import WhisperModel
        import ctranslate2

        a = self.args
        device = a.device
        if device == "auto":
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        name = a.model or ("medium" if device == "cuda" else "small")
        compute = "float16" if device == "cuda" else "int8"
        self.ui_q.put(("status", f"Loading {name} on {device.upper()}..."))
        try:
            return WhisperModel(name, device=device, compute_type=compute), name, device
        except Exception as e:
            if device != "cuda":
                raise
            print(f"CUDA load failed ({e}); falling back to CPU.")
            name = a.model or "small"
            self.ui_q.put(("status", f"GPU failed, loading {name} on CPU..."))
            return WhisperModel(name, device="cpu", compute_type="int8"), name, "cpu"

    def run(self):
        try:
            self._run()
        except Exception as e:
            self.ui_q.put(("error", str(e)))

    def _run(self):
        a = self.args
        model, name, device = self._load_model()
        self.cap = cap = LoopbackCapture()
        print(f"Model: {name} ({device}) | Capturing: {cap.name} @ {cap.sr} Hz")
        self.ui_q.put(("status", "Listening..."))

        sr = cap.sr
        g = gcd(sr, WHISPER_SR)
        up, down = WHISPER_SR // g, sr // g
        task = "transcribe" if a.transcribe else "translate"
        preroll = int(sr * 0.3)

        buf, voiced, last_voice, last_run = [], False, 0.0, 0.0

        while True:
            # Pull everything that has arrived.
            try:
                chunk = cap.q.get(timeout=0.1)
            except queue.Empty:
                chunk = None
            now = time.monotonic()
            while chunk is not None:
                buf.append(chunk)
                if np.sqrt(np.mean(chunk * chunk)) > a.threshold:
                    voiced, last_voice = True, now
                try:
                    chunk = cap.q.get_nowait()
                except queue.Empty:
                    chunk = None

            n = sum(len(b) for b in buf)
            if not voiced:
                # Nothing loud yet: keep only a short pre-roll.
                if n > preroll:
                    buf = [np.concatenate(buf)[-preroll:]]
                continue

            final = (now - last_voice) >= a.silence or n / sr >= a.max_seg
            if not final and now - last_run < a.step:
                continue
            last_run = now

            audio = np.concatenate(buf)
            buf = [audio]
            audio16 = resample_poly(audio, up, down).astype(np.float32)

            segs, info = model.transcribe(
                audio16,
                task=task,
                language=a.lang,
                beam_size=5 if final else 1,
                vad_filter=True,
                condition_on_previous_text=False,
                without_timestamps=True,
            )
            text = " ".join(s.text.strip() for s in segs).strip()
            if text.lower() in HALLUCINATIONS:
                text = ""

            if final:
                buf, voiced = [], False
                if text:
                    self.ui_q.put(("final", text, info.language))
                    print(f"[{info.language}] {text}")
            elif text:
                self.ui_q.put(("partial", text, info.language))


# --------------------------------------------------------------------------
# Subtitle overlay
# --------------------------------------------------------------------------
class Overlay:
    def __init__(self, root, ui_q, args):
        self.root, self.q, self.args = root, ui_q, args
        self.font_size = args.font
        self.final_text = ""
        self.last_activity = time.monotonic()
        self.hidden = False

        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", args.opacity)
        root.configure(bg=BG)

        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        self.w = int(sw * args.width)
        self.cx = sw // 2
        self.bottom = sh - int(sh * 0.08)

        self.prev = tk.Label(root, bg=BG, fg="#9a9a9a", justify="center",
                             wraplength=self.w - 40)
        self.cur = tk.Label(root, bg=BG, fg="#ffffff", justify="center",
                            wraplength=self.w - 40)
        self.prev.pack(fill="x", padx=20, pady=(10, 0))
        self.cur.pack(fill="x", padx=20, pady=(2, 12))
        self._apply_fonts()
        self.cur.config(text="Starting...")

        root.bind("<ButtonPress-1>", self._drag_start)
        root.bind("<B1-Motion>", self._drag_move)
        root.bind("<MouseWheel>", self._wheel)
        self.menu = tk.Menu(root, tearoff=0)
        self.menu.add_command(label="Clear", command=self._clear)
        self.menu.add_separator()
        self.menu.add_command(label="Quit", command=root.destroy)
        root.bind("<Button-3>", lambda e: self.menu.tk_popup(e.x_root, e.y_root))

        self._layout()
        root.after(50, self._poll)

    def _apply_fonts(self):
        self.prev.config(font=("Segoe UI", max(10, int(self.font_size * 0.75))))
        self.cur.config(font=("Segoe UI", self.font_size, "bold"))

    def _layout(self):
        self.root.update_idletasks()
        h = self.root.winfo_reqheight()
        x = self.cx - self.w // 2
        y = self.bottom - h
        self.root.geometry(f"{self.w}x{h}+{x}+{y}")

    def _show(self):
        self.last_activity = time.monotonic()
        if self.hidden:
            self.root.attributes("-alpha", self.args.opacity)
            self.hidden = False

    def _clear(self):
        self.final_text = ""
        self.prev.config(text="")
        self.cur.config(text="")
        self._layout()

    def _poll(self):
        changed = False
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "status":
                    self.cur.config(text=msg[1], fg="#9ad0ff")
                elif kind == "error":
                    self.cur.config(text=f"Error: {msg[1]}", fg="#ff7b7b")
                    print("Error:", msg[1])
                elif kind == "partial":
                    self.cur.config(text=msg[1], fg="#fff3b0")  # yellow = still listening
                elif kind == "final":
                    if self.final_text:
                        self.prev.config(text=self.final_text)
                    self.final_text = msg[1]
                    self.cur.config(text=msg[1], fg="#ffffff")
                changed = True
        except queue.Empty:
            pass

        if changed:
            self._show()
            self._layout()
        elif not self.hidden and self.args.hold > 0 and \
                time.monotonic() - self.last_activity > self.args.hold:
            self.root.attributes("-alpha", 0.0)  # fade out when nobody's talking
            self.hidden = True

        self.root.after(50, self._poll)

    def _drag_start(self, e):
        self._drag = (e.x_root, e.y_root, self.cx, self.bottom)

    def _drag_move(self, e):
        x0, y0, cx0, b0 = self._drag
        self.cx = cx0 + (e.x_root - x0)
        self.bottom = b0 + (e.y_root - y0)
        self._layout()

    def _wheel(self, e):
        self.font_size = max(12, min(72, self.font_size + (2 if e.delta > 0 else -2)))
        self._apply_fonts()
        self._layout()


# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Real-time translated subtitles for system audio.")
    p.add_argument("--model", help="tiny/base/small/medium/large-v3 (default: medium on GPU, small on CPU)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--lang", default=None, help="source language code, e.g. ja, ko, es (default: auto-detect)")
    p.add_argument("--transcribe", action="store_true", help="show original language instead of English")
    p.add_argument("--font", type=int, default=26, help="subtitle font size")
    p.add_argument("--opacity", type=float, default=0.82, help="box opacity 0-1")
    p.add_argument("--width", type=float, default=0.7, help="box width as fraction of screen")
    p.add_argument("--hold", type=float, default=7.0, help="hide after N seconds of silence (0 = never)")
    p.add_argument("--threshold", type=float, default=0.008, help="loudness that counts as speech")
    p.add_argument("--silence", type=float, default=0.7, help="pause (s) that ends a sentence")
    p.add_argument("--max-seg", type=float, default=8.0, help="force a line break after N seconds")
    p.add_argument("--step", type=float, default=0.8, help="live-preview refresh interval (s)")
    return p.parse_args()


def main():
    args = parse_args()
    if sys.platform == "win32":
        try:  # crisp text on high-DPI screens
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    ui_q = queue.Queue()
    root = tk.Tk()
    root.title("LiveSubs")
    Overlay(root, ui_q, args)
    worker = Translator(args, ui_q)
    worker.start()

    signal.signal(signal.SIGINT, lambda *_: root.destroy())
    root.mainloop()
    if worker.cap:
        worker.cap.close()


if __name__ == "__main__":
    main()
