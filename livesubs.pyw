"""
LiveSubs - real-time subtitles for whatever your PC is playing.

Speech -> text:   SenseVoice (Chinese, Cantonese, Korean, Japanese, English - fast, runs on CPU)
                  or Whisper (any language)
Text -> English:  a local AI model (Qwen3, run by a built-in llama.cpp engine on your GPU) that translates for
                  natural meaning, using the previous lines as context

Everything runs offline on your PC after setup.

First time:  double-click "Install LiveSubs.bat"
After that:  open "LiveSubs" from your desktop or Start menu
"""

import collections
import itertools
import json
import os
import queue
import re
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import urllib.request
from math import gcd

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(APP_DIR, "settings.json")
LOG_PATH = os.path.join(APP_DIR, "livesubs.log")
ICON_PATH = os.path.join(APP_DIR, "livesubs.ico")
MODELS_DIR = os.path.join(APP_DIR, "models")
HISTORY_DIR = os.path.join(APP_DIR, "history")

# Launched without a console (from the shortcut): write output to a log file.
if sys.stdout is None or sys.stderr is None:
    _logf = open(LOG_PATH, "w", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = _logf
    if sys.stderr is None:
        sys.stderr = _logf

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
sys.coinit_flags = 0  # comtypes: use the multithreaded COM apartment (needed for per-app audio)

import numpy as np  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

WHISPER_SR = 16000
BG = "#101010"
DEFAULT_AUDIO = "Default speakers"

ENGINES = {
    "SenseVoice (Chinese / Korean / Japanese - fast)": "sensevoice",
    "Whisper large-v3 (all languages)": "whisper-large-v3",
    "Whisper medium (all languages)": "whisper-medium",
    "Whisper small (all languages, light)": "whisper-small",
}
DEVICES = {"Auto": "auto", "GPU": "cuda", "CPU": "cpu"}
TRANSLATORS = {
    "Qwen3 14B (best)": "qwen3-14b",
    "Qwen3 8B (faster)": "qwen3-8b",
    "Qwen3-VL 8B (faster, no wait after screenshots)": "qwen3-vl-8b",
    "Off": "off",
}
SHOW_MODES = {"Original + English": "both", "English only": "english", "Original only": "original"}

LANGUAGES = [
    ("Auto-detect", ""), ("Chinese (Mandarin)", "zh"), ("Cantonese", "yue"), ("Korean", "ko"),
    ("Japanese", "ja"), ("English", "en"), ("Spanish", "es"), ("French", "fr"), ("German", "de"),
    ("Russian", "ru"), ("Portuguese", "pt"), ("Italian", "it"), ("Arabic", "ar"), ("Hindi", "hi"),
    ("Thai", "th"), ("Vietnamese", "vi"), ("Indonesian", "id"), ("Turkish", "tr"),
    ("Polish", "pl"), ("Dutch", "nl"), ("Ukrainian", "uk"), ("Tagalog", "tl"),
    ("Hebrew", "he"), ("Greek", "el"),
]
LANG_NAME = {code: name for name, code in LANGUAGES if code}
SENSEVOICE_LANGS = {"zh", "yue", "ko", "ja", "en"}

# Windows fonts that actually contain each script's characters.
SCRIPT_FONTS = {
    "ja": "Yu Gothic UI", "zh": "Microsoft YaHei UI", "yue": "Microsoft JhengHei UI",
    "ko": "Malgun Gothic", "th": "Leelawadee UI", "lo": "Leelawadee UI",
    "km": "Leelawadee UI", "hi": "Nirmala UI", "bn": "Nirmala UI", "ta": "Nirmala UI",
    "te": "Nirmala UI", "mr": "Nirmala UI", "gu": "Nirmala UI", "pa": "Nirmala UI",
    "ne": "Nirmala UI", "si": "Nirmala UI", "am": "Ebrima", "my": "Myanmar Text",
}

# Things speech models "hear" in near-silence / music / breathing.
JUNK = {
    "thank you.", "thanks for watching!", "thank you for watching.", "thanks for watching.",
    "you", "bye.", "please subscribe.", "subtitles by the amara.org community",
    "ご視聴ありがとうございました", "시청해주셔서 감사합니다", "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目",
    "嗯", "啊", "呃", "哦", "唉", "嗯嗯", "음", "어", "아", "네", "うん", "ん", "え", "あ",
    "yeah", "uh", "um", "hmm", "oh",
}

DEFAULTS = {
    "engine": "sensevoice", "device": "auto", "translator": "qwen3-14b",
    "language": "", "show": "both", "audio_device": "", "app_mode": "all", "apps": [],
    "font": 20, "opacity": 0.75, "width": 0.5, "hold": 7.0,
    "threshold": 0.008, "silence": 0.6, "max_seg": 8.0,
    "click_through": False, "autostart": True, "pos": None,
    "history": 2, "speakers": True, "save_history": True, "layout_v": 2,
    "game_mode": True,
}

SV_NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
SV_URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{SV_NAME}.tar.bz2"
SV_DIR = os.path.join(MODELS_DIR, SV_NAME)

TRANSLATE_PROMPT = (
    "You are a professional subtitle translator for streaming shows, anime, dramas, games and "
    "livestreams. Translate the new line into English the way a native English speaker would "
    "actually say it in that situation - NOT word-for-word.\n"
    "Rules:\n"
    "- Match the register exactly: casual stays casual, rude stays rude, cute stays cute, "
    "formal stays formal. Use contractions and everyday spoken English.\n"
    "- Turn slang, internet slang and idioms into the natural English equivalent. Never translate "
    "an idiom or slang term literally.\n"
    "- Keep swearing and insults at the same strength - don't censor or soften them.\n"
    "- Drop filler and honorific noise that sounds unnatural in English (e.g. 啊/嘛/吧, 요/죠, ね/よ) "
    "but keep the feeling it adds.\n"
    "- The line comes from automatic speech recognition and may have wrong characters, "
    "homophones or missing words: translate what the speaker most likely MEANT.\n"
    "- Use the earlier lines only as context (who's talking, what's going on). Keep names as names.\n"
    "- Keep it short and readable like a real subtitle.\n"
    "\nExamples (literal -> how you should translate):\n"
    "你在搞什么鬼？  NOT \"What ghost are you doing?\" -> \"What the hell are you doing?\"\n"
    "我真是服了你了。  NOT \"I am really convinced by you.\" -> \"Unbelievable. You're something else.\"\n"
    "绝了！  NOT \"Extinct!\" -> \"That's insane!\"\n"
    "666  -> \"Nice!\"      yyds -> \"GOAT.\"      别装了 -> \"Quit acting.\"\n"
    "미쳤다 진짜  NOT \"Went crazy really\" -> \"That's insane, for real.\"\n"
    "아 진짜 짜증나  -> \"Ugh, so annoying.\"      대박 -> \"No way!\"      ㅋㅋㅋ -> \"lol\"\n"
    "マジでヤバいって！  NOT \"Seriously it is dangerous!\" -> \"Dude, this is seriously bad!\"\n"
    "うるさいな、黙ってろ。  -> \"Ugh, shut up already.\"      なんでやねん -> \"Oh, come on!\"\n"
)
FORMAT_ENGLISH = "\nReply with ONLY the English subtitle - no quotes, notes, labels or romanization."
FORMAT_FIX = (
    "\nReply in exactly this format and nothing else:\n"
    "ENGLISH: <the natural English subtitle>\n"
    "FIXED: <the new line in its original language with word spacing and obvious "
    "speech-recognition mistakes corrected; otherwise unchanged>"
)
FIX_LANGS = {"ko"}  # SenseVoice's Korean word spacing needs repairing; other languages don't

SCREEN_PROMPT = (
    "Read all the text in this screenshot (it may be game UI, a menu, chat, manga, a sign, "
    "a video frame...). Then translate it into natural, fluent English that conveys the meaning.\n"
    "Reply in exactly this format and nothing else:\n"
    "ORIGINAL:\n<the text exactly as written, keeping line breaks>\n"
    "ENGLISH:\n<the natural English translation, matching line breaks>\n"
    "If there is no readable text, reply: ORIGINAL:\n(none)\nENGLISH:\n(no text found)"
)
VISION_MODEL = "qwen3-vl-8b"

_line_ids = itertools.count(1)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


BELOW_NORMAL = 0x4000  # BELOW_NORMAL_PRIORITY_CLASS


def set_priority(pid, low):
    """Game mode: let the game win any fight for the CPU."""
    if sys.platform != "win32":
        return
    try:
        import psutil

        psutil.Process(pid).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if low else psutil.NORMAL_PRIORITY_CLASS)
    except Exception as e:
        log("Couldn't change priority:", e)


def load_settings():
    cfg = dict(DEFAULTS)
    saved = {}
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            saved = json.load(f)
        cfg.update({k: v for k, v in saved.items() if k in DEFAULTS})
    except Exception:
        pass
    if cfg["engine"] not in ENGINES.values():
        cfg["engine"] = DEFAULTS["engine"]
    if cfg["translator"] not in TRANSLATORS.values():
        cfg["translator"] = DEFAULTS["translator"]
    if saved and saved.get("layout_v", 1) < 2:  # older versions saved a much bigger caption box
        cfg.update(font=min(cfg["font"], 20), width=min(cfg["width"], 0.5),
                   history=min(cfg["history"], 2), layout_v=2)
    return cfg


def save_settings(cfg):
    try:
        tmp = SETTINGS_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        os.replace(tmp, SETTINGS_PATH)
    except Exception as e:
        log("Could not save settings:", e)


def is_junk(text):
    t = re.sub(r"[\s。，、！？!?,.…~～\-·]+", "", text.lower())
    return not t or t in {re.sub(r"[\s。，、！？!?,.…~～\-·]+", "", j) for j in JUNK}


def add_nvidia_dll_dirs():
    """Make DLLs from the pip packages nvidia-cublas-cu12 / nvidia-cudnn-cu12 findable."""
    if sys.platform != "win32":
        return
    import site

    roots = list(site.getsitepackages())
    try:
        roots.append(site.getusersitepackages())
    except Exception:
        pass
    for root in roots:
        base = os.path.join(root, "nvidia")
        if not os.path.isdir(base):
            continue
        for pkg in os.listdir(base):
            for sub in ("bin", "lib"):
                d = os.path.join(base, pkg, sub)
                if os.path.isdir(d):
                    try:
                        os.add_dll_directory(d)
                    except Exception:
                        pass
                    os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


add_nvidia_dll_dirs()


# --------------------------------------------------------------------------
# Audio capture (what your speakers/headphones are playing)
# --------------------------------------------------------------------------
def list_audio_devices():
    try:
        import pyaudiowpatch as pyaudio

        p = pyaudio.PyAudio()
        try:
            return [d["name"] for d in p.get_loopback_device_info_generator()]
        finally:
            p.terminate()
    except Exception as e:
        log("Could not list audio devices:", e)
        return []


class LoopbackCapture:
    def __init__(self, device_name=""):
        import pyaudiowpatch as pyaudio

        self.pa = pyaudio
        self.p = pyaudio.PyAudio()
        self.q = queue.Queue(maxsize=600)
        try:
            dev = self._find(device_name)
            self.name = dev["name"]
            self.sr = int(dev["defaultSampleRate"])
            self.ch = int(dev["maxInputChannels"])
            self.stream = self.p.open(
                format=pyaudio.paFloat32, channels=self.ch, rate=self.sr, input=True,
                input_device_index=dev["index"], frames_per_buffer=int(self.sr * 0.05),
                stream_callback=self._cb,
            )
        except Exception:
            self.p.terminate()
            raise

    def _find(self, name):
        loops = list(self.p.get_loopback_device_info_generator())
        if name:
            for lb in loops:
                if lb["name"] == name:
                    return lb
            log(f"Audio device '{name}' not found - using default speakers.")
        try:
            wasapi = self.p.get_host_api_info_by_type(self.pa.paWASAPI)
        except OSError:
            raise RuntimeError("WASAPI not available - LiveSubs needs Windows.")
        spk = self.p.get_device_info_by_index(wasapi["defaultOutputDevice"])
        if spk.get("isLoopbackDevice"):
            return spk
        for lb in loops:
            if spk["name"] in lb["name"]:
                return lb
        raise RuntimeError(f"Can't listen to '{spk['name']}'. Pick another device under 'Listen to'.")

    def _cb(self, data, frames, info, status):
        a = np.frombuffer(data, dtype=np.float32)
        if self.ch > 1:
            a = a.reshape(-1, self.ch).mean(axis=1)
        try:
            self.q.put_nowait(a.copy())
        except queue.Full:
            pass
        return (None, self.pa.paContinue)

    def close(self):
        try:
            self.stream.stop_stream()
            self.stream.close()
        except Exception:
            pass
        try:
            self.p.terminate()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Per-app audio (Windows 10 2004+ "process loopback")
# --------------------------------------------------------------------------
APP_MODES = {"All apps": "all", "All apps except...": "exclude", "Only these apps...": "include"}
APP_SR = 16000


def _app_key(name):
    name = (name or "").strip().lower()
    return name[:-4] if name.endswith(".exe") else name


def _com_init():
    try:
        import comtypes

        comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)
    except Exception:
        pass  # already initialised on this thread


def audio_sessions():
    """{pid: exe name} for every app with an audio session on the default output."""
    from pycaw.pycaw import AudioUtilities

    out = {}
    for s in AudioUtilities.GetAllSessions():
        try:
            if s.ProcessId and s.Process:
                out[s.ProcessId] = s.Process.name()
        except Exception:
            pass
    return out


def list_audio_apps():
    try:
        return sorted(set(audio_sessions().values()), key=str.lower)
    except Exception as e:
        log("Could not list audio apps:", e)
        return []


_PL = None


def _process_loopback():
    """Builds (once) the ctypes/comtypes bindings for process-loopback capture."""
    global _PL
    if _PL:
        return _PL
    import ctypes
    from ctypes import wintypes, POINTER, HRESULT
    import comtypes
    from comtypes import GUID, IUnknown, COMMETHOD, COMObject

    class IActivateAudioInterfaceAsyncOperation(IUnknown):
        _iid_ = GUID("{72A22D78-CDE4-431D-B8CC-843A71199B6D}")
        _methods_ = [COMMETHOD([], HRESULT, "GetActivateResult",
                               (["out"], POINTER(ctypes.c_long), "activateResult"),
                               (["out"], POINTER(POINTER(IUnknown)), "activatedInterface"))]

    class IActivateAudioInterfaceCompletionHandler(IUnknown):
        _iid_ = GUID("{41D949AB-9862-444A-80F6-C261334DA5EB}")
        _methods_ = [COMMETHOD([], HRESULT, "ActivateCompleted",
                               (["in"], POINTER(IActivateAudioInterfaceAsyncOperation), "activateOperation"))]

    class IAgileObject(IUnknown):  # without this, activation fails with E_ILLEGAL_METHOD_CALL
        _iid_ = GUID("{94EA2B94-E9CC-49E0-C0FF-EE64CA8F5B90}")
        _methods_ = []

    class WAVEFORMATEX(ctypes.Structure):
        _fields_ = [("wFormatTag", wintypes.WORD), ("nChannels", wintypes.WORD),
                    ("nSamplesPerSec", wintypes.DWORD), ("nAvgBytesPerSec", wintypes.DWORD),
                    ("nBlockAlign", wintypes.WORD), ("wBitsPerSample", wintypes.WORD),
                    ("cbSize", wintypes.WORD)]

    class IAudioClient(IUnknown):
        _iid_ = GUID("{1CB9AD4C-DBFA-4C32-B178-C2F568A703B2}")
        _methods_ = [
            COMMETHOD([], HRESULT, "Initialize", (["in"], wintypes.DWORD, "ShareMode"),
                      (["in"], wintypes.DWORD, "StreamFlags"), (["in"], ctypes.c_longlong, "BufferDuration"),
                      (["in"], ctypes.c_longlong, "Periodicity"), (["in"], POINTER(WAVEFORMATEX), "pFormat"),
                      (["in"], ctypes.c_void_p, "AudioSessionGuid")),
            COMMETHOD([], HRESULT, "GetBufferSize", (["out"], POINTER(ctypes.c_uint32), "n")),
            COMMETHOD([], HRESULT, "GetStreamLatency", (["out"], POINTER(ctypes.c_longlong), "l")),
            COMMETHOD([], HRESULT, "GetCurrentPadding", (["out"], POINTER(ctypes.c_uint32), "p")),
            COMMETHOD([], HRESULT, "IsFormatSupported", (["in"], wintypes.DWORD, "m"),
                      (["in"], POINTER(WAVEFORMATEX), "f"), (["out"], POINTER(POINTER(WAVEFORMATEX)), "c")),
            COMMETHOD([], HRESULT, "GetMixFormat", (["out"], POINTER(POINTER(WAVEFORMATEX)), "f")),
            COMMETHOD([], HRESULT, "GetDevicePeriod", (["out"], POINTER(ctypes.c_longlong), "d"),
                      (["out"], POINTER(ctypes.c_longlong), "m")),
            COMMETHOD([], HRESULT, "Start"),
            COMMETHOD([], HRESULT, "Stop"),
            COMMETHOD([], HRESULT, "Reset"),
            COMMETHOD([], HRESULT, "SetEventHandle", (["in"], wintypes.HANDLE, "h")),
            COMMETHOD([], HRESULT, "GetService", (["in"], POINTER(GUID), "riid"),
                      (["out"], POINTER(POINTER(IUnknown)), "ppv")),
        ]

    class IAudioCaptureClient(IUnknown):
        _iid_ = GUID("{C8ADBD64-E71E-48A0-A4DE-185C395CD317}")
        _methods_ = [
            COMMETHOD([], HRESULT, "GetBuffer", (["out"], POINTER(ctypes.c_void_p), "data"),
                      (["out"], POINTER(ctypes.c_uint32), "frames"), (["out"], POINTER(wintypes.DWORD), "flags"),
                      (["out"], POINTER(ctypes.c_ulonglong), "devpos"),
                      (["out"], POINTER(ctypes.c_ulonglong), "qpcpos")),
            COMMETHOD([], HRESULT, "ReleaseBuffer", (["in"], ctypes.c_uint32, "frames")),
            COMMETHOD([], HRESULT, "GetNextPacketSize", (["out"], POINTER(ctypes.c_uint32), "frames")),
        ]

    class LOOPBACK_PARAMS(ctypes.Structure):
        _fields_ = [("TargetProcessId", wintypes.DWORD), ("ProcessLoopbackMode", ctypes.c_int)]

    class ACTIVATION_PARAMS(ctypes.Structure):
        _fields_ = [("ActivationType", ctypes.c_int), ("ProcessLoopbackParams", LOOPBACK_PARAMS)]

    class BLOB(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.ULONG), ("pBlobData", ctypes.c_void_p)]

    class PROPVARIANT(ctypes.Structure):
        _fields_ = [("vt", ctypes.c_ushort), ("r1", ctypes.c_ushort), ("r2", ctypes.c_ushort),
                    ("r3", ctypes.c_ushort), ("blob", BLOB)]

    class Handler(COMObject):
        _com_interfaces_ = [IActivateAudioInterfaceCompletionHandler, IAgileObject]

        def __init__(self):
            super().__init__()
            self.done = threading.Event()

        def ActivateCompleted(self, activateOperation):
            self.done.set()
            return 0

    activate_fn = ctypes.WinDLL("Mmdevapi.dll").ActivateAudioInterfaceAsync
    activate_fn.restype = ctypes.c_long
    activate_fn.argtypes = [wintypes.LPCWSTR, POINTER(GUID), POINTER(PROPVARIANT),
                            POINTER(IActivateAudioInterfaceCompletionHandler),
                            POINTER(POINTER(IActivateAudioInterfaceAsyncOperation))]

    def open_client(pid):
        """Audio client capturing one process tree, as 16 kHz mono 16-bit PCM."""
        params = ACTIVATION_PARAMS(1, LOOPBACK_PARAMS(pid, 0))  # PROCESS_LOOPBACK, INCLUDE tree
        pv = PROPVARIANT()
        pv.vt = 65  # VT_BLOB
        pv.blob.cbSize = ctypes.sizeof(params)
        pv.blob.pBlobData = ctypes.cast(ctypes.pointer(params), ctypes.c_void_p)
        handler = Handler()
        op = POINTER(IActivateAudioInterfaceAsyncOperation)()
        hr = activate_fn("VAD\\Process_Loopback", ctypes.byref(IAudioClient._iid_), ctypes.byref(pv),
                         handler.QueryInterface(IActivateAudioInterfaceCompletionHandler), ctypes.byref(op))
        if hr < 0:
            raise OSError(f"ActivateAudioInterfaceAsync failed (0x{hr & 0xFFFFFFFF:08X}) - "
                          "per-app audio needs Windows 10 version 2004 or newer")
        if not handler.done.wait(5):
            raise TimeoutError("Windows didn't respond to the per-app audio request")
        hr2, unk = op.GetActivateResult()
        if hr2 < 0:
            raise OSError(f"Per-app audio activation failed (0x{hr2 & 0xFFFFFFFF:08X})")
        client = unk.QueryInterface(IAudioClient)
        fmt = WAVEFORMATEX(1, 1, APP_SR, APP_SR * 2, 2, 16, 0)
        # LOOPBACK | EVENTCALLBACK | AUTOCONVERTPCM | SRC_DEFAULT_QUALITY
        client.Initialize(0, 0x00020000 | 0x00040000 | 0x80000000 | 0x08000000, 2_000_000, 0,
                          ctypes.byref(fmt), None)
        capture = client.GetService(ctypes.byref(IAudioCaptureClient._iid_)).QueryInterface(IAudioCaptureClient)
        return client, capture

    _PL = {"open_client": open_client, "comtypes": comtypes}
    return _PL


class AppStream(threading.Thread):
    """Captures one app's audio into a small buffer."""

    def __init__(self, pid, name):
        super().__init__(daemon=True)
        self.pid, self.name = pid, name
        self.buf = collections.deque()
        self.size = 0
        self.lock = threading.Lock()
        self.stop_evt = threading.Event()
        self.failed = None

    def take(self, n):
        out = np.zeros(n, np.float32)
        with self.lock:
            # drop anything older than ~1 s so a stalled app can't add delay
            while self.size > APP_SR and self.buf:
                self.size -= len(self.buf.popleft())
            i = 0
            while i < n and self.buf:
                chunk = self.buf[0]
                k = min(n - i, len(chunk))
                out[i:i + k] = chunk[:k]
                i += k
                if k == len(chunk):
                    self.buf.popleft()
                else:
                    self.buf[0] = chunk[k:]
                self.size -= k
        return out

    def run(self):
        import ctypes
        from ctypes import wintypes

        _com_init()
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateEventW.restype = wintypes.HANDLE  # 64-bit handles: don't truncate
        kernel32.CreateEventW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        event = None
        try:
            client, capture = _process_loopback()["open_client"](self.pid)
            event = kernel32.CreateEventW(None, False, False, None)
            client.SetEventHandle(event)
            client.Start()
            log(f"Capturing app: {self.name} (pid {self.pid})")
            while not self.stop_evt.is_set():
                kernel32.WaitForSingleObject(event, 100)
                while True:
                    if not capture.GetNextPacketSize():
                        break
                    data, frames, flags, _, _ = capture.GetBuffer()
                    if frames:
                        if flags & 0x2 or not data:  # AUDCLNT_BUFFERFLAGS_SILENT
                            arr = np.zeros(frames, np.float32)
                        else:
                            raw = (ctypes.c_char * (frames * 2)).from_address(data)
                            arr = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
                        with self.lock:
                            self.buf.append(arr)
                            self.size += len(arr)
                    capture.ReleaseBuffer(frames)
            client.Stop()
        except Exception as e:
            self.failed = str(e)
            log(f"App capture failed for {self.name}: {e}")
        finally:
            if event:
                kernel32.CloseHandle(event)

    def stop(self):
        self.stop_evt.set()


class AppMixCapture:
    """Listens to only the chosen apps (or every app except the chosen ones), mixed together.
    Same interface as LoopbackCapture: .name, .sr, .q, .close()."""

    def __init__(self, mode, apps, on_error=log):
        self.on_error = on_error
        self.failed = {}  # pid -> time it failed (don't retry for a while)
        if sys.platform != "win32":
            raise RuntimeError("Per-app audio needs Windows.")
        self.mode = mode
        self.apps = {_app_key(a) for a in apps}
        if mode == "include" and not self.apps:
            raise RuntimeError("No apps chosen - click 'Apps...' and tick the ones to listen to.")
        listed = ", ".join(sorted(self.apps)) or "nothing"
        self.name = f"only {listed}" if mode == "include" else f"all apps except {listed}"
        self.sr = APP_SR
        self.q = queue.Queue(maxsize=600)
        self.streams = {}
        self.stop_evt = threading.Event()
        _process_loopback()  # fail early if this Windows can't do it
        threading.Thread(target=self._manage, daemon=True).start()
        threading.Thread(target=self._mix, daemon=True).start()

    def _wanted(self, sessions):
        import psutil

        me = os.getpid()
        pids = {pid for pid, name in sessions.items()
                if pid != me and ((_app_key(name) in self.apps) == (self.mode == "include"))}
        # an app's helper processes are captured with their parent - don't record them twice
        out = {}
        for pid in pids:
            try:
                parents = {p.pid for p in psutil.Process(pid).parents()}
            except Exception:
                parents = set()
            if not parents & pids:
                out[pid] = sessions[pid]
        return out

    def _manage(self):
        _com_init()
        while not self.stop_evt.is_set():
            try:
                wanted = self._wanted(audio_sessions())
                now = time.monotonic()
                for pid in list(self.streams):
                    s = self.streams[pid]
                    if s.failed and not s.is_alive():
                        self.failed[pid] = now
                        self.on_error(f"Can't capture {s.name}: {s.failed}")
                    if pid not in wanted or (s.failed and not s.is_alive()):
                        s.stop()
                        del self.streams[pid]
                for pid, name in wanted.items():
                    if pid not in self.streams and now - self.failed.get(pid, -1e9) > 30:
                        s = AppStream(pid, name)
                        self.streams[pid] = s
                        s.start()
            except Exception as e:
                log("App audio scan failed:", e)
            self.stop_evt.wait(2.0)

    def _mix(self):
        block = APP_SR // 20  # 50 ms
        next_t = time.monotonic()
        while not self.stop_evt.is_set():
            next_t += 0.05
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()
            mix = np.zeros(block, np.float32)
            for s in list(self.streams.values()):
                mix += s.take(block)
            np.clip(mix, -1.0, 1.0, out=mix)
            try:
                self.q.put_nowait(mix)
            except queue.Full:
                pass

    def close(self):
        self.stop_evt.set()
        for s in list(self.streams.values()):
            s.stop()


def open_capture(cfg, on_error=log):
    mode = cfg.get("app_mode", "all")
    if mode == "all":
        return LoopbackCapture(cfg.get("audio_device", ""))
    return AppMixCapture(mode, cfg.get("apps", []), on_error)


# --------------------------------------------------------------------------
# Speech -> text engines
# --------------------------------------------------------------------------
def download(url, dest, status, label):
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, last = 0, -1
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                if pct // 5 != last // 5:
                    last = pct
                    status(f"Downloading {label}... {pct}%")
    os.replace(tmp, dest)


def ensure_sensevoice(status):
    model, tokens = os.path.join(SV_DIR, "model.int8.onnx"), os.path.join(SV_DIR, "tokens.txt")
    if os.path.exists(model) and os.path.exists(tokens):
        return model, tokens
    os.makedirs(MODELS_DIR, exist_ok=True)
    archive = os.path.join(MODELS_DIR, SV_NAME + ".tar.bz2")
    download(SV_URL, archive, status, "SenseVoice (first time only)")
    status("Unpacking SenseVoice...")
    with tarfile.open(archive, "r:bz2") as t:
        members = [m for m in t.getmembers() if m.name.endswith(("model.int8.onnx", "tokens.txt"))]
        try:
            t.extractall(MODELS_DIR, members=members, filter="data")
        except TypeError:
            t.extractall(MODELS_DIR, members=members)
    os.remove(archive)
    return model, tokens


SV_GPU_CACHE = os.path.join(MODELS_DIR, "sensevoice-gpu-check.json")


def _console_python():
    """python.exe next to pythonw.exe (the GPU check needs a real stdout)."""
    exe = sys.executable
    alt = os.path.join(os.path.dirname(exe), "python.exe")
    return alt if exe.lower().endswith("pythonw.exe") and os.path.exists(alt) else exe


def sensevoice_gpu_ok(status=log):
    """Tests the GPU speech engine in a separate process, so a broken GPU setup can't crash LiveSubs.
    The answer is remembered until the speech engine is updated."""
    try:
        import sherpa_onnx

        version = getattr(sherpa_onnx, "__version__", "?")
    except Exception:
        return False
    if "cuda" not in version:  # the normal (CPU-only) build is installed
        return False
    try:
        with open(SV_GPU_CACHE, encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("version") == version:
            return bool(cached.get("ok"))
    except Exception:
        pass
    status("Testing the GPU speech engine (first time only)...")
    ok, detail = False, ""
    try:
        r = subprocess.run([_console_python(), os.path.abspath(__file__), "--probe-sv-gpu"],
                           capture_output=True, text=True, timeout=180, cwd=APP_DIR,
                           creationflags=0x08000000 if sys.platform == "win32" else 0)
        out = (r.stdout or "").strip().splitlines()
        res = json.loads(out[-1]) if out else {}
        detail = (r.stderr or "")[-1500:]
        fell_back = "fallback to cpu" in detail.lower() or "fall back to cpu" in detail.lower()
        ok = r.returncode == 0 and res.get("ok") and not fell_back
        log(f"SenseVoice GPU check: ok={ok} result={res} rc={r.returncode}")
        if not ok and detail:
            log("GPU check output:", detail)
    except Exception as e:
        log("SenseVoice GPU check failed:", e)
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(SV_GPU_CACHE, "w", encoding="utf-8") as f:
            json.dump({"version": version, "ok": ok}, f)
    except Exception:
        pass
    return ok


def probe_sensevoice_gpu():
    """Child process for sensevoice_gpu_ok(): load on CUDA, decode, report timing."""
    import sherpa_onnx

    model, tokens = ensure_sensevoice(lambda s: None)
    rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=model, tokens=tokens, num_threads=1, use_itn=True, language="auto", provider="cuda")
    audio = (np.random.default_rng(0).normal(0, 0.05, WHISPER_SR * 5)).astype(np.float32)
    times = []
    for _ in range(3):
        t0 = time.monotonic()
        st = rec.create_stream()
        st.accept_waveform(WHISPER_SR, audio)
        rec.decode_stream(st)
        times.append((time.monotonic() - t0) * 1000)
    print(json.dumps({"ok": True, "ms": round(min(times), 1)}), flush=True)


class SenseVoiceASR:
    kind = "sensevoice"
    name = "SenseVoice"
    live_step = 0.4

    def __init__(self, status, game_mode=True, device="auto"):
        import sherpa_onnx

        model, tokens = ensure_sensevoice(status)
        use_gpu = device in ("auto", "cuda") and sensevoice_gpu_ok(status)
        if device == "cuda" and not use_gpu:
            status("GPU speech engine not available - using CPU (run the installer to add it)")
            log("SenseVoice: GPU requested but not available; using CPU")
        self.device = "cuda" if use_gpu else "cpu"
        status(f"Loading SenseVoice on {'GPU' if use_gpu else 'CPU'}...")
        # on the CPU, 2 threads is plenty for real-time speech; more just steals cores from the game
        threads = 1 if use_gpu else (2 if game_mode else max(2, min(8, (os.cpu_count() or 4) // 2)))
        self.rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model, tokens=tokens, num_threads=threads, use_itn=True, language="auto",
            provider="cuda" if use_gpu else "cpu",
        )
        self.transcribe(np.zeros(WHISPER_SR, np.float32), "", True)  # warm-up

    def transcribe(self, audio16, language, final):
        s = self.rec.create_stream()
        s.accept_waveform(WHISPER_SR, audio16)
        self.rec.decode_stream(s)
        lang = (s.result.lang or "").strip("<|>") or "zh"
        text = s.result.text.strip()
        return ("" if is_junk(text) else text), lang


class WhisperASR:
    kind = "whisper"
    live_step = 0.8

    def __init__(self, size, device, status):
        from faster_whisper import WhisperModel
        from faster_whisper.utils import download_model
        import ctranslate2

        if device == "auto":
            try:
                device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                device = "cpu"
        try:  # use the copy already on disk; no internet check
            path = download_model(size, local_files_only=True)
        except Exception:
            status(f"Downloading Whisper {size} (first time only)...")
            path = download_model(size)

        last = None
        for dev in (["cuda", "cpu"] if device == "cuda" else ["cpu"]):
            try:
                status(f"Loading Whisper {size} on {'GPU' if dev == 'cuda' else 'CPU'}...")
                self.model = WhisperModel(path, device=dev, compute_type="float16" if dev == "cuda" else "int8")
                self.device, self.name = dev, f"Whisper {size}"
                self.transcribe(np.zeros(WHISPER_SR, np.float32), "en", False)  # loads GPU libraries now
                return
            except Exception as e:
                last = e
                log(f"Whisper {size} on {dev} failed: {e}")
                if dev == "cuda":
                    status("GPU failed - using CPU instead...")
        raise last

    def _run(self, audio16, task, language, final):
        segs, info = self.model.transcribe(
            audio16, task=task, language=language or None, beam_size=5 if final else 1,
            vad_filter=True, condition_on_previous_text=False, without_timestamps=True,
        )
        text = " ".join(s.text.strip() for s in segs).strip()
        return ("" if is_junk(text) else text), info.language

    def transcribe(self, audio16, language, final):
        return self._run(audio16, "transcribe", language, final)

    def translate(self, audio16, language):
        return self._run(audio16, "translate", language, True)[0]


def load_asr(engine, device, status, game_mode=True):
    if engine == "sensevoice":
        return SenseVoiceASR(status, game_mode, device)
    return WhisperASR(engine.split("-", 1)[1], device, status)


# --------------------------------------------------------------------------
# Who's speaking (voice fingerprints, ~30 ms per line on the CPU)
# --------------------------------------------------------------------------
SPK_FILE = "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
SPK_URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/{SPK_FILE}"
SPEAKER_COLORS = ["#7fd1ff", "#ffb86b", "#9be38a", "#ff8fc8", "#c9a2ff", "#ffe066"]


def ensure_speaker_model(status):
    path = os.path.join(MODELS_DIR, SPK_FILE)
    if not os.path.exists(path):
        os.makedirs(MODELS_DIR, exist_ok=True)
        download(SPK_URL, path, status, "speaker model (first time only)")
    return path


class SpeakerTracker:
    """Groups lines by voice: same voice -> same speaker number."""
    SAME = 0.5        # similarity above this = same person
    MIN_SECONDS = 0.8  # shorter clips are too short to tell voices apart reliably

    def __init__(self, status):
        import sherpa_onnx

        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=ensure_speaker_model(status), num_threads=2)
        self.ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.centroids, self.counts, self.last = [], [], None

    def identify(self, audio16):
        """Returns a speaker number (0, 1, 2...) or None if unsure."""
        try:
            with self.lock:
                if len(audio16) < WHISPER_SR * self.MIN_SECONDS:
                    return None
                s = self.ex.create_stream()
                s.accept_waveform(WHISPER_SR, audio16)
                s.input_finished()
                e = np.asarray(self.ex.compute(s), np.float32)
                e /= np.linalg.norm(e) + 1e-9
                sims = [float(c @ e) for c in self.centroids]
                best = int(np.argmax(sims)) if sims else -1
                if best >= 0 and (sims[best] >= self.SAME or len(self.centroids) >= len(SPEAKER_COLORS)):
                    n = self.counts[best] = min(self.counts[best] + 1, 20)
                    c = self.centroids[best] * (n - 1) + e
                    self.centroids[best] = c / (np.linalg.norm(c) + 1e-9)
                else:
                    self.centroids.append(e)
                    self.counts.append(1)
                    best = len(self.centroids) - 1
                self.last = best
                return best
        except Exception as ex:
            log("Speaker check failed:", ex)
            return None


# --------------------------------------------------------------------------
# Local AI engine (llama.cpp, bundled in the LiveSubs folder - no separate app)
# --------------------------------------------------------------------------
LLAMA_TAG = "b12017"
_REL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}"
LLAMA_DIR = os.path.join(MODELS_DIR, "llama")
# Tried in order. Each is pinned to its GPU device so it can never silently fall back to the CPU.
LLAMA_BUILDS = [
    ("cuda", "CUDA0", [f"{_REL}/llama-{LLAMA_TAG}-bin-win-cuda-13.4-x64.zip",
                       f"{_REL}/cudart-llama-bin-win-cuda-13.4-x64.zip"]),
    ("vulkan", "Vulkan0", [f"{_REL}/llama-{LLAMA_TAG}-bin-win-vulkan-x64.zip"]),
]
LLM_MODELS = {
    "qwen3-14b": {"label": "Qwen3 14B", "repo": "Qwen/Qwen3-14B-GGUF",
                  "file": "Qwen3-14B-Q4_K_M.gguf", "size": "9 GB"},
    "qwen3-8b": {"label": "Qwen3 8B", "repo": "Qwen/Qwen3-8B-GGUF",
                 "file": "Qwen3-8B-Q4_K_M.gguf", "size": "5 GB"},
    "qwen3-vl-8b": {"label": "Qwen3-VL 8B", "repo": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
                    "file": "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
                    "mmproj": "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf", "size": "6 GB"},
}
LLM_LOG = os.path.join(APP_DIR, "ai-engine.log")


def llama_exe(backend):
    return os.path.join(LLAMA_DIR, backend, "llama-server.exe")


def ensure_llama_builds(status):
    import zipfile

    for backend, _dev, urls in LLAMA_BUILDS:
        if os.path.exists(llama_exe(backend)):
            continue
        dest = os.path.join(LLAMA_DIR, backend)
        os.makedirs(dest, exist_ok=True)
        for url in urls:
            zpath = os.path.join(LLAMA_DIR, url.rsplit("/", 1)[1])
            download(url, zpath, status, f"AI engine ({backend})")
            with zipfile.ZipFile(zpath) as z:
                z.extractall(dest)
            os.remove(zpath)


def model_files(key):
    m = LLM_MODELS[key]
    return [m["file"]] + ([m["mmproj"]] if m.get("mmproj") else [])


def model_ready(key):
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in model_files(key))


def ensure_model(key, status):
    from huggingface_hub import hf_hub_download

    m = LLM_MODELS[key]
    for f in model_files(key):
        if not os.path.exists(os.path.join(MODELS_DIR, f)):
            status(f"Downloading {m['label']} ({f}, resumes if interrupted)...")
            hf_hub_download(m["repo"], f, local_dir=MODELS_DIR)


def _free_port():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


_JOB = None


def _kill_with_us(proc):
    """Windows job object: the AI engine is closed automatically if LiveSubs exits or crashes."""
    global _JOB
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

        class BASIC(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class EXTENDED(ctypes.Structure):
            _fields_ = [("Basic", BASIC), ("IoInfo", ctypes.c_uint64 * 6),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        if not _JOB:
            job = k32.CreateJobObjectW(None, None)
            info = EXTENDED()
            info.Basic.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
            _JOB = job
        k32.AssignProcessToJobObject(_JOB, int(proc._handle))
    except Exception as e:
        log("Couldn't tie the AI engine to LiveSubs:", e)


class LlamaHost:
    """Runs one hidden llama-server at a time and swaps models when asked for a different one."""

    def __init__(self):
        self.proc = None
        self.key = None
        self.port = None
        self.backend = None  # remembered once one works
        self.lock = threading.RLock()
        self.game_mode = True
        self.lean = True  # thread-saving options; turned off automatically if this engine build rejects them

    def running(self, key=None):
        return self.proc is not None and self.proc.poll() is None and (key is None or key == self.key)

    def use(self, key, status=log):
        with self.lock:
            if self.running(key):
                return
            self.stop()
            label = LLM_MODELS[key]["label"]
            if not model_ready(key):
                raise RuntimeError(f"{label} isn't downloaded - run 'Install LiveSubs.bat'.")
            builds = sorted(LLAMA_BUILDS, key=lambda b: b[0] != self.backend)
            errors = []
            for backend, device, _urls in builds:
                exe = llama_exe(backend)
                if not os.path.exists(exe):
                    continue
                status(f"Loading {label} on the GPU...")
                try:
                    try:
                        self._start(exe, device, key)
                    except Exception as first:
                        if not self.lean or not self._rejected_option():
                            raise
                        log(f"Engine didn't accept the thread options ({first}); retrying without them")
                        self.stop()
                        self.lean = False
                        self._start(exe, device, key)
                    self.backend = backend
                    log(f"AI engine ready: {label} via {backend}")
                    return
                except Exception as e:
                    errors.append(f"{backend}: {e}")
                    log(f"AI engine ({backend}) failed: {e}")
                    self.stop()
            if not errors:
                raise RuntimeError("AI engine isn't downloaded - run 'Install LiveSubs.bat'.")
            raise RuntimeError("AI engine couldn't use the GPU (" + "; ".join(errors) + ")")

    def _start(self, exe, device, key):
        m = LLM_MODELS[key]
        port = _free_port()
        ctx = "8192" if m.get("mmproj") else "4096"  # subtitles are short; less VRAM left over for the game
        args = [exe, "-m", os.path.join(MODELS_DIR, m["file"]), "--host", "127.0.0.1", "--port", str(port),
                "-dev", device, "-ngl", "all", "-c", ctx, "-np", "1", "--no-webui",
                "--reasoning-budget", "0"]
        if m.get("mmproj"):
            args += ["--mmproj", os.path.join(MODELS_DIR, m["mmproj"])]
        if self.lean:
            # everything runs on the GPU, so the CPU side only needs a couple of threads,
            # and --poll 0 stops them spinning at 100% while waiting
            args += ["-t", "2", "-tb", "2", "--poll", "0"]
        logf = open(LLM_LOG, "ab")
        logf.write(f"\n===== {time.ctime()} {' '.join(args)}\n".encode())
        logf.flush()
        flags = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW
        if sys.platform == "win32" and self.game_mode:
            flags |= BELOW_NORMAL
        self.proc = subprocess.Popen(args, cwd=os.path.dirname(exe), stdout=logf, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, creationflags=flags)
        logf.close()
        _kill_with_us(self.proc)
        self.port, self.key = port, key
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"stopped with code {self.proc.returncode} (details in ai-engine.log)")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:
                pass  # 503 while the model is still loading
            time.sleep(0.4)
        raise TimeoutError("took too long to load")

    @staticmethod
    def _rejected_option():
        try:
            with open(LLM_LOG, "rb") as f:
                f.seek(max(0, os.path.getsize(LLM_LOG) - 4000))
                tail = f.read().decode("utf-8", "replace").lower()
            return any(w in tail for w in ("invalid argument", "unknown argument", "error: unrecognized"))
        except Exception:
            return False

    def set_game_mode(self, on):
        self.game_mode = on
        if self.running():
            set_priority(self.proc.pid, on)

    def chat(self, key, messages, max_tokens=300, temperature=0.2, timeout=60, status=log, on_text=None):
        """on_text(text_so_far) is called as the reply streams in, if given."""
        with self.lock:
            self.use(key, status)
            body = json.dumps({
                "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                "chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": True,
                "stream": on_text is not None,
            }).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if on_text is None:
                    res = json.loads(r.read().decode())
                    return res["choices"][0]["message"].get("content") or ""
                acc = ""
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        delta = json.loads(data)["choices"][0].get("delta", {}).get("content") or ""
                    except Exception:
                        continue
                    if delta:
                        acc += delta
                        on_text(acc)
                return acc

    def stop(self):
        with self.lock:
            p, self.proc, self.key = self.proc, None, None
            if p and p.poll() is None:
                p.terminate()
                try:
                    p.wait(5)
                except Exception:
                    p.kill()


HOST = LlamaHost()


def _unquote(text):
    text = text.strip()
    if len(text) > 1 and text[0] in "\"'“「" and text[-1] in "\"'”」":
        text = text[1:-1].strip()
    return text


def parse_translation(text, original):
    """Returns (fixed_original, english). Handles 'ENGLISH:/FIXED:' replies and plain replies."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    eng_m = re.search(r"ENGLISH:\s*(.*?)\s*(?=\n\s*FIXED:|$)", text, flags=re.S | re.I)
    fix_m = re.search(r"FIXED:\s*(.*?)\s*(?=\n\s*ENGLISH:|$)", text, flags=re.S | re.I)
    if eng_m:
        eng = eng_m.group(1)
    elif fix_m:  # only a FIXED line came back
        eng = text[:fix_m.start()]
    else:
        eng = text
    eng = _unquote(eng.strip().split("\n\n")[0])
    fixed = _unquote(fix_m.group(1).strip().split("\n")[0]) if fix_m else ""
    # Only accept the "fix" if it's a light touch-up, never a rewrite.
    if not fixed or abs(len(fixed.replace(" ", "")) - len(original.replace(" ", ""))) > max(3, len(original) // 4):
        fixed = original
    return fixed, eng


def english_so_far(text):
    """The English part of a reply that's still streaming in."""
    text = re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.S).lstrip()
    if "ENGLISH:".startswith(text.upper()):  # the label itself is still arriving
        return ""
    text = re.sub(r"^ENGLISH:\s*", "", text, flags=re.I).lstrip("\"'“「")
    text = re.split(r"\n\s*F(?:I(?:X(?:E(?:D(?::)?)?)?)?)?\s*$|\n\s*FIXED:", text, flags=re.I)[0]
    return _unquote(text.strip().split("\n\n")[0]) if text.strip() else ""


def parse_screen(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"ORIGINAL:\s*(.*?)\s*ENGLISH:\s*(.*)", text, flags=re.S | re.I)
    if not m:
        return "", text
    return m.group(1).strip(), m.group(2).strip()


class Translator(threading.Thread):
    """Translates in the background so subtitles never wait on it.
    Finished lines come first; when idle it previews the line still being spoken."""

    def __init__(self, cfg, out):
        super().__init__(daemon=True)
        self.cfg, self.out = cfg, out
        self.inbox = queue.Queue()
        self.history = collections.deque(maxlen=3)
        self.ready_model = None  # model name once loaded on the GPU
        self.error = None
        self.plock = threading.Lock()
        self.preview_slot = None           # (utt_id, text, lang) - newest partial only
        self.previewed = {}                # utt_id -> text length last previewed
        self.finished = collections.deque(maxlen=20)

    def prepare(self, status=log):
        """Load the translation model onto the GPU (call from a thread). Returns an error or None."""
        model = self.cfg["translator"]
        self.ready_model = None
        if model == "off":
            return None
        try:
            if not model_ready(model):
                ensure_model(model, status)  # e.g. picked "Qwen3 8B" for the first time
            HOST.use(model, status)
            self.translate_text("你好", "zh", [])  # warm-up
            self.ready_model, self.error = model, None
            return None
        except Exception as e:
            self.error = str(e)
            log(traceback.format_exc())
            return self.error

    def _messages(self, text, lang, context, fix):
        ctx = "\n".join(context) if context else "(none)"
        user = f"Earlier lines:\n{ctx}\n\nNew line ({LANG_NAME.get(lang, lang)}):\n{text}"
        return [{"role": "system", "content": TRANSLATE_PROMPT + (FORMAT_FIX if fix else FORMAT_ENGLISH)},
                {"role": "user", "content": user + "\n/no_think"}]

    def translate_text(self, text, lang, context, on_english=None, fix=None):
        """Returns (fixed_original, english). on_english(text) gets the English as it streams in."""
        fix = (lang in FIX_LANGS) if fix is None else fix
        last = [""]

        def stream(acc):
            eng = english_so_far(acc)
            if eng and eng != last[0]:
                last[0] = eng
                on_english(eng)

        reply = HOST.chat(self.cfg["translator"], self._messages(text, lang, context, fix),
                          max_tokens=250 if fix else 120, temperature=0.3, timeout=60,
                          on_text=stream if on_english else None)
        return parse_translation(reply, text)

    def submit(self, line_id, text, lang):
        self.finished.append(line_id)
        with self.plock:
            if self.preview_slot and self.preview_slot[0] == line_id:
                self.preview_slot = None
        self.inbox.put((line_id, text, lang))

    def preview(self, utt_id, text, lang):
        """Offer a still-being-spoken line for a quick early translation."""
        if not self.ready_model or utt_id in self.finished:
            return
        if len(text) - self.previewed.get(utt_id, 0) < 3:  # wait until it's grown a bit
            return
        with self.plock:
            self.preview_slot = (utt_id, text, lang)

    def _do_preview(self):
        with self.plock:
            item, self.preview_slot = self.preview_slot, None
        if not item or not self.ready_model:
            return
        utt_id, text, lang = item
        if utt_id in self.finished:
            return
        self.previewed[utt_id] = len(text)
        if len(self.previewed) > 50:
            self.previewed.pop(next(iter(self.previewed)))
        try:
            _, eng = self.translate_text(text, lang, list(self.history), fix=False)
            if eng and utt_id not in self.finished:
                self.out.put(("live_eng", utt_id, eng))
        except Exception as e:
            log("Preview translation failed:", e)

    def run(self):
        while True:
            try:
                line_id, text, lang = self.inbox.get(timeout=0.05)
            except queue.Empty:
                self._do_preview()
                continue
            if line_id is None:
                return
            fixed = text
            try:
                if not self.ready_model or self.ready_model != self.cfg["translator"]:
                    continue
                t0 = time.monotonic()
                fixed, eng = self.translate_text(
                    text, lang, list(self.history),
                    on_english=lambda e: self.out.put(("translating", line_id, e)))
                log(f"  -> {fixed} | {eng}  ({(time.monotonic() - t0) * 1000:.0f} ms)")
                if eng:
                    self.out.put(("translated", line_id, eng, fixed))
            except Exception as e:
                log("Translation failed:", e)
            finally:
                self.history.append(fixed)


# --------------------------------------------------------------------------
# Listening loop
# --------------------------------------------------------------------------
class Listener(threading.Thread):
    def __init__(self, asr, translator, cfg, out_q, speakers=None):
        super().__init__(daemon=True)
        self.asr, self.translator, self.cfg, self.out = asr, translator, cfg, out_q
        self.speakers = speakers
        self.utt_id = None  # id of the line being spoken right now
        self.stop_evt = threading.Event()

    def stop(self):
        self.stop_evt.set()

    def run(self):
        cap = None
        try:
            cap = open_capture(self.cfg, on_error=lambda e: self.out.put(("error", e)))
            log(f"Listening to: {cap.name} @ {cap.sr} Hz")
            self.out.put(("listening", cap.name))
            self.loop(cap.q, cap.sr)
        except Exception as e:
            log(traceback.format_exc())
            self.out.put(("error", str(e)))
        finally:
            if cap:
                cap.close()
            self.out.put(("stopped", self))

    def language(self):
        lang = self.cfg.get("language") or ""
        if self.asr.kind == "sensevoice" and lang not in SENSEVOICE_LANGS:
            return ""
        return lang

    def finish_line(self, audio16):
        lang = self.language()
        orig, det = self.asr.transcribe(audio16, lang, True)
        line_id, self.utt_id = self.utt_id or next(_line_ids), None
        if not orig:
            self.out.put(("dropped", line_id))
            return
        can_translate = bool(self.translator.ready_model) or self.asr.kind == "whisper"
        needs_english = det != "en" and self.cfg["show"] != "original" and can_translate
        spk = self.speakers.identify(audio16) if self.speakers and self.cfg.get("speakers", True) else None
        self.out.put(("final", line_id, orig, det, needs_english, spk))
        log(f"[{det}] (speaker {spk}) {orig}")
        if not needs_english:
            return
        if self.translator.ready_model:
            self.translator.submit(line_id, orig, det)
        elif self.asr.kind == "whisper":  # no AI translator: fall back to Whisper's own
            eng = self.asr.translate(audio16, det)
            if eng:
                self.out.put(("translated", line_id, eng, orig))

    def loop(self, q, sr):
        g = gcd(sr, WHISPER_SR)
        up, down = WHISPER_SR // g, sr // g
        preroll = int(sr * 0.3)
        buf, voiced, last_voice, last_run = [], False, 0.0, 0.0

        while not self.stop_evt.is_set():
            c = self.cfg
            try:
                chunk = q.get(timeout=0.1)
            except queue.Empty:
                chunk = None
            now = time.monotonic()
            while chunk is not None:
                buf.append(chunk)
                if np.sqrt(np.mean(chunk * chunk)) > c["threshold"]:
                    voiced, last_voice = True, now
                try:
                    chunk = q.get_nowait()
                except queue.Empty:
                    chunk = None

            n = sum(len(b) for b in buf)
            if not voiced:
                if n > preroll:
                    buf = [np.concatenate(buf)[-preroll:]]
                continue

            final = (now - last_voice) >= c["silence"] or n / sr >= c["max_seg"]
            step = self.asr.live_step * (1.5 if c.get("game_mode", True) else 1.0)
            if not final and now - last_run < step:
                continue
            last_run = now

            audio = np.concatenate(buf)
            buf = [audio]
            audio16 = resample_poly(audio, up, down).astype(np.float32)

            if final:
                buf, voiced = [], False
                self.finish_line(audio16)
            else:
                text, det = self.asr.transcribe(audio16, self.language(), False)
                if text:
                    if self.utt_id is None:
                        self.utt_id = next(_line_ids)
                    self.out.put(("partial", text, det, self.utt_id))
                    if det != "en" and self.cfg["show"] != "original" and not self.cfg.get("game_mode", True):
                        self.translator.preview(self.utt_id, text, det)


# --------------------------------------------------------------------------
# Subtitle overlay
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Game-safe windows: never take focus, so fullscreen games don't minimise / tab out
# --------------------------------------------------------------------------
_U32 = None


def _user32():
    global _U32
    if _U32 is None:
        import ctypes
        from ctypes import wintypes

        u = ctypes.WinDLL("user32", use_last_error=True)
        u.GetParent.restype = wintypes.HWND
        u.GetParent.argtypes = [wintypes.HWND]
        u.GetWindowLongW.restype = wintypes.LONG
        u.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
        u.SetWindowLongW.restype = wintypes.LONG
        u.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.LONG]
        u.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int, wintypes.UINT]
        u.GetForegroundWindow.restype = wintypes.HWND
        u.SetForegroundWindow.argtypes = [wintypes.HWND]
        u.IsWindow.argtypes = [wintypes.HWND]
        _U32 = u
    return _U32


def _hwnd(win):
    win.update_idletasks()
    return _user32().GetParent(win.winfo_id())


SWP_FLAGS = 0x0001 | 0x0002 | 0x0010 | 0x0200  # NOSIZE | NOMOVE | NOACTIVATE | NOOWNERZORDER


def keep_on_top(win):
    """Put a window above everything WITHOUT activating it (Tk's -topmost activates it)."""
    if sys.platform != "win32":
        win.attributes("-topmost", True)
        return
    try:
        _user32().SetWindowPos(_hwnd(win), -1, 0, 0, 0, 0, SWP_FLAGS)  # -1 = HWND_TOPMOST
    except Exception as e:
        log("keep_on_top failed:", e)


def game_safe(win, click_through=False):
    """No focus, no taskbar button, not in Alt+Tab; optionally lets clicks pass through."""
    if sys.platform != "win32":
        win.attributes("-topmost", True)
        return
    try:
        u, hwnd = _user32(), _hwnd(win)
        style = u.GetWindowLongW(hwnd, -20)  # GWL_EXSTYLE
        style |= 0x80000 | 0x80 | 0x08000000  # LAYERED | TOOLWINDOW | NOACTIVATE
        style &= ~0x40000                     # not APPWINDOW
        style = (style | 0x20) if click_through else (style & ~0x20)  # TRANSPARENT = click-through
        u.SetWindowLongW(hwnd, -20, style)
        u.SetWindowPos(hwnd, -1, 0, 0, 0, 0, SWP_FLAGS | 0x0020)  # + FRAMECHANGED
    except Exception as e:
        log("Window setup failed:", e)


def foreground_window():
    if sys.platform != "win32":
        return None
    try:
        return _user32().GetForegroundWindow()
    except Exception:
        return None


def give_focus_back(hwnd):
    """Return keyboard focus to the game after the screenshot picker used it."""
    if sys.platform != "win32" or not hwnd:
        return
    try:
        if _user32().IsWindow(hwnd):
            _user32().SetForegroundWindow(hwnd)
    except Exception:
        pass


class Overlay:
    """Shows the last few finished lines (older ones dimmed) plus the line being spoken now."""

    def __init__(self, root, cfg, on_change):
        import tkinter as tk

        self.tk = tk
        self.cfg, self.on_change = cfg, on_change
        self.win = w = tk.Toplevel(root)
        w.overrideredirect(True)
        w.configure(bg=BG)
        self.body = tk.Frame(w, bg=BG)
        self.body.pack(fill="both", padx=12, pady=(4, 6))

        self.faded = self.user_hidden = False
        self.last_activity = self._last_top = time.monotonic()
        self.lines = []      # finished lines: dicts {id, orig, lang, eng, needs, spk}
        self.spk_seen = set()  # speaker tags only appear once 2+ different voices have been heard
        self.live = None     # line being spoken right now: {orig, lang}
        self.status_text = None
        self._place_default()

        self.menu = tk.Menu(w, tearoff=0)
        self.menu.add_command(label="Lock in place   (Ctrl+Alt+L)", command=lambda: self.on_lock(True))
        self.menu.add_command(label="Clear", command=self.clear)
        self.menu.add_command(label="Reset position", command=self.reset_position)
        self.on_lock = lambda locked: None  # set by the App
        self._bind(w)

        self.apply()
        self.status("Starting...")

    def _bind(self, widget):
        widget.bind("<ButtonPress-1>", self._drag_start)
        widget.bind("<B1-Motion>", self._drag_move)
        widget.bind("<ButtonRelease-1>", lambda e: self._moved())
        widget.bind("<MouseWheel>", self._wheel)
        widget.bind("<Button-3>", lambda e: self.menu.tk_popup(e.x_root, e.y_root))

    # ---- settings / placement ----
    def _place_default(self):
        sw, sh = self.win.winfo_screenwidth(), self.win.winfo_screenheight()
        pos = self.cfg.get("pos")
        if pos:
            self.cx, self.bottom = int(pos[0]), int(pos[1])
        else:
            self.cx, self.bottom = sw // 2, sh - int(sh * 0.08)

    def reset_position(self):
        self.cfg["pos"] = None
        self._place_default()
        self.layout()
        self.on_change()

    def apply(self):
        c = self.cfg
        self.w = int(self.win.winfo_screenwidth() * c["width"])
        self._set_alpha(0.0 if self.faded else c["opacity"])
        self._redraw()

    def _font(self, lang, big, scale=1.0):
        size = max(10, int(self.cfg["font"] * scale))
        family = SCRIPT_FONTS.get(lang, "Segoe UI")
        return (family, size, "bold") if big else (family, max(9, int(size * 0.72)))

    def _set_alpha(self, a):
        # hidden (alpha 0) also means click-through, so an invisible box never blocks clicks
        hidden = a <= 0 or self.user_hidden
        self.win.attributes("-alpha", 0.0 if self.user_hidden else a)
        game_safe(self.win, click_through=self.cfg["click_through"] or hidden)

    def layout(self):
        """Box hugs the text (up to the max width) and stays anchored at its bottom-centre point."""
        self.win.update_idletasks()
        h = self.win.winfo_reqheight()
        w = min(self.w, max(120, self.win.winfo_reqwidth()))
        self.win.geometry(f"{w}x{h}+{self.cx - w // 2}+{self.bottom - h}")

    def set_visible(self, visible):
        # Hidden = fully transparent. (Withdraw/deiconify would make Windows activate the box.)
        self.user_hidden = not visible
        self.apply()

    # ---- content ----
    def _label(self, text, font, color, top_pad, marker=None):
        """marker: (speaker number, colour) shown as a coloured tag before the text."""
        if marker is None:
            lbl = self.tk.Label(self.body, text=text, font=font, fg=color, bg=BG, justify="center",
                                wraplength=self.w - 24)
            lbl.pack(fill="x", pady=(top_pad, 0))
            self._bind(lbl)
            return
        row = self.tk.Frame(self.body, bg=BG)
        row.pack(pady=(top_pad, 0))
        num, mcol = marker
        tag = self.tk.Label(row, text=f"▍{num + 1}", font=("Segoe UI", max(9, int(font[1] * 0.75)), "bold"),
                            fg=mcol, bg=BG)
        tag.pack(side="left", anchor="n", padx=(0, 6))
        lbl = self.tk.Label(row, text=text, font=font, fg=color, bg=BG, justify="left",
                            wraplength=self.w - 90)
        lbl.pack(side="left")
        for wdg in (row, tag, lbl):
            self._bind(wdg)

    @staticmethod
    def _blend(color, amount):
        """Mix a colour toward the background (amount 0 = unchanged, 1 = background)."""
        c = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
        g = [int(BG[i:i + 2], 16) for i in (1, 3, 5)]
        return "#" + "".join(f"{round(x + (y - x) * amount):02x}" for x, y in zip(c, g))

    def _draw_line(self, ln, state, first):
        """state: 'old' (dimmed), 'new' (latest finished line) or 'live' (being spoken)."""
        orig, lang, needs = ln["orig"], ln["lang"], ln.get("needs", True)
        eng, final_eng = ln.get("eng"), ln.get("done", False)
        scale = 0.75 if state == "old" else 1.0
        gap = 0 if first else max(2, int(self.cfg["font"] * 0.2))
        show = self.cfg["show"]

        # colours: each voice gets its own colour once 2+ voices have been heard
        spk = ln.get("spk")
        voice = (SPEAKER_COLORS[spk % len(SPEAKER_COLORS)]
                 if spk is not None and len(self.spk_seen) > 1 and self.cfg.get("speakers", True) else None)
        if state == "live":
            main, sub = "#fff3b0", "#d8cf98"
        else:
            main = voice or "#ffffff"
            sub = self._blend(main, 0.3) if voice else "#a8dcff"
            if state == "old":
                main, sub = self._blend(main, 0.4), self._blend(sub, 0.45)
        if eng and not final_eng:  # early / still-streaming English: a bit softer
            eng_col = self._blend(main, 0.3)
        else:
            eng_col = main
        marker = (spk, self._blend(voice, 0.4) if state == "old" else voice) if voice else None

        if show == "original" or lang == "en" or (not needs and not eng):
            self._label(orig, self._font(lang, True, scale), main, gap, marker)
        elif show == "english":
            if eng:
                self._label(eng, self._font("en", True, scale), eng_col, gap, marker)
            else:  # nothing in English yet: show what was said, small
                self._label(orig, self._font(lang, False, scale), sub, gap, marker)
        else:
            self._label(orig, self._font(lang, False, scale), sub, gap, marker)
            self._label(eng or "…", self._font("en", True, scale), eng_col if eng else "#6a6a6a", 1)

    def _redraw(self):
        """Batch updates: at most ~8 redraws a second (streaming text can arrive every few ms)."""
        if getattr(self, "_redraw_job", None):
            return
        wait = max(0, int((0.12 - (time.monotonic() - getattr(self, "_last_draw", 0))) * 1000))
        self._redraw_job = self.win.after(wait, self._draw_now)

    def _signature(self):
        keep = int(self.cfg.get("history", 2))
        lines = tuple((ln["id"], ln["orig"], ln.get("eng"), ln.get("done"), ln.get("spk"))
                      for ln in self.lines[-keep:])
        live = self.live and (self.live.get("orig"), self.live.get("eng"))
        look = (self.cfg["font"], self.cfg["show"], self.w, keep, self.cfg.get("speakers"), len(self.spk_seen) > 1)
        return (self.status_text, lines, live, look)

    def _draw_now(self):
        self._redraw_job = None
        self._last_draw = time.monotonic()
        sig = self._signature()
        if sig == getattr(self, "_last_sig", None):
            return  # nothing visible changed - skip the repaint
        self._last_sig = sig
        for wdg in self.body.winfo_children():
            wdg.destroy()
        if self.status_text:
            text, color = self.status_text
            self._label(text, self._font("en", True), color, 0)
        else:
            keep = max(1, int(self.cfg.get("history", 3)))
            n_old = keep - 1 if self.live else keep
            shown = self.lines[-n_old:] if n_old > 0 else []
            for i, ln in enumerate(shown):
                state = "new" if (i == len(shown) - 1 and not self.live) else "old"
                self._draw_line(ln, state, i == 0)
            if self.live:
                self._draw_line(self.live, "live", not shown)
        self.layout()

    def status(self, text, color="#9ad0ff"):
        self.status_text = (text, color)
        self.lines, self.live = [], None
        self._redraw()
        self._activity()

    def partial(self, text, lang, utt_id=None):
        self.status_text = None
        if self.live and self.live.get("id") == utt_id:
            self.live.update(orig=text, lang=lang)
        else:
            self.live = {"id": utt_id, "orig": text, "lang": lang, "eng": None, "needs": True}
        self._redraw()
        self._activity()

    def live_english(self, utt_id, eng):
        """Early translation of the line still being spoken."""
        if self.live and self.live.get("id") == utt_id:
            self.live["eng"] = eng
            self._redraw()

    def _find(self, line_id):
        return next((ln for ln in reversed(self.lines) if ln["id"] == line_id), None)

    def final(self, line_id, orig, lang, needs_english, spk=None):
        self.status_text = None
        early = self.live.get("eng") if self.live and self.live.get("id") == line_id else None
        if self.live and self.live.get("id") == line_id:
            self.live = None
        if spk is not None:
            self.spk_seen.add(spk)
        self.lines.append({"id": line_id, "orig": orig, "lang": lang, "eng": early if needs_english else None,
                           "done": False, "needs": needs_english, "spk": spk})
        del self.lines[:-8]
        self._redraw()
        self._activity()

    def dropped(self, line_id):
        if self.live and self.live.get("id") == line_id:
            self.live = None
            self._redraw()

    def translating(self, line_id, eng):
        """English streaming in for a finished line."""
        ln = self._find(line_id)
        if ln and not ln["done"]:
            ln["eng"] = eng
            self._redraw()

    def translated(self, line_id, eng, fixed):
        ln = self._find(line_id)  # can land after the next line started - still shown
        if ln:
            ln.update(eng=eng, orig=fixed or ln["orig"], done=True)
            self._redraw()
            self._activity()

    def clear(self):
        self.status_text = None
        self.lines, self.live = [], None
        self.spk_seen = set()
        self._redraw()

    def _activity(self):
        self.last_activity = time.monotonic()
        if self.faded:
            self.faded = False
            self._set_alpha(self.cfg["opacity"])

    def tick(self, listening):
        now = time.monotonic()
        hold = self.cfg["hold"]
        if listening and not self.faded and hold > 0 and now - self.last_activity > hold:
            self.faded = True
            self._set_alpha(0.0)  # fade out when nobody's talking
        if not self.user_hidden and now - self._last_top > 2:
            self._last_top = now
            keep_on_top(self.win)  # stay above games/videos without stealing focus

    # ---- mouse ----
    def flash(self, text, color="#9ad0ff"):
        """Show a short notice (e.g. 'Captions locked') then go back to the captions."""
        saved = self.status_text
        self.status_text = (text, color)
        self._redraw()
        self._activity()

        def back():
            if self.status_text == (text, color):
                self.status_text = saved
                self._redraw()

        self.win.after(1800, back)

    def _drag_start(self, e):
        self._drag = None if self.cfg["click_through"] else (e.x_root, e.y_root, self.cx, self.bottom)

    def _drag_move(self, e):
        if not getattr(self, "_drag", None):
            return  # locked
        x0, y0, cx0, b0 = self._drag
        self.cx, self.bottom = cx0 + e.x_root - x0, b0 + e.y_root - y0
        self.layout()

    def _moved(self):
        if not getattr(self, "_drag", None):
            return
        self.cfg["pos"] = [self.cx, self.bottom]
        self.on_change()

    def _wheel(self, e):
        if self.cfg["click_through"]:
            return
        self.cfg["font"] = max(10, min(72, self.cfg["font"] + (2 if e.delta > 0 else -2)))
        self.apply()
        self.on_change()


# --------------------------------------------------------------------------
# Screenshot -> translate
# --------------------------------------------------------------------------
HOTKEY_LABEL = "Ctrl+Alt+T"       # drag a box (game loses focus briefly, then gets it back)
HOTKEY_FULL_LABEL = "Ctrl+Alt+Y"  # whole screen, instantly - never leaves the game
HOTKEY_LOCK_LABEL = "Ctrl+Alt+L"  # lock / unlock the captions
HOTKEY_HIDE_LABEL = "Ctrl+Alt+H"  # hide / show the captions


def start_hotkey(callback):
    """Global hotkeys that work inside games. Calls callback("region") or callback("full") from a thread."""
    if sys.platform != "win32":
        return

    def run():
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        mods = 0x2 | 0x1 | 0x4000  # MOD_CONTROL | MOD_ALT | MOD_NOREPEAT
        keys = {1: ("region", "T", HOTKEY_LABEL), 2: ("full", "Y", HOTKEY_FULL_LABEL),
                3: ("lock", "L", HOTKEY_LOCK_LABEL), 4: ("hide", "H", HOTKEY_HIDE_LABEL)}
        for hid, (_kind, key, label) in keys.items():
            if not user32.RegisterHotKey(None, hid, mods, ord(key)):
                log(f"Couldn't register {label} (another app is using it).")
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == 0x0312 and msg.wParam in keys:  # WM_HOTKEY
                callback(keys[msg.wParam][0])

    threading.Thread(target=run, daemon=True).start()


def script_font(text):
    if re.search(r"[가-힯ᄀ-ᇿ]", text):
        return "Malgun Gothic"
    if re.search(r"[぀-ヿ]", text):
        return "Yu Gothic UI"
    if re.search(r"[一-鿿]", text):
        return "Microsoft YaHei UI"
    if re.search(r"[฀-๿]", text):
        return "Leelawadee UI"
    if re.search(r"[ऀ-ॿ]", text):
        return "Nirmala UI"
    return "Segoe UI"


def grab_screen():
    from PIL import ImageGrab

    if sys.platform == "win32":
        return ImageGrab.grab()
    return ImageGrab.grab(xdisplay=os.environ.get("DISPLAY"))


def read_and_translate(image, status=log):
    """Send a screenshot to the local vision model. Returns (original, english)."""
    import base64
    import io

    w, h = image.size
    scale = 1600 / max(w, h) if max(w, h) > 1600 else (2 if max(w, h) < 500 else 1)
    if scale != 1:
        image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    png = io.BytesIO()
    image.convert("RGB").save(png, "PNG")
    data_uri = "data:image/png;base64," + base64.b64encode(png.getvalue()).decode()
    reply = HOST.chat(VISION_MODEL, [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": data_uri}},
        {"type": "text", "text": SCREEN_PROMPT},
    ]}], max_tokens=1500, temperature=0.1, timeout=180, status=status)
    return parse_screen(reply)


class Snipper:
    """Freezes the screen and lets you drag a box around the text to translate."""

    def __init__(self, root, on_done):
        import tkinter as tk
        from PIL import ImageEnhance, ImageTk

        self.on_done = on_done
        self.shot = grab_screen()
        self.win = w = tk.Toplevel(root)
        w.overrideredirect(True)
        w.attributes("-topmost", True)
        sw, sh = self.shot.size
        w.geometry(f"{sw}x{sh}+0+0")
        self.photo = ImageTk.PhotoImage(ImageEnhance.Brightness(self.shot).enhance(0.55))
        self.c = tk.Canvas(w, width=sw, height=sh, highlightthickness=0, cursor="crosshair")
        self.c.pack()
        self.c.create_image(0, 0, image=self.photo, anchor="nw")
        self.c.create_text(sw // 2, 40, text="Drag a box around the text to translate  ·  Esc to cancel",
                           fill="white", font=("Segoe UI", 16, "bold"))
        self.rect = None
        self.c.bind("<ButtonPress-1>", self._start)
        self.c.bind("<B1-Motion>", self._move)
        self.c.bind("<ButtonRelease-1>", self._end)
        w.bind("<Escape>", lambda e: self.close())
        w.bind("<Button-3>", lambda e: self.close())
        w.focus_force()

    def _start(self, e):
        self.x0, self.y0 = e.x, e.y
        self.rect = self.c.create_rectangle(e.x, e.y, e.x, e.y, outline="#4aa3ff", width=2)

    def _move(self, e):
        if self.rect:
            self.c.coords(self.rect, self.x0, self.y0, e.x, e.y)

    def _end(self, e):
        x1, y1, x2, y2 = min(self.x0, e.x), min(self.y0, e.y), max(self.x0, e.x), max(self.y0, e.y)
        if x2 - x1 > 8 and y2 - y1 > 8:
            self.on_done(self.shot.crop((x1, y1, x2, y2)), (x1, y1, x2, y2))
        self.close()

    def close(self):
        self.win.destroy()


class ResultPopup:
    def __init__(self, root, box):
        import tkinter as tk

        self.root = root
        self.win = w = tk.Toplevel(root)
        w.overrideredirect(True)
        w.configure(bg="#161a20", highlightbackground="#4aa3ff", highlightthickness=1)
        self.box = box
        self.eng = tk.Label(w, bg="#161a20", fg="white", font=("Segoe UI", 15, "bold"),
                            justify="left", anchor="w", wraplength=640, text="Reading screenshot...")
        self.eng.pack(fill="x", padx=14, pady=(12, 4))
        self.orig = tk.Label(w, bg="#161a20", fg="#a8dcff", justify="left", anchor="w", wraplength=640)
        self.orig.pack(fill="x", padx=14)
        bar = tk.Frame(w, bg="#161a20")
        bar.pack(fill="x", padx=10, pady=10)
        self.copy_btn = tk.Button(bar, text="Copy English", relief="flat", bg="#2a3240", fg="white",
                                  activebackground="#3a4458", command=self.copy, state="disabled")
        self.copy_btn.pack(side="left", padx=4)
        tk.Button(bar, text="Close", relief="flat", bg="#2a3240", fg="white",
                  activebackground="#3a4458", command=self.close).pack(side="right", padx=4)
        self.timer = tk.Label(bar, bg="#161a20", fg="#6c7686", font=("Segoe UI", 9))
        self.timer.pack(side="right", padx=6)
        self.english = ""
        self.closed = False
        self.deadline = None
        self.hovering = False
        for wdg in (w, self.eng, self.orig):
            wdg.bind("<Enter>", lambda e: setattr(self, "hovering", True))
            wdg.bind("<Leave>", lambda e: setattr(self, "hovering", False))
        self.place()
        game_safe(w)  # shows on top of the game without taking focus
        self._tick()

    def _tick(self):
        """Closes itself a while after the answer shows (paused while the mouse is over it)."""
        if self.closed:
            return
        if self.deadline is not None:
            if self.hovering:
                self.deadline = max(self.deadline, time.monotonic() + 3)
            left = self.deadline - time.monotonic()
            if left <= 0:
                return self.close()
            self.timer.config(text=f"closes in {int(left) + 1}s")
        self.win.after(250, self._tick)

    def _start_timer(self, chars):
        self.deadline = time.monotonic() + min(30, 8 + chars * 0.06)

    def place(self):
        self.win.update_idletasks()
        sw, sh = self.win.winfo_screenwidth(), self.win.winfo_screenheight()
        ww, wh = self.win.winfo_reqwidth(), self.win.winfo_reqheight()
        x1, y1, x2, y2 = self.box
        x = min(max(0, x1), sw - ww)
        y = y2 + 8 if y2 + 8 + wh < sh else max(0, y1 - wh - 8)
        self.win.geometry(f"+{x}+{y}")

    def show(self, original, english):
        self.english = english
        self.eng.config(text=english or "(no text found)")
        self.orig.config(text=original if original and original != "(none)" else "",
                         font=(script_font(original), 12))
        self.copy_btn.config(state="normal")
        self.place()
        self._start_timer(len(english or "") + len(original or ""))

    def error(self, msg):
        self.eng.config(text=msg, fg="#ff8f8f")
        self.place()
        self._start_timer(len(msg))

    def copy(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.english)
        self.copy_btn.config(text="Copied!")

    def close(self):
        if not self.closed:
            self.closed = True
            self.win.destroy()


# --------------------------------------------------------------------------
# History / chat log
# --------------------------------------------------------------------------
def format_entry(e):
    """One entry as plain text (used for the .txt log, copy and export)."""
    when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e["t"]))
    if e["kind"] == "screen":
        who = "Screenshot"
    else:
        who = f"Speaker {e['spk'] + 1}" if e.get("spk") is not None else "Speaker"
        who += f" · {LANG_NAME.get(e.get('lang'), e.get('lang') or '?')}"
    out = f"[{when}] {who}\n  {e.get('orig') or ''}\n"
    if e.get("eng"):
        out += f"  → {e['eng']}\n"
    return out


def list_sessions():
    """Saved sessions, newest first."""
    try:
        names = [f[:-6] for f in os.listdir(HISTORY_DIR) if f.endswith(".jsonl")]
    except FileNotFoundError:
        return []
    return sorted(names, reverse=True)


def load_session(name):
    out = []
    try:
        with open(os.path.join(HISTORY_DIR, name + ".jsonl"), encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    return sorted(out, key=lambda e: e["t"])


class History:
    """Keeps every finished line and screenshot translation, and writes them to history/."""
    WAIT_FOR_ENGLISH = 20  # seconds to wait for a translation before logging without one

    def __init__(self, cfg):
        self.cfg = cfg
        self.session = time.strftime("%Y-%m-%d %H-%M-%S")
        self.entries = []
        self.pending = {}     # line_id -> entry still waiting for its English
        self.listeners = []   # called with the full entry list whenever it changes

    def _write(self, e):
        if not self.cfg.get("save_history", True):
            return
        try:
            os.makedirs(HISTORY_DIR, exist_ok=True)
            base = os.path.join(HISTORY_DIR, self.session)
            with open(base + ".jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
            with open(base + ".txt", "a", encoding="utf-8") as f:
                f.write(format_entry(e) + "\n")
        except Exception as ex:
            log("Couldn't save history:", ex)

    def _commit(self, e):
        self.entries.append(e)
        if len(self.entries) > 1 and self.entries[-2]["t"] > e["t"]:
            self.entries.sort(key=lambda x: x["t"])
        self._write(e)
        for fn in list(self.listeners):
            try:
                fn(self.entries)
            except Exception:
                log(traceback.format_exc())

    def line(self, line_id, orig, lang, needs_english, spk):
        e = {"kind": "speech", "t": time.time(), "spk": spk, "lang": lang, "orig": orig, "eng": None}
        if needs_english:
            self.pending[line_id] = e
        else:
            self._commit(e)

    def translated(self, line_id, eng, fixed):
        e = self.pending.pop(line_id, None)
        if e:
            e["eng"], e["orig"] = eng, fixed or e["orig"]
            self._commit(e)

    def screenshot(self, orig, eng):
        if eng or orig:
            self._commit({"kind": "screen", "t": time.time(), "spk": None, "lang": None,
                          "orig": orig if orig != "(none)" else "", "eng": eng})

    def tick(self):
        now = time.time()
        for lid, e in list(self.pending.items()):
            if now - e["t"] > self.WAIT_FOR_ENGLISH:
                self._commit(self.pending.pop(lid))

    def flush(self):
        for lid in list(self.pending):
            self._commit(self.pending.pop(lid))


class HistoryWindow:
    """Browse, search, copy and export the chat log."""
    THIS = "This session"

    def __init__(self, root, history, cfg, icon=None):
        import tkinter as tk
        from tkinter import ttk

        self.tk, self.history, self.cfg = tk, history, cfg
        self.win = w = tk.Toplevel(root)
        w.title("LiveSubs - History")
        w.geometry("760x620")
        w.minsize(480, 300)
        if icon:
            try:
                w.iconbitmap(icon)
            except Exception:
                pass

        top = ttk.Frame(w, padding=(10, 10, 10, 6))
        top.pack(fill="x")
        ttk.Label(top, text="Session").pack(side="left")
        self.session_cb = ttk.Combobox(top, state="readonly", width=22)
        self.session_cb.pack(side="left", padx=(6, 12))
        self.session_cb.bind("<<ComboboxSelected>>", lambda e: self.reload())
        ttk.Label(top, text="Search").pack(side="left")
        self.search = tk.StringVar()
        ent = ttk.Entry(top, textvariable=self.search)
        ent.pack(side="left", fill="x", expand=True, padx=(6, 0))
        self.search.trace_add("write", lambda *a: self.render())

        mid = ttk.Frame(w, padding=(10, 0, 10, 0))
        mid.pack(fill="both", expand=True)
        self.text = tk.Text(mid, bg=BG, fg="#e8e8e8", insertbackground="white", relief="flat",
                            wrap="word", padx=12, pady=10, spacing1=2, font=("Segoe UI", 11),
                            cursor="arrow")
        sb = ttk.Scrollbar(mid, command=self.text.yview)
        self.text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)
        self.text.tag_configure("time", foreground="#7b8594", font=("Consolas", 9))
        self.text.tag_configure("who", foreground="#9aa4b2", font=("Segoe UI", 9, "bold"))
        self.text.tag_configure("eng", foreground="#ffffff", font=("Segoe UI", 12, "bold"),
                                lmargin1=14, lmargin2=14, spacing3=10)
        self.text.tag_configure("screen", foreground="#c9a2ff", font=("Segoe UI", 9, "bold"))
        self.text.tag_configure("hit", background="#4a3d00")
        for i, col in enumerate(SPEAKER_COLORS):
            self.text.tag_configure(f"spk{i}", foreground=col, font=("Segoe UI", 9, "bold"))
        # read-only, but text can still be selected and copied
        self.text.bind("<Key>", lambda e: None if (e.state & 0x4 and e.keysym.lower() in ("c", "a")) else "break")

        bot = ttk.Frame(w, padding=10)
        bot.pack(fill="x")
        self.save_var = tk.BooleanVar(value=cfg.get("save_history", True))
        ttk.Checkbutton(bot, text="Save chat logs to disk", variable=self.save_var,
                        command=self._save_toggled).pack(side="left")
        self.count = tk.StringVar()
        ttk.Label(bot, textvariable=self.count, foreground="#888").pack(side="left", padx=12)
        ttk.Button(bot, text="Open folder", command=self.open_folder).pack(side="right")
        ttk.Button(bot, text="Export...", command=self.export).pack(side="right", padx=6)
        ttk.Button(bot, text="Copy", command=self.copy).pack(side="right")

        self.entries, self.shown = [], []
        self.history.listeners.append(self._live)
        w.protocol("WM_DELETE_WINDOW", self.close)
        self.refresh_sessions()
        self.reload()

    # ---- data ----
    def refresh_sessions(self):
        past = [s for s in list_sessions() if s != self.history.session]
        self.session_cb.config(values=[self.THIS] + past)
        if not self.session_cb.get():
            self.session_cb.set(self.THIS)

    def reload(self):
        sel = self.session_cb.get()
        self.entries = list(self.history.entries) if sel == self.THIS else load_session(sel)
        self.render(scroll_end=True)

    def _live(self, entries):
        if self.session_cb.get() == self.THIS:
            self.entries = list(entries)
            at_end = self.text.yview()[1] > 0.98
            self.render(scroll_end=at_end)

    def _filtered(self):
        q = self.search.get().strip().lower()
        if not q:
            return self.entries
        return [e for e in self.entries
                if q in (e.get("orig") or "").lower() or q in (e.get("eng") or "").lower()]

    # ---- drawing ----
    def render(self, scroll_end=False):
        t = self.text
        y = t.yview()[0]
        t.delete("1.0", "end")
        self.shown = self._filtered()
        multi = len({e.get("spk") for e in self.entries if e.get("spk") is not None}) > 1
        last_day = None
        for e in self.shown:
            day = time.strftime("%A %d %B %Y", time.localtime(e["t"]))
            if day != last_day:
                t.insert("end", f"{day}\n", "time")
                last_day = day
            t.insert("end", time.strftime("%H:%M:%S", time.localtime(e["t"])) + "  ", "time")
            if e["kind"] == "screen":
                t.insert("end", "SCREENSHOT\n", "screen")
            else:
                spk = e.get("spk")
                tag = f"spk{spk % len(SPEAKER_COLORS)}" if (multi and spk is not None) else "who"
                who = f"SPEAKER {spk + 1} · " if spk is not None else ""
                t.insert("end", f"{who}{LANG_NAME.get(e.get('lang'), e.get('lang') or '').upper()}\n", tag)
            if e.get("orig"):
                otag = f"orig_{e.get('lang') or 'x'}"
                t.tag_configure(otag, foreground="#a8dcff", lmargin1=14, lmargin2=14,
                                font=(SCRIPT_FONTS.get(e.get("lang"), script_font(e["orig"])), 11))
                t.insert("end", e["orig"] + "\n", (otag,) if e.get("eng") else (otag, "eng"))
            if e.get("eng"):
                t.insert("end", e["eng"] + "\n", "eng")
        q = self.search.get().strip()
        if q:
            start = "1.0"
            while True:
                pos = t.search(q, start, stopindex="end", nocase=True)
                if not pos:
                    break
                end = f"{pos}+{len(q)}c"
                t.tag_add("hit", pos, end)
                start = end
        n, total = len(self.shown), len(self.entries)
        self.count.set(f"{n} of {total} lines" if q else f"{total} lines")
        if not self.entries:
            t.insert("end", "Nothing here yet - translated lines show up as they happen.", "time")
        if scroll_end:
            t.see("end")
        else:
            t.yview_moveto(y)

    # ---- buttons ----
    def _save_toggled(self):
        self.cfg["save_history"] = bool(self.save_var.get())
        save_settings(self.cfg)

    def _as_text(self):
        return "\n".join(format_entry(e) for e in self.shown)

    def copy(self):
        try:
            sel = self.text.get("sel.first", "sel.last")
        except Exception:
            sel = self._as_text()
        self.win.clipboard_clear()
        self.win.clipboard_append(sel)

    def export(self):
        from tkinter import filedialog

        name = self.session_cb.get()
        default = (self.history.session if name == self.THIS else name) + ".txt"
        path = filedialog.asksaveasfilename(parent=self.win, defaultextension=".txt", initialfile=default,
                                            filetypes=[("Text file", "*.txt"), ("All files", "*.*")])
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._as_text())

    def open_folder(self):
        os.makedirs(HISTORY_DIR, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(HISTORY_DIR)
        else:
            subprocess.Popen(["xdg-open", HISTORY_DIR])

    def show(self):
        self.win.deiconify()
        self.win.lift()
        self.refresh_sessions()

    def close(self):
        if self._live in self.history.listeners:
            self.history.listeners.remove(self._live)
        self.win.destroy()


# --------------------------------------------------------------------------
# Control panel
# --------------------------------------------------------------------------
class App:
    def __init__(self):
        import tkinter as tk
        from tkinter import ttk

        self.tk, self.ttk = tk, ttk
        self.cfg = load_settings()
        self.q = queue.Queue()
        self.asr = None
        self.speakers = None
        self.listener = None
        self.loading = self.reload_pending = False
        self.want_listen = bool(self.cfg["autostart"])
        self.translator = Translator(self.cfg, self.q)
        self.translator.start()
        self.history = History(self.cfg)
        self.history_win = None
        HOST.set_game_mode(bool(self.cfg.get("game_mode", True)))
        self.tr_loading = False

        self.root = tk.Tk()
        self.root.report_callback_exception = lambda *a: log("".join(traceback.format_exception(*a)))
        self.overlay = Overlay(self.root, self.cfg, self.settings_changed)
        self.build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.load_asr()
        self.load_translator()
        start_hotkey(lambda kind: self.q.put(("screenshot", kind)))
        self.apply_game_mode()
        self.root.after(50, self.poll)

    # ---- UI ----
    def build_ui(self):
        tk, ttk, c, r = self.tk, self.ttk, self.cfg, self.root
        r.title("LiveSubs")
        r.resizable(False, False)
        if os.path.exists(ICON_PATH):
            try:
                r.iconbitmap(default=ICON_PATH)
            except Exception:
                pass
        bg = ttk.Style().lookup("TFrame", "background") or "SystemButtonFace"

        f = ttk.Frame(r, padding=16)
        f.grid(sticky="nsew")
        f.columnconfigure(1, weight=1)

        # Status lines
        def status_line(row_i):
            fr = ttk.Frame(f)
            fr.grid(row=row_i, column=0, columnspan=2, sticky="ew")
            dot = tk.Label(fr, text="●", fg="#d9a400", bg=bg, font=("Segoe UI", 12))
            dot.pack(side="left")
            var = tk.StringVar(value="Starting...")
            ttk.Label(fr, textvariable=var).pack(side="left", padx=4)
            return fr, dot, var

        top, self.dot, self.status_var = status_line(0)
        self.hear_var = tk.StringVar()
        ttk.Label(top, textvariable=self.hear_var, foreground="#777").pack(side="right")
        _, self.tr_dot, self.tr_var = status_line(1)
        ttk.Frame(f, height=10).grid(row=2, column=0)

        def row(i, label):
            ttk.Label(f, text=label).grid(row=i, column=0, sticky="w", padx=(0, 12), pady=4)

        # Speech model
        row(3, "Speech model")
        mf = ttk.Frame(f)
        mf.grid(row=3, column=1, sticky="ew")
        mf.columnconfigure(0, weight=1)
        self.engine_cb = ttk.Combobox(mf, values=list(ENGINES), state="readonly", width=40)
        self.engine_cb.set({v: k for k, v in ENGINES.items()}[c["engine"]])
        self.engine_cb.grid(row=0, column=0, sticky="ew")
        self.dev_cb = ttk.Combobox(mf, values=list(DEVICES), state="readonly", width=6)
        self.dev_cb.set({v: k for k, v in DEVICES.items()}.get(c["device"], "Auto"))
        self.dev_cb.grid(row=0, column=1, padx=(6, 0))
        for cb in (self.engine_cb, self.dev_cb):
            cb.bind("<<ComboboxSelected>>", lambda e: self.engine_changed())

        # Translator
        row(4, "Translator")
        self.tr_cb = ttk.Combobox(f, values=list(TRANSLATORS), state="readonly")
        self.tr_cb.set({v: k for k, v in TRANSLATORS.items()}.get(c["translator"], "Qwen3 14B (best)"))
        self.tr_cb.grid(row=4, column=1, sticky="ew")
        self.tr_cb.bind("<<ComboboxSelected>>", lambda e: self.translator_changed())

        # Language
        row(5, "Spoken language")
        self.lang_cb = ttk.Combobox(f, values=[n for n, _ in LANGUAGES], state="readonly")
        self.lang_cb.set(next((n for n, code in LANGUAGES if code == c["language"]), "Auto-detect"))
        self.lang_cb.grid(row=5, column=1, sticky="ew")
        self.lang_cb.bind("<<ComboboxSelected>>", lambda e: self.simple_changed())

        # Show mode
        row(6, "Show")
        self.show_cb = ttk.Combobox(f, values=list(SHOW_MODES), state="readonly")
        self.show_cb.set({v: k for k, v in SHOW_MODES.items()}.get(c["show"], "Original + English"))
        self.show_cb.grid(row=6, column=1, sticky="ew")
        self.show_cb.bind("<<ComboboxSelected>>", lambda e: self.simple_changed())

        # Which apps to listen to
        row(7, "Listen to")
        lf = ttk.Frame(f)
        lf.grid(row=7, column=1, sticky="ew")
        lf.columnconfigure(0, weight=1)
        self.appmode_cb = ttk.Combobox(lf, values=list(APP_MODES), state="readonly")
        self.appmode_cb.set({v: k for k, v in APP_MODES.items()}.get(c["app_mode"], "All apps"))
        self.appmode_cb.grid(row=0, column=0, sticky="ew")
        self.appmode_cb.bind("<<ComboboxSelected>>", lambda e: self.app_mode_changed())
        self.apps_btn = ttk.Button(lf, text="Apps...", command=self.pick_apps)
        self.apps_btn.grid(row=0, column=1, padx=(6, 0))
        self.apps_var = tk.StringVar()
        ttk.Label(f, textvariable=self.apps_var, foreground="#777").grid(row=8, column=1, sticky="w")

        # Output device (for "All apps")
        row(9, "Output device")
        af = ttk.Frame(f)
        af.grid(row=9, column=1, sticky="ew")
        af.columnconfigure(0, weight=1)
        self.audio_cb = ttk.Combobox(af, state="readonly")
        self.audio_cb.grid(row=0, column=0, sticky="ew")
        self.audio_cb.bind("<<ComboboxSelected>>", lambda e: self.audio_changed())
        ttk.Button(af, text="↻", width=3, command=self.refresh_audio).grid(row=0, column=1, padx=(6, 0))
        self.refresh_audio()

        ttk.Separator(f).grid(row=10, column=0, columnspan=2, sticky="ew", pady=10)

        def slider(i, label, key, lo, hi, cast):
            row(i, label)
            var = tk.DoubleVar(value=c[key])
            ttk.Scale(f, from_=lo, to=hi, variable=var,
                      command=lambda v: self.look_changed(key, cast(float(v)))).grid(row=i, column=1, sticky="ew")
            return var

        self.font_var = slider(11, "Text size", "font", 10, 48, int)
        slider(12, "Background", "opacity", 0.3, 1.0, float)
        slider(13, "Max width", "width", 0.25, 1.0, float)
        slider(14, "Lines on screen", "history", 1, 4, lambda v: int(round(v)))

        self.spk_var = tk.BooleanVar(value=c["speakers"])
        ttk.Checkbutton(f, text="Label different speakers", variable=self.spk_var,
                        command=lambda: self.look_changed("speakers", self.spk_var.get())
                        ).grid(row=15, column=0, columnspan=2, sticky="w", pady=(8, 0))

        self.click_var = tk.BooleanVar(value=c["click_through"])
        ttk.Checkbutton(f, text=f"Lock captions in place   ({HOTKEY_LOCK_LABEL})", variable=self.click_var,
                        command=lambda: self.set_locked(self.click_var.get())
                        ).grid(row=16, column=0, columnspan=2, sticky="w")
        self.overlay.on_lock = self.set_locked
        self.game_var = tk.BooleanVar(value=c["game_mode"])
        ttk.Checkbutton(f, text="Game mode (smoother games - English shows when each line ends)",
                        variable=self.game_var, command=self.game_mode_changed
                        ).grid(row=17, column=0, columnspan=2, sticky="w")
        self.auto_var = tk.BooleanVar(value=c["autostart"])
        ttk.Checkbutton(f, text="Start subtitles when the app opens", variable=self.auto_var,
                        command=self.simple_changed).grid(row=18, column=0, columnspan=2, sticky="w")

        bf = ttk.Frame(f)
        bf.grid(row=19, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        bf.columnconfigure((0, 1, 2), weight=1)
        self.start_btn = ttk.Button(bf, text="Start subtitles", command=self.toggle_listen)
        self.start_btn.grid(row=0, column=0, sticky="ew")
        self.vis_btn = ttk.Button(bf, text="Hide box", command=self.toggle_visible)
        self.vis_btn.grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(bf, text="Reset position", command=self.overlay.reset_position).grid(row=0, column=2, sticky="ew")
        ttk.Button(bf, text=f"Translate part of the screen   ({HOTKEY_LABEL})", command=self.start_snip
                   ).grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        ttk.Button(bf, text="History & chat log", command=self.open_history
                   ).grid(row=2, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        ttk.Label(bf, text=f"In game:  {HOTKEY_FULL_LABEL} translate screen  ·  {HOTKEY_LOCK_LABEL} lock captions"
                           f"  ·  {HOTKEY_HIDE_LABEL} hide captions",
                  foreground="#888").grid(row=3, column=0, columnspan=3, pady=(4, 0))

        ttk.Label(f, text="Drag the box to move it · scroll on it to resize text · right-click for options",
                  foreground="#888").grid(row=20, column=0, columnspan=2, pady=(10, 0))
        self._sync_device_box()
        self._sync_app_widgets()

    def _sync_device_box(self):
        self.dev_cb.config(state="readonly")  # every speech model can run on the GPU now

    def _sync_app_widgets(self):
        mode = self.cfg["app_mode"]
        apps = ", ".join(self.cfg["apps"]) or "none chosen"
        self.apps_btn.config(state="disabled" if mode == "all" else "normal")
        self.audio_cb.config(state="readonly" if mode == "all" else "disabled")
        self.apps_var.set({"all": "Everything your PC plays",
                           "exclude": f"Ignoring: {apps}",
                           "include": f"Only: {apps}"}[mode])

    def app_mode_changed(self):
        self.cfg["app_mode"] = APP_MODES[self.appmode_cb.get()]
        save_settings(self.cfg)
        self._sync_app_widgets()
        if self.cfg["app_mode"] != "all" and not self.cfg["apps"]:
            self.pick_apps()
        self.restart_listening()

    def restart_listening(self):
        if self.listener:
            self.stop_listening()
            self.start_listening()

    def pick_apps(self):
        tk, ttk = self.tk, self.ttk
        d = tk.Toplevel(self.root)
        d.title("Choose apps")
        d.transient(self.root)
        d.resizable(False, False)
        d.grab_set()
        f = ttk.Frame(d, padding=14)
        f.pack(fill="both")
        verb = "ignore" if self.cfg["app_mode"] == "exclude" else "listen to"
        ttk.Label(f, text=f"Tick the apps to {verb}:", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Label(f, text="Apps show up here once they've played sound.\nOpen the game / Spotify first, then Refresh.",
                  foreground="#777").pack(anchor="w", pady=(2, 8))
        box = ttk.Frame(f)
        box.pack(fill="x")
        checks = {}

        def fill():
            for wdg in box.winfo_children():
                wdg.destroy()
            chosen = {_app_key(a) for a in self.cfg["apps"]} | {k for k, v in checks.items() if v.get()}
            names = {_app_key(n): n for n in list_audio_apps()}
            for a in self.cfg["apps"]:
                names.setdefault(_app_key(a), a)
            for k, v in list(checks.items()):
                names.setdefault(k, k)
            checks.clear()
            if not names:
                ttk.Label(box, text="(no apps playing sound right now)", foreground="#999").pack(anchor="w")
            for key, name in sorted(names.items()):
                var = tk.BooleanVar(value=key in chosen)
                checks[key] = var
                ttk.Checkbutton(box, text=name, variable=var).pack(anchor="w")

        add = ttk.Frame(f)
        add.pack(fill="x", pady=(10, 0))
        entry = ttk.Entry(add, width=24)
        entry.pack(side="left", fill="x", expand=True)
        entry.insert(0, "")

        def add_app():
            name = entry.get().strip()
            if name:
                checks[_app_key(name)] = tk.BooleanVar(value=True)
                entry.delete(0, "end")
                fill()

        ttk.Button(add, text="Add by name", command=add_app).pack(side="left", padx=(6, 0))
        ttk.Label(f, text="e.g. spotify.exe, chrome.exe, RobloxPlayerBeta.exe", foreground="#999").pack(anchor="w")

        def save():
            self.cfg["apps"] = sorted(k for k, v in checks.items() if v.get())
            save_settings(self.cfg)
            self._sync_app_widgets()
            d.destroy()
            self.restart_listening()

        btns = ttk.Frame(f)
        btns.pack(fill="x", pady=(12, 0))
        ttk.Button(btns, text="Refresh", command=fill).pack(side="left")
        ttk.Button(btns, text="Save", command=save).pack(side="right")
        fill()

    # ---- screenshot translate ----
    def open_history(self):
        if self.history_win and self.history_win.win.winfo_exists():
            self.history_win.show()
        else:
            self.history_win = HistoryWindow(self.root, self.history, self.cfg,
                                             ICON_PATH if os.path.exists(ICON_PATH) else None)

    def _translate_image(self, image, box):
        for p in getattr(self, "popups", []):  # one answer on screen at a time
            p.close()
        popup = ResultPopup(self.root, box)
        self.popups = [popup]

        def work():
            try:
                t0 = time.monotonic()
                orig, eng = read_and_translate(
                    image, status=lambda m: self.q.put(("screen_status", popup, m)))
                log(f"Screenshot: {orig!r} -> {eng!r} ({time.monotonic() - t0:.1f}s)")
                self.q.put(("screen_result", popup, orig, eng, None))
            except Exception as e:
                log(traceback.format_exc())
                self.q.put(("screen_result", popup, "", "", str(e)))
            # put the subtitle translator back on the GPU so the next line isn't delayed
            tr = self.cfg["translator"]
            if tr not in ("off", VISION_MODEL) and self.translator.ready_model:
                try:
                    HOST.use(tr)
                except Exception as e:
                    log("Couldn't reload translator:", e)

        threading.Thread(target=work, daemon=True).start()

    def _hide_for_shot(self):
        """Make our own windows invisible for a moment so they're not in the screenshot."""
        self.overlay.win.attributes("-alpha", 0.0)
        for p in getattr(self, "popups", []):
            p.close()
        self.root.update()

    def start_full_shot(self):
        """Whole screen, no dragging and no focus change - safe mid-game."""
        if getattr(self, "_snipping", False):
            return
        try:
            self._hide_for_shot()
            time.sleep(0.05)
            image = grab_screen()
        except Exception as e:
            log(traceback.format_exc())
            self.set_status(f"Screenshot failed: {e}", "#d9534f")
            return
        finally:
            self.overlay.apply()
        sw = self.root.winfo_screenwidth()
        self._translate_image(image, (sw // 2 - 330, 40, sw // 2 + 330, 40))

    def start_snip(self):
        if getattr(self, "_snipping", False):
            return
        self._snipping = True
        game = foreground_window()  # remember the game so it gets focus back afterwards
        self._hide_for_shot()

        def finish(*args):
            if not self._snipping:
                return
            self._snipping = False
            self.overlay.apply()
            self.root.after(30, lambda: give_focus_back(game))
            if args:
                self._translate_image(*args)

        try:
            snip = Snipper(self.root, finish)
            snip.win.bind("<Destroy>", lambda e: e.widget is snip.win and finish())
        except Exception as e:
            log(traceback.format_exc())
            finish()
            self.set_status(f"Screenshot failed: {e}", "#d9534f")

    def set_status(self, text, color):
        self.status_var.set(text)
        self.dot.config(fg=color)

    def set_tr_status(self, text, color):
        self.tr_var.set(text)
        self.tr_dot.config(fg=color)

    def refresh_audio(self):
        names = [DEFAULT_AUDIO] + list_audio_devices()
        self.audio_cb.config(values=names)
        cur = self.cfg["audio_device"]
        self.audio_cb.set(cur if cur in names else DEFAULT_AUDIO)

    # ---- setting changes ----
    def settings_changed(self):
        save_settings(self.cfg)
        try:
            self.font_var.set(self.cfg["font"])
        except Exception:
            pass

    def simple_changed(self):
        self.cfg["language"] = dict(LANGUAGES).get(self.lang_cb.get(), "")
        self.cfg["show"] = SHOW_MODES.get(self.show_cb.get(), "both")
        self.cfg["autostart"] = bool(self.auto_var.get())
        save_settings(self.cfg)
        self.overlay.apply()
        lang = self.cfg["language"]
        if self.cfg["engine"] == "sensevoice" and lang and lang not in SENSEVOICE_LANGS:
            self.set_status(f"SenseVoice can't do {LANG_NAME[lang]} - pick a Whisper model", "#d9534f")

    def look_changed(self, key, value):
        self.cfg[key] = value
        self.overlay.apply()
        save_settings(self.cfg)

    def audio_changed(self):
        sel = self.audio_cb.get()
        self.cfg["audio_device"] = "" if sel == DEFAULT_AUDIO else sel
        save_settings(self.cfg)
        self.restart_listening()

    def engine_changed(self):
        self.cfg["engine"] = ENGINES[self.engine_cb.get()]
        self.cfg["device"] = DEVICES.get(self.dev_cb.get(), "auto")
        self._sync_device_box()
        save_settings(self.cfg)
        self.load_asr()

    def translator_changed(self):
        self.cfg["translator"] = TRANSLATORS[self.tr_cb.get()]
        save_settings(self.cfg)
        self.load_translator()

    # ---- loading ----
    def load_asr(self):
        if self.loading:
            self.reload_pending = True
            return
        if self.listener:
            self.want_listen = True
            self.stop_listening()
        self.loading = True
        self.asr = None
        engine, dev = self.cfg["engine"], self.cfg["device"]
        self.set_status("Loading speech model...", "#d9a400")
        self.overlay.status("Loading...")

        def work():
            try:
                asr = load_asr(engine, dev, status=lambda s: self.q.put(("status", s)),
                               game_mode=self.cfg.get("game_mode", True))
                if not self.speakers:
                    try:
                        self.speakers = SpeakerTracker(status=lambda s: self.q.put(("status", s)))
                    except Exception as e:
                        log("Speaker labels unavailable:", e)
                self.q.put(("loaded", asr))
            except Exception as e:
                log(traceback.format_exc())
                self.q.put(("load_error", str(e)))

        threading.Thread(target=work, daemon=True).start()

    def load_translator(self):
        if self.tr_loading:
            return
        model = self.cfg["translator"]
        if model == "off":
            self.translator.ready_model = None
            threading.Thread(target=HOST.stop, daemon=True).start()  # free the GPU
            self.set_tr_status("Translator off", "#9a9a9a")
            return
        self.tr_loading = True
        self.set_tr_status(f"Loading translator ({model})...", "#d9a400")

        def work():
            err = self.translator.prepare(status=lambda m: self.q.put(("tr_status", m)))
            self.q.put(("tr_ready", model, err))

        threading.Thread(target=work, daemon=True).start()

    # ---- listening ----
    def start_listening(self):
        self.want_listen = True
        if not self.asr or self.listener:
            return
        if self.speakers:
            self.speakers.reset()
        self.listener = Listener(self.asr, self.translator, self.cfg, self.q, self.speakers)
        self.listener.start()
        self.start_btn.config(text="Stop subtitles")

    def stop_listening(self):
        if self.listener:
            self.listener.stop()
            self.listener = None
        self.start_btn.config(text="Start subtitles")

    def toggle_listen(self):
        if self.listener:
            self.want_listen = False
            self.stop_listening()
            self.overlay.status("Paused", "#9a9a9a")
            self.ready_status()
        else:
            self.start_listening()

    def apply_game_mode(self):
        on = bool(self.cfg.get("game_mode", True))
        set_priority(os.getpid(), on)
        HOST.set_game_mode(on)

    def game_mode_changed(self):
        self.cfg["game_mode"] = bool(self.game_var.get())
        save_settings(self.cfg)
        self.apply_game_mode()
        if self.cfg["engine"] == "sensevoice":
            self.load_asr()  # picks up the new thread count

    def set_locked(self, locked):
        """Locked = can't be dragged or resized, and clicks go straight through to the game."""
        self.cfg["click_through"] = bool(locked)
        self.click_var.set(bool(locked))
        save_settings(self.cfg)
        self.overlay.apply()
        self.overlay.flash(f"Captions locked  ·  {HOTKEY_LOCK_LABEL} to unlock" if locked
                           else "Captions unlocked  ·  drag to move, scroll to resize")

    def toggle_visible(self):
        show = self.overlay.user_hidden
        self.overlay.set_visible(show)
        self.vis_btn.config(text="Hide box" if show else "Show box")

    def ready_status(self):
        if not self.asr:
            return
        dev = "GPU" if self.asr.device == "cuda" else "CPU"
        state = "Listening" if self.listener else "Ready"
        self.set_status(f"{state} · {self.asr.name} ({dev})", "#2fb35a" if self.listener else "#4a90d9")

    # ---- main loop ----
    def poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "status":
                    self.set_status(msg[1], "#d9a400")
                    self.overlay.status(msg[1])
                elif kind == "loaded":
                    self.loading = False
                    if self.reload_pending:
                        self.reload_pending = False
                        self.load_asr()
                        continue
                    self.asr = msg[1]
                    log(f"Speech model ready: {self.asr.name} on {self.asr.device}")
                    self.overlay.status("Starting..." if self.want_listen else "Ready")
                    if self.want_listen:
                        self.start_listening()
                    self.ready_status()
                elif kind == "load_error":
                    self.loading = False
                    self.set_status("Speech model failed (see livesubs.log)", "#d9534f")
                    self.overlay.status(f"Error: {msg[1]}", "#ff7b7b")
                elif kind == "tr_ready":
                    self.tr_loading = False
                    model, err = msg[1], msg[2]
                    if model != self.cfg["translator"]:
                        self.load_translator()  # changed while loading
                    elif err:
                        self.set_tr_status(err, "#d9534f")
                    else:
                        engine = "GPU" + (" · Vulkan" if HOST.backend == "vulkan" else "")
                        self.set_tr_status(f"Translator ready · {LLM_MODELS[model]['label']} ({engine})", "#2fb35a")
                elif kind == "tr_status":
                    self.set_tr_status(msg[1], "#d9a400")
                elif kind == "listening":
                    self.ready_status()
                    self.overlay.status("Listening...")
                elif kind == "partial":
                    self.overlay.partial(msg[1], msg[2], msg[3])
                    self.hear_var.set(f"Hearing: {LANG_NAME.get(msg[2], msg[2])}")
                elif kind == "final":
                    self.overlay.final(*msg[1:])
                    _lid, _orig, _lang, _needs, _spk = msg[1:6]
                    self.history.line(_lid, _orig, _lang, _needs, _spk)
                    self.hear_var.set(f"Hearing: {LANG_NAME.get(msg[3], msg[3])}")
                elif kind == "translated":
                    self.overlay.translated(msg[1], msg[2], msg[3])
                    self.history.translated(msg[1], msg[2], msg[3])
                elif kind == "translating":
                    self.overlay.translating(msg[1], msg[2])
                elif kind == "live_eng":
                    self.overlay.live_english(msg[1], msg[2])
                elif kind == "dropped":
                    self.overlay.dropped(msg[1])
                elif kind == "screenshot":
                    if msg[1] == "full":
                        self.start_full_shot()
                    elif msg[1] == "region":
                        self.start_snip()
                    elif msg[1] == "lock":
                        self.set_locked(not self.cfg["click_through"])
                    elif msg[1] == "hide":
                        self.toggle_visible()
                elif kind == "screen_status":
                    try:
                        msg[1].eng.config(text=msg[2])
                    except Exception:
                        pass
                elif kind == "screen_result":
                    popup, orig, eng, err = msg[1:]
                    if not err:
                        self.history.screenshot(orig, eng)
                    try:
                        popup.error(err) if err else popup.show(orig, eng)
                    except Exception:
                        pass  # popup was closed already
                elif kind == "error":
                    self.set_status(f"Audio error: {msg[1]}", "#d9534f")
                    self.overlay.status(f"Error: {msg[1]}", "#ff7b7b")
                elif kind == "stopped":
                    if msg[1] is self.listener:
                        self.listener = None
                        self.start_btn.config(text="Start subtitles")
        except queue.Empty:
            pass
        self.overlay.tick(self.listener is not None)
        self.history.tick()
        self.root.after(50, self.poll)

    def quit(self):
        lst = self.listener
        self.stop_listening()
        save_settings(self.cfg)
        if lst:
            lst.join(timeout=1.5)
        self.history.flush()
        HOST.stop()  # close the AI engine and free the GPU memory
        self.root.destroy()
        os._exit(0)

    def run(self):
        self.root.mainloop()


# --------------------------------------------------------------------------
# Setup helpers (used by "Install LiveSubs.bat")
# --------------------------------------------------------------------------
def prepare():
    cfg = load_settings()

    def say(s):
        print("       " + s, flush=True)

    asr = load_asr(cfg["engine"], cfg["device"], say)
    say(f"Speech model ready: {asr.name} ({'GPU' if asr.device == 'cuda' else 'CPU'})")
    SpeakerTracker(say)
    say("Speaker model ready.")

    ensure_llama_builds(say)
    say("AI engine downloaded.")
    keys = [VISION_MODEL] + ([cfg["translator"]] if cfg["translator"] not in ("off", VISION_MODEL) else [])
    for key in keys:
        ensure_model(key, say)
    say("AI models downloaded.")

    try:
        if cfg["translator"] != "off":
            tr = Translator(cfg, queue.Queue())
            err = tr.prepare(say)
            if err:
                raise SystemExit("       " + err)
            say(f"Translator running on the GPU ({HOST.backend}).")
            for line, lang in (("今天天气真不错，我们出去走走吧。", "zh"),
                               ("조 금만 생각 을 하 면서 살 면 훨씬 편할 거야.", "ko")):
                t0 = time.monotonic()
                fixed, out = tr.translate_text(line, lang, [])
                say(f"Test: {fixed} -> {out}  ({(time.monotonic() - t0) * 1000:.0f} ms)")

        from PIL import Image, ImageDraw

        img = Image.new("RGB", (360, 90), "white")
        ImageDraw.Draw(img).text((20, 30), "START GAME   OPTIONS", fill="black")
        t0 = time.monotonic()
        orig, eng = read_and_translate(img, say)
        say(f"Screenshot test: read {orig!r}  ({time.monotonic() - t0:.1f} s)")
    finally:
        HOST.stop()


def install_shortcuts():
    pyw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
    if not os.path.exists(pyw):
        pyw = sys.executable
    script = os.path.abspath(__file__)
    icon = ICON_PATH if os.path.exists(ICON_PATH) else pyw

    def q(s):
        return s.replace("'", "''")

    ps = f"""
$ws = New-Object -ComObject WScript.Shell
foreach ($dir in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) {{
  $s = $ws.CreateShortcut((Join-Path $dir 'LiveSubs.lnk'))
  $s.TargetPath = '{q(pyw)}'
  $s.Arguments = '"{q(script)}"'
  $s.WorkingDirectory = '{q(APP_DIR)}'
  $s.IconLocation = '{q(icon)}'
  $s.Description = 'Live translated subtitles'
  $s.Save()
}}
"""
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps], check=True)
    print("       Shortcuts created on your Desktop and in the Start menu.")


def main():
    if "--probe-sv-gpu" in sys.argv:
        return probe_sensevoice_gpu()
    if "--prepare" in sys.argv:
        return prepare()
    if "--install-shortcuts" in sys.argv:
        return install_shortcuts()

    if sys.platform == "win32":
        try:  # crisp text on high-DPI screens
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    App().run()


if __name__ == "__main__":
    main()
