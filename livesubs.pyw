"""
LiveSubs - real-time subtitles for whatever your PC is playing.

Speech -> text:   Qwen3-ASR and SenseVoice (Chinese, English, Korean, Japanese and more), or Whisper
Text -> captions: a local AI model (Qwen3, run by a built-in llama.cpp engine on your GPU) that translates for
                  natural meaning into English or another language, using the previous lines as context

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
TEXT_KEY = "#0f0f0f"  # text-layer background that Windows makes see-through (close to BG so letter edges blend in)
DEFAULT_AUDIO = "Default speakers"

ENGINES = {
    "Best accuracy: Qwen3-ASR + SenseVoice (mixed languages, noise)": "qwen3",
    "Fastest: SenseVoice only": "sensevoice",
    "Whisper large-v3 (all languages)": "whisper-large-v3",
    "Whisper medium (all languages)": "whisper-medium",
    "Whisper small (all languages, light)": "whisper-small",
}
DEVICES = {"Auto": "auto", "GPU": "cuda", "CPU": "cpu"}
TRANSLATORS = {
    "Qwen3 14B (best)": "qwen3-14b",
    "Qwen3 8B (faster, smaller download)": "qwen3-8b",
    "Off": "off",
}
SHOW_MODES = {"Original + translation": "both", "Translation only": "english", "Original only": "original"}

# What the captions are translated into. (Scripts Tk can't lay out properly, like Arabic, are left out.)
TARGETS = [
    ("English", "en"), ("Chinese, Simplified (简体中文)", "zh"), ("Chinese, Traditional (繁體中文)", "zh-Hant"),
    ("Korean (한국어)", "ko"), ("Japanese (日本語)", "ja"), ("Spanish (Español)", "es"),
    ("French (Français)", "fr"), ("German (Deutsch)", "de"), ("Portuguese (Português)", "pt"),
    ("Italian (Italiano)", "it"), ("Russian (Русский)", "ru"), ("Vietnamese (Tiếng Việt)", "vi"),
    ("Thai", "th"), ("Indonesian (Bahasa Indonesia)", "id"), ("Filipino (Tagalog)", "tl"),
    ("Turkish (Türkçe)", "tr"),
]
TARGET_SHORT = {"en": "English", "zh": "Chinese", "zh-Hant": "Chinese", "ko": "Korean", "ja": "Japanese",
                "es": "Spanish", "fr": "French", "de": "German", "pt": "Portuguese", "it": "Italian",
                "ru": "Russian", "vi": "Vietnamese", "th": "Thai", "id": "Indonesian", "tl": "Filipino",
                "tr": "Turkish"}
# how the translator is told which language to write
TARGET_PROMPT_NAME = dict(TARGET_SHORT, **{
    "zh": "Simplified Chinese (everyday Mandarin as spoken in mainland China)",
    "zh-Hant": "Traditional Chinese (everyday Mandarin as spoken in Taiwan)",
    "tl": "Filipino (Tagalog)"})

LANGUAGES = [
    ("Auto-detect", ""), ("Chinese (Mandarin)", "zh"), ("Cantonese", "yue"), ("Korean", "ko"),
    ("Japanese", "ja"), ("English", "en"), ("Spanish", "es"), ("French", "fr"), ("German", "de"),
    ("Russian", "ru"), ("Portuguese", "pt"), ("Italian", "it"), ("Arabic", "ar"), ("Hindi", "hi"),
    ("Thai", "th"), ("Vietnamese", "vi"), ("Indonesian", "id"), ("Turkish", "tr"),
    ("Polish", "pl"), ("Dutch", "nl"), ("Ukrainian", "uk"), ("Tagalog", "tl"),
    ("Hebrew", "he"), ("Greek", "el"),
]
LANG_NAME = {code: name for name, code in LANGUAGES if code}
LANG_NAME["other"] = "Other language"  # Latin-script speech that isn't English, language not set
SENSEVOICE_LANGS = {"zh", "yue", "ko", "ja", "en"}
EAST_ASIAN = {"zh", "yue", "ja", "ko"}
LATIN_SCRIPT = {"en", "es", "fr", "de", "pt", "it", "id", "tr", "pl", "nl", "tl", "vi",
                "ms", "sv", "da", "fi", "cs", "ro", "hu"}


def spoken_label(code):
    """How a choice appears under Spoken language. English is always allowed next to the language
    picked, because people mix it in (English words in Mandarin, an English-speaking teammate...)."""
    if not code:
        return "Auto-detect"
    return "English only" if code == "en" else f"{LANG_NAME[code]} + English"


def allowed_languages(picked):
    """What speech can be recognised as: the language picked under Spoken language, plus English.
    None = any language (auto-detect)."""
    return {picked, "en"} if picked else None


# The languages of the last few dozen clear lines this run of LiveSubs has heard (for auto-detect)
HEARD_LANGUAGES = collections.deque(maxlen=40)


def note_language(text, lang):
    """Counts a finished line towards the languages being spoken, if it's long enough to be sure of
    (short lines are the ones that get the wrong language)."""
    cjk = len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", text))
    words = len(re.findall(r"[^\W\d_\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]{2,}", text))
    if lang and lang != "other" and (cjk >= 6 or words >= 4):
        HEARD_LANGUAGES.append(lang)


def usual_languages():
    """The languages being spoken, most heard first: each one at least 2 of the recent clear lines and
    at least 15% of them, so a few early mistakes fade out."""
    heard = list(HEARD_LANGUAGES)  # (lines are added from two threads)
    return [c for c, n in collections.Counter(heard).most_common() if n >= 2 and n >= 0.15 * len(heard)]


def language_family(code):
    """Languages that get taken for one another: Mandarin, Cantonese, Japanese and Korean; English and
    the other European languages; Russian and Ukrainian."""
    return next((f for f in (EAST_ASIAN, LATIN_SCRIPT, {"ru", "uk"}) if code in f), {code})


def closest_expected(heard, expect, strict):
    """Without Qwen3-ASR's odds: a guess at what speech heard as `heard` (not expected) really was,
    or None to keep it as heard."""
    same = [c for c in expect if c in language_family(heard)]
    if same:  # e.g. Mandarin heard as Cantonese, Japanese or Korean, or English heard as German
        return same[0]
    return expect[0] if strict and expect else None


def name_odds(top, code):
    """Qwen3-ASR's log-odds for language `code` where it names the language: the best of the candidate
    tokens that are its name or the start of it (' Cant' for Cantonese). None if it isn't a candidate."""
    names = [n for n, c in QWEN3_LANGS.items() if c == code]
    best = None
    for t in top:
        piece = (t.get("token") or "").strip().lower()
        if piece and any(n == piece or (len(piece) >= 3 and n.startswith(piece)) for n in names):
            best = t["logprob"] if best is None else max(best, t["logprob"])
    return best


def fits_language(text, det, picked):
    """Could this line be in the picked Spoken language or English? Judged by its writing, so
    Chinese characters alone fit Mandarin, Cantonese and Japanese alike."""
    if not picked or not has_foreign_script(text):  # plain Latin letters: English at least
        return True
    lang = text_language(text, det or picked)
    if lang in (picked, "en"):
        return True
    if lang in ("zh", "yue"):
        return picked in ("zh", "yue", "ja")
    if lang in ("ru", "uk"):
        return picked in ("ru", "uk")
    return False

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
    "engine": "qwen3", "device": "auto", "translator": "qwen3-14b",
    "topic": "", "words": "", "hide_english": False, "asr_v": 2,
    "language": "", "show": "both", "audio_device": "", "app_mode": "all", "apps": [],
    "font": 20, "opacity": 0.75, "width": 0.5, "hold": 7.0,
    "threshold": 0.008, "silence": 0.6, "max_seg": 8.0,
    "click_through": False, "autostart": True, "pos": None,
    "history": 2, "speakers": True, "layout_v": 2,
    "game_mode": True, "target": "en", "both_ways": False,
    "heard_languages": [],  # languages of recent clear lines, so auto-detect knows them from the start
}

# Chat logs of this many sessions stay on disk (a session = each time LiveSubs is opened);
# older ones delete themselves.
KEEP_SESSIONS = 2

_ASR_REL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
SV_NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
SV_URL = f"{_ASR_REL}/{SV_NAME}.tar.bz2"
SV_DIR = os.path.join(MODELS_DIR, SV_NAME)
# full-precision SenseVoice: the GPU needs this one (the compact int8 model only runs well on a CPU)
SVF_NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
SVF_URL = f"{_ASR_REL}/{SVF_NAME}.tar.bz2"
SVF_DIR = os.path.join(MODELS_DIR, SVF_NAME)
# Qwen3-ASR: second, more careful listen of each finished line (mixed languages, background noise)
Q3_NAME = "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"
Q3_URL = f"{_ASR_REL}/{Q3_NAME}.tar.bz2"
Q3_DIR = os.path.join(MODELS_DIR, Q3_NAME)
# Silero VAD: tells speech apart from game sounds / music
VAD_URL = f"{_ASR_REL}/silero_vad.onnx"
VAD_PATH = os.path.join(MODELS_DIR, "silero_vad.onnx")

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
    "- Speakers often mix languages in one sentence (for example Mandarin with English words). "
    "Keep the English parts as they were said and blend everything into ONE natural English line. "
    "An English word may have been written as similar-sounding Chinese characters by mistake.\n"
    "- If you are given two versions of the line from two speech recognizers, they heard the same "
    "audio and may each be partly wrong: work out what was really said from both, then translate that.\n"
    "- Use the earlier lines only as context (who's talking, what's going on). Keep names as names.\n"
    "- Keep it short and readable like a real subtitle.\n"
    "\nExamples (literal -> how you should translate):\n"
    "你在搞什么鬼？  NOT \"What ghost are you doing?\" -> \"What the hell are you doing?\"\n"
    "我真是服了你了。  NOT \"I am really convinced by you.\" -> \"Unbelievable. You're something else.\"\n"
    "绝了！  NOT \"Extinct!\" -> \"That's insane!\"\n"
    "666  -> \"Nice!\"      yyds -> \"GOAT.\"      别装了 -> \"Quit acting.\"\n"
    "这个 bug 太离谱了，我直接 carry 全场。  -> \"This bug is ridiculous. I straight-up carried the whole game.\"\n"
    "今天 meeting 取消了，boss 说 next week 再约。  -> \"Today's meeting got cancelled. Boss says we'll reschedule next week.\"\n"
    "versions \"这个是平凡的 frequently\" / \"就是频繁的 frequent\"  -> \"This one means 'frequently'.\"\n"
    "미쳤다 진짜  NOT \"Went crazy really\" -> \"That's insane, for real.\"\n"
    "아 진짜 짜증나  -> \"Ugh, so annoying.\"      대박 -> \"No way!\"      ㅋㅋㅋ -> \"lol\"\n"
    "マジでヤバいって！  NOT \"Seriously it is dangerous!\" -> \"Dude, this is seriously bad!\"\n"
    "うるさいな、黙ってろ。  -> \"Ugh, shut up already.\"      なんでやねん -> \"Oh, come on!\"\n"
)
TRANSLATE_PROMPT_ANY = (
    "You are a professional subtitle translator for streaming shows, anime, dramas, games and "
    "livestreams. Translate the new line into {name} the way a native speaker would actually say it "
    "in that situation - NOT word-for-word.\n"
    "Rules:\n"
    "- Match the register exactly: casual stays casual, rude stays rude, cute stays cute, "
    "formal stays formal. Use everyday spoken {short}.\n"
    "- Turn slang, internet slang and idioms into the natural {short} equivalent. Never translate "
    "an idiom or slang term literally.\n"
    "- Keep swearing and insults at the same strength - don't censor or soften them.\n"
    "- The line comes from automatic speech recognition and may have wrong words, homophones or "
    "missing words: translate what the speaker most likely MEANT.\n"
    "- Speakers often mix languages in one sentence. Blend everything into ONE natural {short} line. "
    "Names, and game terms or callouts that {short}-speaking players normally say in English, can stay "
    "in English.\n"
    "- If you are given two versions of the line from two speech recognizers, they heard the same "
    "audio and may each be partly wrong: work out what was really said from both, then translate that.\n"
    "- Use the earlier lines only as context (who's talking, what's going on). Keep names as names.\n"
    "- Keep it short and readable like a real subtitle.\n"
)
TRANSLATE_EXTRA = {  # script rules and a few examples of the tone wanted
    "zh": "- Write Simplified Chinese characters only, with Chinese punctuation.\n"
          "\nExamples (literal -> how you should translate):\n"
          "What the hell are you doing?  NOT \"你在做什么地狱？\" -> \"你在搞什么鬼？\"\n"
          "That's insane!  -> \"绝了！\"      No way!  -> \"不是吧！\"      lol  -> \"哈哈哈\"\n"
          "Nice shot!  -> \"好枪！\"      I'm one shot, someone heal me.  -> \"我残血了，谁奶我一口。\"\n"
          "GG, we got thrown.  -> \"GG，这把被演了。\"      Bro, you're actually so bad.  -> \"兄弟，你是真的菜。\"\n",
    "zh-Hant": "- Write Traditional Chinese characters only (Taiwan usage), with Chinese punctuation.\n"
               "\nExamples (literal -> how you should translate):\n"
               "What the hell are you doing?  NOT \"你在做什麼地獄？\" -> \"你在搞什麼鬼？\"\n"
               "That's insane!  -> \"太扯了吧！\"      No way!  -> \"不會吧！\"      lol  -> \"笑死\"\n"
               "Nice shot!  -> \"漂亮！\"      I'm one shot, someone heal me.  -> \"我殘血，誰來補我一下。\"\n"
               "GG, we got thrown.  -> \"GG，這場被隊友雷了。\"      Bro, you're actually so bad.  -> \"兄弟，你真的很爛欸。\"\n",
    "ko": "- Match how the speaker talks: 반말 for casual speech, 존댓말 for polite speech.\n"
          "\nExamples (literal -> how you should translate):\n"
          "What the hell are you doing?  -> \"너 지금 뭐 하는 거야?\"      That's insane!  -> \"미쳤다!\"\n"
          "Nice shot!  -> \"나이스!\"      lol  -> \"ㅋㅋㅋ\"      GG, we got thrown.  -> \"GG, 이번 판 트롤 당했네.\"\n",
    "ja": "\nExamples (literal -> how you should translate):\n"
          "What the hell are you doing?  -> \"何やってんだよ？\"      That's insane!  -> \"ヤバすぎ！\"\n"
          "Nice shot!  -> \"ナイス！\"      lol  -> \"ｗｗｗ\"      GG, we got thrown.  -> \"GG、戦犯いたわ。\"\n",
}


def translation_prompt(target):
    """System prompt for translating speech into the caption language."""
    if target == "en":
        return TRANSLATE_PROMPT
    name, short = TARGET_PROMPT_NAME.get(target, target), TARGET_SHORT.get(target, target)
    return TRANSLATE_PROMPT_ANY.format(name=name, short=short) + TRANSLATE_EXTRA.get(target, "")


def reply_format(target, fix):
    if target == "en":
        return FORMAT_FIX if fix else FORMAT_ENGLISH
    name = TARGET_PROMPT_NAME.get(target, target)
    if fix:
        return FORMAT_FIX.replace("ENGLISH: <the natural English subtitle>", f"TRANSLATION: <the natural {name} subtitle>")
    return f"\nReply with ONLY the {name} subtitle - no quotes, notes, labels, romanization or pinyin."


FORMAT_ENGLISH = "\nReply with ONLY the English subtitle - no quotes, notes, labels or romanization."
FORMAT_FIX = (
    "\nReply in exactly this format and nothing else:\n"
    "ENGLISH: <the natural English subtitle>\n"
    "FIXED: <the new line in its original language with word spacing and obvious "
    "speech-recognition mistakes corrected; otherwise unchanged>"
)
FIX_LANGS = {"ko"}  # SenseVoice's Korean word spacing needs repairing; other languages don't

# Qwen3-VL's own text-spotting prompt (from Qwen's OCR cookbook); boxes come back on a 0-999 scale
SPOT_PROMPT = ("Spotting all the text in the image with line-level, and output in JSON format as "
               "[{'bbox_2d': [x1, y1, x2, y2], 'text_content': 'text'}, ...].")
SCREEN_TRANSLATE_PROMPT = (
    "You translate text read from a screenshot (game UI, menus, chat, subtitles, signs, manga) into natural "
    "English. Each numbered item is one line or label, possibly misread by OCR: translate what it most likely "
    "says. Keep buttons and labels short, keep names as names, and keep slang and tone (casual stays casual, "
    "rude stays rude). If an item is already English, repeat it unchanged.\n"
    "Reply with exactly one line per item, numbered the same way:\n1. <English>\n2. <English>\n"
    "No notes, no quotes, no extra lines."
)
VISION_MODEL = "qwen3-vl-8b"


def screen_prompt(target):
    if target == "en":
        return SCREEN_TRANSLATE_PROMPT
    name, short = TARGET_PROMPT_NAME.get(target, target), TARGET_SHORT.get(target, target)
    return (f"You translate text read from a screenshot (game UI, menus, chat, subtitles, signs, manga) into "
            f"natural {name}. Each numbered item is one line or label, possibly misread by OCR: translate what "
            "it most likely says. Keep buttons and labels short, keep names as names, and keep slang and tone "
            f"(casual stays casual, rude stays rude). If an item is already in {short}, repeat it unchanged.\n"
            "Reply with exactly one line per item, numbered the same way:\n1. <translation>\n2. <translation>\n"
            "No notes, no quotes, no extra lines.")

_line_ids = itertools.count(1)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


_log_lock = threading.Lock()
_fresh_logs = set()


def open_log(path, binary=False):
    """Opens a log for a helper program to write to. Like livesubs.log, each log starts empty the
    first time it's used in a run, so the logs never pile up."""
    with _log_lock:
        if path not in _fresh_logs:
            _fresh_logs.add(path)
            try:
                open(path, "w").close()
            except OSError:
                pass
    if binary:
        return open(path, "ab")
    return open(path, "a", encoding="utf-8", errors="replace")


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


def set_gpu_priority(proc=None, low=True):
    """Game mode: ask Windows to schedule this process's GPU work below the game's frames
    (D3DKMT scheduling priority class). proc=None means LiveSubs itself."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        fn = ctypes.WinDLL("gdi32").D3DKMTSetProcessSchedulingPriorityClass
        fn.argtypes = [wintypes.HANDLE, ctypes.c_int]
        fn.restype = ctypes.c_long
        if proc is None:
            k32 = ctypes.WinDLL("kernel32")
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            handle = k32.GetCurrentProcess()
        else:
            handle = int(proc._handle)
        status = fn(handle, 1 if low else 2)  # 1 = below normal, 2 = normal
        if status != 0:
            log(f"GPU priority unchanged (status 0x{status & 0xFFFFFFFF:08X})")
    except Exception as e:
        log("GPU priority not available:", e)


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
    if cfg["translator"] == "qwen3-vl-8b":  # no longer offered as a translator: the closest choice
        cfg["translator"] = "qwen3-8b"
    if cfg["translator"] not in TRANSLATORS.values():
        cfg["translator"] = DEFAULTS["translator"]
    if cfg.get("target") not in dict(TARGETS).values():
        cfg["target"] = "en"
    heard = cfg.get("heard_languages")
    cfg["heard_languages"] = ([c for c in heard if isinstance(c, str) and c in LANG_NAME][-40:]
                              if isinstance(heard, list) else [])
    if saved and saved.get("asr_v", 1) < 2 and cfg["engine"] == "sensevoice":
        cfg["engine"] = "qwen3"  # one-time switch to the more accurate two-pass speech recognition
    cfg["asr_v"] = 2
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


_PUNCT = re.compile(r"[\s。，、！？!?,.…~～\-·:：;；\"'“”‘’()（）]+")
ENGLISH_COMMON = set(
    "the be to of and a in that have i it for not on with he as you do at this but his by from they we say "
    "her she or an will my one all would there their what so up out if about who get which go me when make "
    "can like time no just him know take people into year your good some could them see other than then now "
    "look only come its over think also back after use two how our work first well way even new want because "
    "any these give day most us is are was were am im dont thats yes yeah ok okay oh hey nice here let lets "
    "got going right wait why where did does has had been youre theyre cant wont didnt need help someone "
    "should gonna wanna really very much more guys guy bro dude thanks thank please sorry kill killed push "
    "enemy team game play shot round".split())
# short words that are also everyday Spanish/French/German/Italian/Portuguese - not evidence either way
AMBIGUOUS_WORDS = set("a i no me he in on to do so us an be die son come".split())
# everyday words of other Latin-script languages that English speech practically never contains
OTHER_COMMON = set(
    "que de la el y en es un una por para lo se su le les des du et est je tu il nous vous pas ne der das und "
    "ist nicht ich sie wir ein eine zu auf che di sono eu uma mas muy pero como esta este esto bien gracias "
    "hola ser tengo tiene donde cuando porque nada vamos eso oye mira pues vale oui merci avec dans sur mais "
    "cest jai suis fait veux sais rien allez ce cette qui quoi nein danke bitte sehr auch noch doch schon "
    "jetzt ciao andiamo bene grazie cosa questo isso tudo obrigado ik het niet een wat dat maar zijn mijn "
    "saya tidak apa ini itu yang aku kamu sudah hindi ko ang ng mga ako ikaw talaga naman".split())


def has_foreign_script(text):
    """True if the text contains letters that aren't Latin (Chinese, Korean, Japanese, Cyrillic, Arabic...)."""
    return any(ch.isalpha() and ord(ch) > 0x24F for ch in text)


def text_language(text, hint="en"):
    """Language of a line judged by what was actually written, with the recognizer's guess as a tie-break."""
    if re.search(r"[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]", text):
        return "ko"
    if re.search(r"[\u3040-\u30ff]", text):
        return "ja"
    if re.search(r"[\u3400-\u9fff\uf900-\ufaff]", text):
        return "yue" if hint == "yue" else "zh"
    for pattern, code in ((r"[\u0400-\u04ff]", "uk" if hint == "uk" else "ru"), (r"[\u0600-\u06ff]", "ar"),
                          (r"[\u0e00-\u0e7f]", "th"), (r"[\u0590-\u05ff]", "he"), (r"[\u0370-\u03ff]", "el"),
                          (r"[\u0900-\u097f]", "hi"), (r"[\u1ea0-\u1ef9]", "vi")):
        if re.search(pattern, text):
            return code
    if has_foreign_script(text):
        return hint
    return hint if hint not in ("zh", "yue", "ja", "ko") else "en"


def looks_english(text):
    """Latin-script text counts as English unless there's evidence of another language
    (so game callouts and names like "Jett dash B main" stay English)."""
    if re.search(r"[\u00c0-\u024f¿¡]", text):
        return False  # accented letters: Spanish, French, German, Polish, Turkish, Vietnamese...
    words = [w.replace("'", "") for w in re.findall(r"[a-z']+", text.lower())]
    english = sum(w in ENGLISH_COMMON and w not in AMBIGUOUS_WORDS for w in words)
    other = sum(w in OTHER_COMMON for w in words)
    return other == 0 or english > other


def needs_translation(text, det, trust=False):
    """Decided from the text itself: a 'mostly English' tag on a mixed line must not skip the Chinese in it.
    trust=True: det is reliable (Whisper's guess, or a language the user picked), so follow it."""
    if has_foreign_script(text):
        return True
    if trust:
        return det != "en"
    words = re.findall(r"[A-Za-zÀ-ɏ']+", text)
    return len(words) >= 3 and not looks_english(text)


_HAN = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
_KANA = re.compile(r"[\u3040-\u30ff]")
_HANGUL = re.compile(r"[\uac00-\ud7af]")
_SCRIPT_OF = {"ru": re.compile(r"[\u0400-\u04ff]"), "th": re.compile(r"[\u0e00-\u0e7f]")}


def in_language(text, lang, target, trust=False):
    """Is the line already in the caption language (so it's shown as it is, not translated)?
    lang: what the line was recognised as. A few words of English mixed in don't change the answer."""
    if target == "en":
        return not needs_translation(text, lang, trust)
    latin_words = len(re.findall(r"[A-Za-z\u00c0-\u024f']+", text))
    if target in ("zh", "zh-Hant"):
        if _KANA.search(text) or _HANGUL.search(text) or lang == "yue":
            return False  # Japanese, Korean - or Cantonese, which Mandarin speakers get translated
        han = len(_HAN.findall(text))
        return han > 0 and han >= latin_words
    if target == "ja":
        kana, han = len(_KANA.findall(text)), len(_HAN.findall(text))
        return (kana > 0 or (lang == "ja" and han > 0)) and not _HANGUL.search(text) and kana + han >= latin_words
    if target == "ko":
        hangul = len(_HANGUL.findall(text))
        return hangul > 0 and hangul >= latin_words
    if target in _SCRIPT_OF:
        letters = len(_SCRIPT_OF[target].findall(text))
        return letters > 0 and letters >= latin_words
    # Latin-script caption languages (Spanish, French, Vietnamese...): only the recogniser can tell
    non_latin = any(ch.isalpha() and ord(ch) > 0x24F and not 0x1E00 <= ord(ch) <= 0x1EFF for ch in text)
    return not non_latin and lang == target


def caption_target(text, lang, cfg, trust=False):
    """The language this line should be captioned in, or None if it's fine as it is.
    Both ways: lines already in the caption language get English captions instead."""
    target = cfg.get("target") or "en"
    if not in_language(text, lang, target, trust):
        return target
    if cfg.get("both_ways") and target != "en":
        return "en"
    return None


def runaway_text(text):
    """Speech models occasionally get stuck repeating themselves; such output can't be trusted."""
    t = _PUNCT.sub("", text)
    return bool(re.search(r"(.{2,12}?)\1{4,}", t))


def parse_words(raw):
    """'Jett, Sage，spike' -> ['Jett', 'Sage', 'spike'] (max 30 short entries)."""
    items = [w.strip() for w in re.split(r"[,，;；\n]+", raw or "")]
    return [w for w in items if 0 < len(w) <= 30][:30]


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
def use_windows_certificates():
    """Check HTTPS certificates through Windows itself, the way browsers and pip do. On its own, Python only
    trusts the root certificates already saved on the PC, and Windows saves many of those only once a
    browser first needs one - so on a fresh PC, downloads from GitHub or Hugging Face could fail with
    CERTIFICATE_VERIFY_FAILED. This also trusts certificates that antivirus or a work/school network added."""
    import importlib

    for name in ("truststore", "pip._vendor.truststore"):  # pip ships its own copy as a fallback
        try:
            importlib.import_module(name).inject_into_ssl()
            return True
        except Exception:
            continue
    log("Couldn't use Windows' certificate checks; using Python's own.")
    return False


def certificate_error(e):
    import ssl

    reason = getattr(e, "reason", None)
    return (isinstance(e, ssl.SSLCertVerificationError) or isinstance(reason, ssl.SSLCertVerificationError)
            or "CERTIFICATE_VERIFY_FAILED" in str(e))


CERT_HELP = ("Couldn't confirm the download site's security certificate (CERTIFICATE_VERIFY_FAILED). "
             "This is usually antivirus that scans secure connections, or a school/work network. Try "
             "again with the antivirus' web/HTTPS scanning paused, or on another network.")


def download(url, dest, status, label, attempts=5):
    """Downloads with a progress %, and resumes from where it stopped if the connection drops."""
    tmp = dest + ".part"
    for attempt in range(attempts):
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:  # server ignored the resume request: start over
                    have = 0
                total = int(r.headers.get("Content-Length") or 0) + have
                done, last = have, -1
                with open(tmp, "ab" if have else "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            pct = done * 100 // total
                            if pct != last:
                                last = pct
                                size = f" ({done / 1e9:.1f} / {total / 1e9:.1f} GB)" if total > 2e8 else ""
                                status(f"Downloading {label}... {pct}%{size}")
            if total and os.path.getsize(tmp) < total:
                raise IOError("download ended early")
            os.replace(tmp, dest)
            return
        except urllib.error.HTTPError as e:
            if e.code == 416 and have:  # already have every byte
                os.replace(tmp, dest)
                return
            if attempt == attempts - 1:
                raise
        except Exception as e:
            if certificate_error(e):  # retrying won't help
                raise RuntimeError(CERT_HELP) from e
            if attempt == attempts - 1:
                raise
        status(f"Connection dropped - resuming {label}...")
        time.sleep(2 + attempt * 3)


def fetch_archive(url, name, wanted, status, label):
    """Download a model .tar.bz2 into models/ and unpack only the files we need."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    archive = os.path.join(MODELS_DIR, name + ".tar.bz2")
    download(url, archive, status, f"{label} (first time only)")
    status(f"Unpacking {label}...")
    with tarfile.open(archive, "r:bz2") as t:
        members = [m for m in t.getmembers() if wanted(m.name)]
        try:
            t.extractall(MODELS_DIR, members=members, filter="data")
        except TypeError:
            t.extractall(MODELS_DIR, members=members)
    os.remove(archive)


def ensure_sensevoice(status, full_precision=False):
    if full_precision:
        model, tokens = os.path.join(SVF_DIR, "model.onnx"), os.path.join(SVF_DIR, "tokens.txt")
        if not (os.path.exists(model) and os.path.exists(tokens)):
            fetch_archive(SVF_URL, SVF_NAME, lambda n: n.endswith(("/model.onnx", "/tokens.txt")),
                          status, "SenseVoice for the GPU")
        return model, tokens
    model, tokens = os.path.join(SV_DIR, "model.int8.onnx"), os.path.join(SV_DIR, "tokens.txt")
    if not (os.path.exists(model) and os.path.exists(tokens)):
        fetch_archive(SV_URL, SV_NAME, lambda n: n.endswith(("model.int8.onnx", "tokens.txt")),
                      status, "SenseVoice")
    return model, tokens


def ensure_qwen3asr(status):
    files = {"conv_frontend": os.path.join(Q3_DIR, "conv_frontend.onnx"),
             "encoder": os.path.join(Q3_DIR, "encoder.int8.onnx"),
             "decoder": os.path.join(Q3_DIR, "decoder.int8.onnx"),
             "tokenizer": os.path.join(Q3_DIR, "tokenizer")}
    need = list(files.values())[:3] + [os.path.join(Q3_DIR, "tokenizer", "vocab.json")]
    if not all(os.path.exists(f) for f in need):
        fetch_archive(Q3_URL, Q3_NAME, lambda n: n.endswith(".onnx") or "/tokenizer" in n,
                      status, "Qwen3-ASR (0.9 GB)")
    return files


def ensure_vad(status):
    if not os.path.exists(VAD_PATH):
        os.makedirs(MODELS_DIR, exist_ok=True)
        download(VAD_URL, VAD_PATH, status, "voice detector")
    return VAD_PATH


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
        if cached.get("version") == version and cached.get("model") == "full":
            return bool(cached.get("ok"))
    except Exception:
        pass
    ok, detail = False, ""
    try:
        ensure_sensevoice(status, full_precision=True)
        status("Testing the GPU speech engine (first time only)...")
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
            json.dump({"version": version, "model": "full", "ok": ok}, f)
    except Exception:
        pass
    return ok


def probe_sensevoice_gpu():
    """Child process for sensevoice_gpu_ok(): load on CUDA, decode, report timing."""
    import sherpa_onnx

    model, tokens = ensure_sensevoice(lambda s: None, full_precision=True)
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


NON_SPEECH_EVENTS = {"BGM", "Applause", "Laughter", "Cry", "Sneeze", "Breath", "Cough"}


class SenseVoiceASR:
    kind = "sensevoice"
    name = "SenseVoice"
    live_step = 0.4
    refiner = None  # optional second, more careful recognizer (Qwen3-ASR)

    def __init__(self, status, game_mode=True, device="auto"):
        import sherpa_onnx

        self._so = sherpa_onnx
        self.cpu_model, self.tokens = ensure_sensevoice(status)
        # The GPU whenever it works: games like Valorant are held back by the processor, not the graphics card
        want_gpu = device in ("cuda", "auto")
        use_gpu = want_gpu and sensevoice_gpu_ok(status)
        if device == "cuda" and not use_gpu:
            status("GPU speech engine not available - using CPU (run the installer to add it)")
            log("SenseVoice: GPU requested but not available; using CPU")
        self.device = "cuda" if use_gpu else "cpu"
        status(f"Loading SenseVoice on {'GPU' if use_gpu else 'CPU'}...")
        # on the CPU, 2 threads is plenty for real-time speech; more just steals cores from the game
        self.threads = 2 if game_mode else max(2, min(8, (os.cpu_count() or 4) // 2))
        if use_gpu:
            model, _ = ensure_sensevoice(status, full_precision=True)
            self.rec = self._make(model, "auto", 1, "cuda")
        else:
            self.rec = self._make(self.cpu_model, "auto", self.threads, "cpu")
        self.forced = {}  # language -> recognizer told to expect that language
        self.lock = threading.Lock()  # a restart can briefly have two listeners using this model
        self.transcribe(np.zeros(WHISPER_SR, np.float32), "", True)  # warm-up

    def _make(self, model, language, threads, provider):
        return self._so.OfflineRecognizer.from_sense_voice(
            model=model, tokens=self.tokens, num_threads=threads, use_itn=True, language=language,
            provider=provider)

    def _decode(self, rec, audio16):
        s = rec.create_stream()
        s.accept_waveform(WHISPER_SR, audio16)
        rec.decode_stream(s)
        r = s.result
        lang = (r.lang or "").strip("<|>") or "zh"
        event = (getattr(r, "event", "") or "").strip("<|>")
        text = r.text.strip()
        letters = _PUNCT.sub("", text)
        if is_junk(text) or (event in NON_SPEECH_EVENTS and len(letters) < 4):
            text = ""  # music, laughter, a stray syllable...
        elif len(letters) <= 1 and len(audio16) > WHISPER_SR * 1.5:
            text = ""  # one character from 1.5+ seconds of sound is noise, not a word
        return text, lang

    def transcribe(self, audio16, language, final):
        """language: the 'Spoken language' setting ('' = auto). English is always allowed alongside it,
        so mixed speech works; any OTHER guess (e.g. Mandarin heard as Cantonese) is re-done as that language."""
        with self.lock:
            text, lang = self._decode(self.rec, audio16)
            if language in SENSEVOICE_LANGS and lang not in (language, "en"):
                if language not in self.forced:
                    self.forced = {language: self._make(self.cpu_model, language, self.threads, "cpu")}
                text, _ = self._decode(self.forced[language], audio16)
                lang = language
        return text, lang


class Qwen3Refiner:
    """Second listen with Qwen3-ASR: slower, but trained for mixed Mandarin/English and noisy audio."""

    def __init__(self, status, game_mode=True, words=()):
        import sherpa_onnx

        f = ensure_qwen3asr(status)
        status("Loading Qwen3-ASR...")
        self.words = list(words)
        self.rec = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
            conv_frontend=f["conv_frontend"], encoder=f["encoder"], decoder=f["decoder"],
            tokenizer=f["tokenizer"], num_threads=2 if game_mode else 4, max_new_tokens=128,
            hotwords=",".join(w.replace(",", " ") for w in self.words))
        self.lock = threading.Lock()
        self.transcribe(np.zeros(WHISPER_SR, np.float32))  # warm-up

    def transcribe(self, audio16):
        """Returns the text, or '' when there was no real speech or the result can't be trusted."""
        with self.lock:
            s = self.rec.create_stream()
            s.accept_waveform(WHISPER_SR, audio16[:WHISPER_SR * 30])
            self.rec.decode_stream(s)
            text = (s.result.text or "").strip()
        return trusted_asr_text(text, len(audio16) / WHISPER_SR)

    def refine(self, audio16, expect=None, strict=True):
        """Second-listen interface, like Qwen3GpuASR.refine. This version can't be told which language
        to expect or say which one it heard, so the caller checks the writing instead."""
        return self.transcribe(audio16), ""


def trusted_asr_text(text, seconds):
    """Qwen3-ASR output, or '' when there was no real speech or it can't be trusted."""
    if text.lower() in ("language", "language none") or is_junk(text) or runaway_text(text):
        return ""
    if len(text) > 40 + seconds * 25:  # far more text than anyone can say in that time
        return ""
    return text


# Qwen3-ASR names the language before the text: "language Chinese<asr_text>..."
QWEN3_LANGS = {
    "chinese": "zh", "mandarin": "zh", "cantonese": "yue", "english": "en", "japanese": "ja", "korean": "ko",
    "spanish": "es", "french": "fr", "german": "de", "russian": "ru", "portuguese": "pt", "italian": "it",
    "arabic": "ar", "hindi": "hi", "thai": "th", "vietnamese": "vi", "indonesian": "id", "turkish": "tr",
    "polish": "pl", "dutch": "nl", "greek": "el", "filipino": "tl", "tagalog": "tl", "ukrainian": "uk",
    "hebrew": "he", "malay": "ms", "swedish": "sv", "danish": "da", "finnish": "fi", "czech": "cs",
    "persian": "fa", "romanian": "ro", "hungarian": "hu", "macedonian": "mk",
}
# Qwen3-ASR can be told which language to write (not Ukrainian or Hebrew: it doesn't know them)
QWEN3_FORCE = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese", "ja": "Japanese", "ko": "Korean", "es": "Spanish",
    "fr": "French", "de": "German", "ru": "Russian", "pt": "Portuguese", "it": "Italian", "ar": "Arabic",
    "hi": "Hindi", "th": "Thai", "vi": "Vietnamese", "id": "Indonesian", "tr": "Turkish", "pl": "Polish",
    "nl": "Dutch", "tl": "Filipino", "el": "Greek",
}


def parse_qwen3_asr(reply):
    """'language Chinese<asr_text>你好' -> ('你好', 'zh'). No speech -> ('', '')."""
    text = re.sub(r"<think>.*?</think>", "", reply or "", flags=re.S).strip()
    lang = ""
    m = re.match(r"language\s+([A-Za-z][A-Za-z \-]*?)\s*<asr_text>", text, flags=re.I)
    if m:
        name = m.group(1).strip().lower()
        lang = "" if name == "none" else QWEN3_LANGS.get(name, "")
        text = text[m.end():]
    elif "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[1]
    elif re.fullmatch(r"language\s+\w+", text, flags=re.I):
        text = ""  # only a language tag: nothing was said
    text = text.replace("</asr_text>", "").strip()
    return text, lang


def plain_asr_text(reply):
    """Qwen3-ASR's reply when it was told the language: just the words."""
    text = re.sub(r"<think>.*?</think>", "", reply or "", flags=re.S)
    if "<asr_text>" in text:  # it named a language anyway
        text = text.split("<asr_text>", 1)[1]
    return text.replace("</asr_text>", "").strip()


def wav_base64(audio16):
    """16 kHz float audio -> base64 of a 16-bit mono WAV file (what llama-server takes as input_audio)."""
    import base64
    import io
    import wave

    pcm = (np.clip(audio16, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(WHISPER_SR)
        w.writeframes(pcm)
    return base64.b64encode(buf.getvalue()).decode()


class Qwen3GpuASR:
    """Qwen3-ASR on the graphics card (run by llama.cpp, like the translator). Valorant and most competitive
    games are held back by the processor, not the graphics card, so in Game mode this is the only speech
    recognizer: all the heavy listening happens on the GPU and the processor is left to the game."""
    kind = "qwen3gpu"
    name = "Qwen3-ASR (GPU)"
    device = "cuda"
    live_step = 0.4
    refiner = None

    def __init__(self, status, words=()):
        if not model_ready(ASR_MODEL):
            ensure_model(ASR_MODEL, status)  # ~1 GB, first time only
        status("Loading Qwen3-ASR on the GPU...")
        ASR_HOST.use(ASR_MODEL, status)
        self.words = [w.replace(",", " ") for w in words]
        self.failures = 0
        self.listen(np.zeros(WHISPER_SR, np.float32), raise_errors=True)  # warm-up: the engine answers

    # On auto-detect, speech heard as a language that isn't expected is kept only if Qwen3-ASR is at
    # least 100 times surer of it than of every expected language (log-odds difference)
    NEW_LANGUAGE_MARGIN = 4.6

    def listen(self, audio16, expect=None, strict=True, raise_errors=False):
        """(text, language code) - text is '' for no speech, noise or an untrustworthy result.
        expect: the languages the speech should be in (None = any). Speech heard as another one
        (Mandarin taken for Cantonese or Japanese, say) is listened to again, told which expected
        language to write: always when strict (a Spoken language was picked), and on auto-detect
        unless Qwen3-ASR is far surer of the language it heard, i.e. a new language really is spoken."""
        audio16 = audio16[:WHISPER_SR * 30]
        seconds = len(audio16) / WHISPER_SR
        max_tokens = int(min(400, 40 + 12 * seconds))
        messages = []
        if self.words:  # Qwen3-ASR reads the system text as context: these names/terms get heard right
            messages.append({"role": "system", "content": ", ".join(self.words)})
        messages.append({"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": wav_base64(audio16), "format": "wav"}}]})
        try:
            text, lang = parse_qwen3_asr(self._ask(messages, max_tokens))
            if expect and text and lang and lang not in expect:
                redo = self._recheck(messages, expect, strict, lang)
                if redo:
                    # the reply starts the way Qwen3-ASR's own "language=" option makes it start
                    reply = self._ask(messages + [{"role": "assistant",
                                                   "content": f"language {QWEN3_FORCE[redo]}<asr_text>"}], max_tokens)
                    log(f"  heard as {LANG_NAME.get(lang, lang)}, listened again as {LANG_NAME[redo]}")
                    text, lang = plain_asr_text(reply), redo
            self.failures = 0
        except Exception as e:
            if raise_errors:
                raise
            self.failures += 1
            log("Qwen3-ASR (GPU) failed:", e)
            if self.failures >= 3:  # e.g. the engine crashed: start it again on the next line
                ASR_HOST.stop()
                self.failures = 0
            return "", ""
        return trusted_asr_text(text, seconds), lang

    @staticmethod
    def _ask(messages, max_tokens, **kw):
        return ASR_HOST.chat(ASR_MODEL, messages, max_tokens=max_tokens, temperature=0.0, timeout=30, **kw)

    def _recheck(self, messages, expect, strict, heard):
        """The expected language to listen again as, or None to keep what was heard. Decided by
        Qwen3-ASR's own odds for each language, at the point where it names the language."""
        if strict and any(c not in QWEN3_FORCE for c in expect):
            return None  # it can't be told the language picked (Ukrainian, Hebrew), so leave it be
        choices = [c for c in expect if c in QWEN3_FORCE]
        if not strict:  # auto-detect: only a mix-up with a similar language is fixed
            choices = [c for c in choices if c in language_family(heard)]
        if not choices:
            return None
        try:
            res = self._ask(messages + [{"role": "assistant", "content": "language"}], 1,
                            extra={"logprobs": True, "top_logprobs": 20}, full=True)
            top = res["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except Exception as e:
            log("Qwen3-ASR's language odds unavailable:", e)
            top = None
        if not top:
            guess = closest_expected(heard, choices, strict)
            return guess if guess in QWEN3_FORCE else None
        odds = {c: name_odds(top, c) for c in choices}
        odds = {c: lp for c, lp in odds.items() if lp is not None}
        if not odds:  # none of the expected languages is even in the running
            return closest_expected(heard, choices, True) if strict else None
        best = max(odds, key=odds.get)
        if strict:
            return best
        sure = max(t["logprob"] for t in top)  # the language it heard is its first choice
        return None if sure - odds[best] >= self.NEW_LANGUAGE_MARGIN else best

    def transcribe(self, audio16, language="", final=True, expect=None, strict=True):
        """Speech-model interface (Game mode: the only recognizer). Returns (text, language code)."""
        text, lang = self.listen(audio16, expect, strict)
        return text, lang or text_language(text, "en")

    def refine(self, audio16, expect=None, strict=True):
        """Second-listen interface (outside Game mode, after SenseVoice's quick version):
        (text, language code)."""
        return self.listen(audio16, expect, strict)


class SpeechGate:
    """Is someone talking right now? Uses the Silero voice detector, so game sound effects and
    constant background noise don't count as speech (a loudness check can't tell them apart)."""

    def __init__(self, status=log):
        import sherpa_onnx

        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = ensure_vad(status)
        cfg.silero_vad.threshold = 0.5
        cfg.silero_vad.min_silence_duration = 0.1
        cfg.silero_vad.min_speech_duration = 0.1
        cfg.sample_rate = WHISPER_SR
        self.model = sherpa_onnx.VadModel.create(cfg)
        self.win = self.model.window_size()
        self.lock = threading.Lock()
        self.reset(WHISPER_SR)

    def reset(self, sr):
        """sr: sample rate of the audio that will be fed in."""
        with self.lock:
            self._reset(sr)

    def _reset(self, sr):
        self.model.reset()
        g = gcd(sr, WHISPER_SR)
        self.up, self.down = WHISPER_SR // g, sr // g
        # resample in a streaming way: keep a little audio either side so chunk edges don't click
        self.ctx = self.down * max(1, 441 // self.down) if self.down > 1 else 0
        self.hist = np.zeros(2 * self.ctx, np.float32)
        self.tail = np.zeros(0, np.float32)

    def _to16k(self, chunk):
        if self.down == 1 and self.up == 1:
            return chunk
        x = np.concatenate([self.hist, chunk])
        y = resample_poly(x, self.up, self.down).astype(np.float32)
        lo = self.ctx * self.up // self.down
        hi = (len(x) - self.ctx) * self.up // self.down
        self.hist = x[-2 * self.ctx:]
        return y[lo:hi]

    def feed(self, chunk):
        """True if any part of this chunk is speech."""
        with self.lock:
            x = np.concatenate([self.tail, self._to16k(chunk)])
            speech, i = False, 0
            while i + self.win <= len(x):
                if self.model.is_speech(x[i:i + self.win]):
                    speech = True
                i += self.win
            self.tail = x[i:]
        return speech


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
        return ("" if is_junk(text) else text), info.language, info

    def transcribe(self, audio16, language, final, expect=None):
        """language: the Spoken language picked. The speech can be that or English: Whisper decides
        which, and has to pick one of the two if it heard something else. On auto-detect (language
        ''), expect lists the languages usually heard: a similar one is kept only if Whisper is
        100 times surer of it than of each of them (Mandarin taken for Cantonese, say)."""
        if language == "en":
            return self._run(audio16, "transcribe", "en", final)[:2]
        text, lang, info = self._run(audio16, "transcribe", "", final)
        allowed = allowed_languages(language)
        expected = allowed or {c for c in (expect or ()) if c in language_family(lang)}
        if expected and lang not in expected:
            odds = dict(getattr(info, "all_language_probs", None) or [])
            best = max(sorted(expected), key=lambda c: odds.get(c, 0.0))
            if allowed or odds.get(lang, 1.0) < 100 * odds.get(best, 0.0):
                text, lang, _ = self._run(audio16, "transcribe", best, final)
        return text, lang

    def translate(self, audio16, language):
        return self._run(audio16, "translate", language, True)[0]


def load_asr(engine, device, status, game_mode=True, words=()):
    gpu = None
    if engine == "qwen3" and device != "cpu":
        try:
            gpu = Qwen3GpuASR(status, words)
        except Exception as e:
            log("Qwen3-ASR on the GPU unavailable, using the CPU version:", traceback.format_exc())
            status(f"Qwen3-ASR couldn't use the GPU ({str(e)[:60]}) - using the CPU")
    if gpu is None:
        ASR_HOST.stop()  # not needed (CPU chosen, other engine, or it failed): free its video memory
    elif game_mode:
        # Game mode: one recognizer, entirely on the GPU. SenseVoice isn't loaded at all, so the
        # processor - what Valorant and most competitive games are short of - is left to the game.
        return gpu
    if engine in ("sensevoice", "qwen3"):
        asr = SenseVoiceASR(status, game_mode, device)
        if gpu is not None:
            asr.refiner = gpu
            asr.name = "SenseVoice + Qwen3-ASR (GPU)"
        elif engine == "qwen3":
            try:
                asr.refiner = Qwen3Refiner(status, game_mode, words)
                asr.name = "Qwen3-ASR + SenseVoice"
            except Exception as e:
                log("Qwen3-ASR unavailable, using SenseVoice only:", traceback.format_exc())
                asr.name = "SenseVoice (Qwen3-ASR failed - see log)"
                status(f"Qwen3-ASR couldn't load ({str(e)[:80]}) - using SenseVoice only")
        return asr
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
    # speech recognition on the GPU: the same Qwen3-ASR, run by llama.cpp instead of on the CPU
    "qwen3-asr-0.6b": {"label": "Qwen3-ASR (GPU)", "repo": "ggml-org/Qwen3-ASR-0.6B-GGUF",
                       "file": "Qwen3-ASR-0.6B-Q8_0.gguf", "mmproj": "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf",
                       "mmproj_label": "audio reader", "size": "1 GB", "ctx": 2048},
}
ASR_MODEL = "qwen3-asr-0.6b"
LLM_LOG = os.path.join(APP_DIR, "ai-engine.log")


def llama_exe(backend):
    return os.path.join(LLAMA_DIR, backend, "llama-server.exe")


def has_nvidia_gpu():
    if sys.platform == "win32":
        return os.path.exists(os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvcuda.dll"))
    import shutil

    return shutil.which("nvidia-smi") is not None


def ensure_llama_builds(status, backends=None):
    """Downloads the AI engine. By default only the build this PC needs: CUDA for NVIDIA graphics cards
    (590 MB), Vulkan for others (40 MB). The other one is fetched later only if that one fails."""
    import zipfile

    if backends is None:
        backends = ["cuda"] if has_nvidia_gpu() else ["vulkan"]
    for backend, _dev, urls in LLAMA_BUILDS:
        if backend not in backends or os.path.exists(llama_exe(backend)):
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
    m = LLM_MODELS[key]
    os.makedirs(MODELS_DIR, exist_ok=True)
    for f in model_files(key):
        path = os.path.join(MODELS_DIR, f)
        if not os.path.exists(path):
            label = m["label"] + (f" ({m.get('mmproj_label', 'image reader')})" if f.startswith("mmproj") else "")
            download(f"https://huggingface.co/{m['repo']}/resolve/main/{f}", path, status, label)


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
                    if not errors:
                        continue  # only needed if the build we have fails
                    try:
                        ensure_llama_builds(status, [backend])
                    except Exception as e:
                        errors.append(f"{backend}: couldn't download it ({e})")
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
        # Subtitle prompts top out around 1,500 tokens (longest topic, 30 custom words, long lines),
        # so 2,048 is enough and leaves ~320 MB more video memory for the game than 4,096 did.
        ctx = str(m.get("ctx") or (8192 if m.get("mmproj") else 2048))
        args = [exe, "-m", os.path.join(MODELS_DIR, m["file"]), "--host", "127.0.0.1", "--port", str(port),
                "-dev", device, "-ngl", "all", "-c", ctx, "-np", "1", "--no-webui",
                "--reasoning-budget", "0"]
        if m.get("mmproj"):
            args += ["--mmproj", os.path.join(MODELS_DIR, m["mmproj"])]
        else:
            args += ["-ub", "256"]  # smaller working buffers on the GPU; prompts are short anyway
        if self.lean:
            # everything runs on the GPU, so the CPU side only needs a couple of threads,
            # and --poll 0 stops them spinning at 100% while waiting
            args += ["-t", "2", "-tb", "2", "--poll", "0"]
        logf = open_log(LLM_LOG, binary=True)
        logf.write(f"\n===== {time.ctime()} {' '.join(args)}\n".encode())
        logf.flush()
        flags = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW
        if sys.platform == "win32" and self.game_mode:
            flags |= BELOW_NORMAL
        self.proc = subprocess.Popen(args, cwd=os.path.dirname(exe), stdout=logf, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, creationflags=flags)
        logf.close()
        _kill_with_us(self.proc)
        if self.game_mode:
            set_gpu_priority(self.proc, True)
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
            set_gpu_priority(self.proc, on)

    def chat(self, key, messages, max_tokens=300, temperature=0.2, timeout=60, status=log, on_text=None,
             extra=None, full=False):
        """on_text(text_so_far) is called as the reply streams in, if given. A last message from the
        assistant is the start of the reply, which the model carries on from. extra: more request
        fields; full=True returns the whole response rather than just the text."""
        with self.lock:
            self.use(key, status)
            body = {
                "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                "chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": True,
                "stream": on_text is not None,
            }
            body.update(extra or {})
            req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if on_text is None:
                    res = json.loads(r.read().decode())
                    return res if full else res["choices"][0]["message"].get("content") or ""
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


HOST = LlamaHost()      # translator (and the screenshot reader, swapped in when needed)
ASR_HOST = LlamaHost()  # speech recognition on the GPU; stays loaded next to the translator


def _unquote(text):
    text = text.strip()
    if len(text) > 1 and text[0] in "\"'“「" and text[-1] in "\"'”」":
        text = text[1:-1].strip()
    return text


def parse_translation(text, original):
    """Returns (fixed_original, translation). Handles 'ENGLISH:/TRANSLATION:/FIXED:' replies and plain replies."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    eng_m = re.search(r"(?:ENGLISH|TRANSLATION):\s*(.*?)\s*(?=\n\s*FIXED:|$)", text, flags=re.S | re.I)
    fix_m = re.search(r"FIXED:\s*(.*?)\s*(?=\n\s*(?:ENGLISH|TRANSLATION):|$)", text, flags=re.S | re.I)
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
    """The translation part of a reply that's still streaming in."""
    text = re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.S).lstrip()
    if "ENGLISH:".startswith(text.upper()) or "TRANSLATION:".startswith(text.upper()):
        return ""  # the label itself is still arriving
    text = re.sub(r"^(?:ENGLISH|TRANSLATION):\s*", "", text, flags=re.I).lstrip("\"'“「")
    text = re.split(r"\n\s*F(?:I(?:X(?:E(?:D(?::)?)?)?)?)?\s*$|\n\s*FIXED:", text, flags=re.I)[0]
    return _unquote(text.strip().split("\n\n")[0]) if text.strip() else ""


def parse_spots(reply):
    """[{'box': [x1, y1, x2, y2] as the model wrote them, 'text': str}] from Qwen3-VL's text-spotting reply.
    Copes with code fences, single quotes and replies cut off part-way."""
    import ast

    text = re.sub(r"<think>.*?</think>", "", reply or "", flags=re.S)
    text = re.sub(r"```(?:json)?", "", text)
    data = None
    m = re.search(r"\[.*\]", text, flags=re.S)
    if m:
        for loader in (json.loads, ast.literal_eval):
            try:
                data = loader(m.group(0))
                break
            except Exception:
                pass
    if not isinstance(data, list):  # damaged or cut-off list: keep every complete entry
        data = []
        for obj in re.finditer(r"\{[^{}]*\}", text):
            for loader in (json.loads, ast.literal_eval):
                try:
                    data.append(loader(obj.group(0)))
                    break
                except Exception:
                    pass
    out = []
    for d in data:
        if not isinstance(d, dict):
            continue
        box = d.get("bbox_2d") or d.get("bbox")
        t = " ".join(str(d.get("text_content") or d.get("text") or "").split())
        if not t or not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in box)
        except (TypeError, ValueError):
            continue
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0:
            out.append({"box": (x1, y1, x2, y2), "text": t})
    return out


def parse_numbered(reply, n):
    """'1. ...\n2. ...' -> list of n strings ('' where an item is missing)."""
    text = re.sub(r"<think>.*?</think>", "", reply or "", flags=re.S)
    found = {}
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)\s*[.)、:：]\s*(.*)", line)
        if m:
            i = int(m.group(1))
            if 1 <= i <= n and i not in found:
                found[i] = _unquote(m.group(2).strip())
    return [found.get(i, "") for i in range(1, n + 1)]


def clearly_english(text):
    """Latin-only text with real English words in it: leave it on the screenshot as it is."""
    if has_foreign_script(text) or re.search(r"[\u00c0-\u024f¿¡]", text):
        return False
    words = [w.replace("'", "") for w in re.findall(r"[a-z']+", text.lower())]
    return any(w in ENGLISH_COMMON and w not in AMBIGUOUS_WORDS for w in words) and looks_english(text)


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

    def _messages(self, text, lang, context, fix, alt=None, target="en"):
        system = translation_prompt(target)
        topic = (self.cfg.get("topic") or "").strip()
        words = parse_words(self.cfg.get("words"))
        if topic:
            system += f"\nWhat the viewer is watching (use it to make sense of unclear words): {topic[:300]}\n"
        if words:
            system += ("Names and terms that come up here (if a word sounds like one of these, it probably is; "
                       f"keep them as written): {', '.join(words)}\n")
        ctx = "\n".join(context) if context else "(none)"
        mixed = " mixed with English" if lang != "en" and re.search(r"[A-Za-z]{2,}", text) else ""
        user = f"Earlier lines:\n{ctx}\n\nNew line ({LANG_NAME.get(lang, lang)}{mixed}):\n{text}"
        if alt:
            user += f"\n\nA second speech recognizer heard the same line as:\n{alt}"
        return [{"role": "system", "content": system + reply_format(target, fix)},
                {"role": "user", "content": user + "\n/no_think"}]

    def translate_text(self, text, lang, context, on_english=None, fix=None, alt=None, target="en"):
        """Returns (fixed_original, translation into `target`). on_english(text) gets it as it streams in."""
        fix = (lang in FIX_LANGS) if fix is None else fix
        last = [""]

        def stream(acc):
            eng = english_so_far(acc)
            if eng and eng != last[0]:
                last[0] = eng
                on_english(eng)

        reply = HOST.chat(self.cfg["translator"], self._messages(text, lang, context, fix, alt, target),
                          max_tokens=250 if fix else 120, temperature=0.3, timeout=60,
                          on_text=stream if on_english else None)
        return parse_translation(reply, text)

    def submit(self, line_id, text, lang, alt=None, fix=None, target="en"):
        """alt: how a second recognizer heard the line. fix: also ask for a corrected original.
        target: the language to caption it in."""
        self.finished.append(line_id)
        with self.plock:
            if self.preview_slot and self.preview_slot[0] == line_id:
                self.preview_slot = None
        self.inbox.put((line_id, text, lang, alt, fix, target))

    def preview(self, utt_id, text, lang, target="en"):
        """Offer a still-being-spoken line for a quick early translation."""
        if not self.ready_model or utt_id in self.finished:
            return
        if len(text) - self.previewed.get(utt_id, 0) < 3:  # wait until it's grown a bit
            return
        with self.plock:
            self.preview_slot = (utt_id, text, lang, target)

    def _do_preview(self):
        with self.plock:
            item, self.preview_slot = self.preview_slot, None
        if not item or not self.ready_model:
            return
        utt_id, text, lang, target = item
        if utt_id in self.finished:
            return
        self.previewed[utt_id] = len(text)
        if len(self.previewed) > 50:
            self.previewed.pop(next(iter(self.previewed)))
        try:
            _, eng = self.translate_text(text, lang, list(self.history), fix=False, target=target)
            if eng and utt_id not in self.finished:
                self.out.put(("live_eng", utt_id, eng))
        except Exception as e:
            log("Preview translation failed:", e)

    def run(self):
        while True:
            try:
                line_id, text, lang, alt, fix, target = self.inbox.get(timeout=0.05)
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
                    text, lang, list(self.history), fix=fix, alt=alt, target=target,
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
    MAX_SEG = 10.0     # seconds: a line longer than this is split at its most recent pause
    MIN_SPEECH = 0.25  # seconds of detected voice a line needs to count at all

    def __init__(self, asr, translator, cfg, out_q, speakers=None, gate=None):
        super().__init__(daemon=True)
        self.asr, self.translator, self.cfg, self.out = asr, translator, cfg, out_q
        self.speakers = speakers
        self.gate = gate    # voice detector (None = fall back to a loudness check)
        self.utt_id = None  # id of the line being spoken right now
        self.refine_q = queue.Queue()
        self.stop_evt = threading.Event()

    def stop(self):
        self.stop_evt.set()

    def run(self):
        cap = None
        try:
            cap = open_capture(self.cfg, on_error=lambda e: self.out.put(("error", e)))
            log(f"Listening to: {cap.name} @ {cap.sr} Hz")
            self.out.put(("listening", cap.name))
            if getattr(self.asr, "refiner", None):
                threading.Thread(target=self._refine_loop, daemon=True).start()
            self.loop(cap.q, cap.sr)
        except Exception as e:
            log(traceback.format_exc())
            self.out.put(("error", str(e)))
        finally:
            if cap:
                cap.close()
            self.out.put(("stopped", self))

    # ---- language setting ----
    def language(self):
        """The language hint for the speech model: the Spoken language picked. On auto-detect,
        SenseVoice gets the Chinese/Japanese/Korean language usually heard, so a line it takes for
        another of those (Mandarin taken for Cantonese) is read again as that one."""
        lang = self.cfg.get("language") or ""
        if not lang and self.asr.kind == "sensevoice":
            lang = next((c for c in self.expected()[0] if c in EAST_ASIAN), "")
        if self.asr.kind == "sensevoice" and lang not in SENSEVOICE_LANGS:
            return ""
        return lang

    def expected(self):
        """(languages, strict): what speech should be in. A picked Spoken language allows only it and
        English (strict). Auto-detect expects the languages this run has been hearing (none at first);
        others get through when the second listen is sure of them."""
        picked = self.cfg.get("language") or ""
        if picked:
            return ([picked, "en"] if picked != "en" else ["en"]), True
        return usual_languages(), False

    def _transcribe(self, audio16, final):
        if self.asr.kind == "qwen3gpu":
            expect, strict = self.expected()
            return self.asr.transcribe(audio16, final=final, expect=expect, strict=strict)
        if self.asr.kind == "whisper":
            return self.asr.transcribe(audio16, self.language(), final, expect=self.expected()[0])
        return self.asr.transcribe(audio16, self.language(), final)

    def _quick_ok(self, text, det):
        """Can a line be shown as it was first heard? Not if it's in a language the speech isn't
        expected in (SenseVoice guesses from the first second or so, and often takes Mandarin for
        Cantonese): the second listen decides. With nothing to check it with, it's shown on
        auto-detect, and dropped when a Spoken language was picked."""
        expect, strict = self.expected()
        if strict:
            return fits_language(text, det, expect[0])
        if not expect or getattr(self.asr, "refiner", None) is None:
            return True
        lang = text_language(text, det)
        return lang not in EAST_ASIAN or lang in expect

    def fast_pass_ok(self):
        """SenseVoice only knows 5 languages; for others (Spanish, French...) only Qwen3-ASR is used."""
        lang = self.cfg.get("language") or ""
        return self.asr.kind != "sensevoice" or not lang or lang in SENSEVOICE_LANGS

    # ---- finished lines ----
    def finish_line(self, audio16):
        line_id, self.utt_id = self.utt_id or next(_line_ids), None
        refiner = getattr(self.asr, "refiner", None)
        if self.fast_pass_ok():
            text, det = self._transcribe(audio16, True)
        else:
            text, det = "", self.cfg.get("language") or "en"
        # A line in a language the speech isn't expected in isn't shown: the second listen decides.
        # (Even when told the language, SenseVoice writes Japanese or Korean when it's sure enough.)
        unfit = bool(text) and not self._quick_ok(text, det)
        if unfit:
            log(f"  not in the languages expected, not shown yet: {text!r}")
            text = ""
        if not text and not refiner:
            self.out.put(("dropped", line_id))
            return
        spk = self.speakers.identify(audio16) if self.speakers and self.cfg.get("speakers", True) else None
        if not refiner:
            self._publish("final", line_id, text, det, spk, audio16)
            return
        if unfit:
            self.refine_q.put((line_id, audio16, "", det, spk, False))
            return
        # Two-pass: show the quick version now if it looks like real speech, then let Qwen3-ASR
        # re-listen in the background and correct (or remove) it.
        lang, to = self._classify(text, det)
        if self.cfg.get("game_mode", True) and self.fast_pass_ok():
            if not text:  # SenseVoice heard no words: game sounds, not speech
                self.out.put(("dropped", line_id))
                return
            if det == "en" and to is None:  # plain English, captions in English: SenseVoice is reliable here
                self._publish("final", line_id, text, det, spk, audio16)
                return
        shown = len(_PUNCT.sub("", text)) >= 3 and not (self.cfg.get("hide_english") and to is None)
        if shown:
            self.out.put(("final", line_id, text, lang, self._wants_translation(to), spk, True))
        self.refine_q.put((line_id, audio16, text, det, spk, shown))

    def _trusted_language(self):
        """A language the user picked that SenseVoice can't recognise itself (Spanish, French...)."""
        lang = self.cfg.get("language") or ""
        return lang if lang and lang not in SENSEVOICE_LANGS else None

    def _classify(self, text, det):
        """(language label, language to caption it in - None if it's fine as it is),
        judged from what was actually said."""
        if self.asr.kind == "whisper":
            return det, caption_target(text, det, self.cfg, trust=True)
        picked = self.cfg.get("language") or ""
        trusted = self._trusted_language()
        trust = bool(trusted)
        lang = text_language(text, det if picked and det in (picked, "en") else (trusted or det))
        if lang == "en" and needs_translation(text, "en"):
            if not picked:
                lang = "other"  # Latin letters but not English (Spanish, French...) with no language picked
            elif picked in LATIN_SCRIPT:
                lang = picked  # not English, so the other language picked
            else:
                trust = True  # of the two languages picked, only English is written in these letters
        return lang, caption_target(text, lang, self.cfg, trust=trust)

    def _wants_translation(self, to):
        if to is None or self.cfg["show"] == "original":
            return False
        # without the AI translator, Whisper can still translate - but only into English
        return bool(self.translator.ready_model) or (self.asr.kind == "whisper" and to == "en")

    def _refine_loop(self):
        while not self.stop_evt.is_set():
            try:
                line_id, audio16, quick, det, spk, shown = self.refine_q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if self.refine_q.qsize() >= 2 and quick:
                    text = quick  # falling behind: skip the slow second listen for this line
                else:
                    t0 = time.monotonic()
                    expect, strict = self.expected()
                    text, heard = self.asr.refiner.refine(audio16, expect, strict)
                    log(f"  second listen ({(time.monotonic() - t0) * 1000:.0f} ms): {text!r}  (quick: {quick!r})")
                    if strict and text and not fits_language(text, heard or det, expect[0]):
                        log("  second listen is in a language that wasn't picked: keeping the quick version")
                        text = quick
                    elif heard:
                        det = heard
                    elif text and det not in expect and text_language(text, det) in ("zh", "yue"):
                        # the CPU second listen can't say which language it heard, and Chinese
                        # characters look the same either way: go with the Chinese language expected
                        det = next((c for c in expect if c in ("zh", "yue")), det)
            except Exception:
                log(traceback.format_exc())
                text = quick
            if not text:  # Qwen3-ASR heard no real speech: it was noise/music
                self.out.put(("retract" if shown else "dropped", line_id))
                continue
            self._publish("refined", line_id, text, det, spk, audio16, alt=quick)

    def _publish(self, kind, line_id, text, det, spk, audio16, alt=None):
        whisper = self.asr.kind == "whisper"
        lang, to = self._classify(text, det)
        if self.cfg.get("hide_english") and to is None:
            self.out.put(("retract", line_id))
            return
        wanted = self._wants_translation(to)
        self.out.put((kind, line_id, text, lang, wanted, spk, False))
        log(f"[{lang}] (speaker {spk}) {text}")
        note_language(text, lang)
        if not wanted:
            return
        if self.translator.ready_model:
            differs = alt and _PUNCT.sub("", alt).lower() != _PUNCT.sub("", text).lower()
            self.translator.submit(line_id, text, lang, alt=alt if differs else None, target=to,
                                   fix=lang in FIX_LANGS and kind != "refined" and self.asr.kind == "sensevoice")
        elif whisper:  # no AI translator: fall back to Whisper's own (English only)
            eng = self.asr.translate(audio16, det)
            if eng:
                self.out.put(("translated", line_id, eng, text))

    # ---- audio loop ----
    @staticmethod
    def _cut_index(chunks, sr):
        """Where to split a line that has run too long while the person is still talking:
        the most recent pause, or failing that the quietest moment of the last few seconds."""
        n = len(chunks)
        for i in range(n - 1, int(n * 0.4), -1):
            if not chunks[i][1]:
                return i + 1
        tail = max(1, int(3.0 * sr / max(1, len(chunks[-1][0]))))
        lo = max(1, n - tail)
        energies = [float(np.mean(ch * ch)) for ch, _ in chunks[lo:]]
        return lo + int(np.argmin(energies)) + 1

    def loop(self, q, sr):
        g = gcd(sr, WHISPER_SR)
        up, down = WHISPER_SR // g, sr // g
        gate = self.gate
        if gate:
            gate.reset(sr)
        preroll = int(sr * 0.3)
        st = {"chunks": [], "voiced": False, "quiet": 0, "end_quiet": 0}  # chunks: [(samples, is_speech)]
        last_run = 0.0

        def to16k(parts):
            return resample_poly(np.concatenate(parts), up, down).astype(np.float32)

        def end_line(ended):
            chunks = st["chunks"]
            if ended:
                seg, rest = chunks, []
            else:
                k = self._cut_index(chunks, sr)
                seg, rest = chunks[:k], chunks[k:]
            voice = sum(len(ch) for ch, sp in seg if sp) / sr
            st["chunks"] = rest
            st["voiced"] = any(sp for _, sp in rest)
            st["quiet"] = 0
            if voice < self.MIN_SPEECH:  # a click or blip, not a line
                if self.utt_id is not None:
                    self.out.put(("dropped", self.utt_id))
                    self.utt_id = None
                return
            parts = [ch for ch, _ in seg]
            if ended:  # drop most of the trailing silence, keep a short tail
                extra = st["end_quiet"] - int(0.25 * sr)
                while extra > 0 and len(parts) > 1 and len(parts[-1]) <= extra:
                    extra -= len(parts.pop())
            self.finish_line(to16k(parts))

        while not self.stop_evt.is_set():
            c = self.cfg
            try:
                chunk = q.get(timeout=0.1)
            except queue.Empty:
                chunk = None
            while chunk is not None:
                if gate:
                    try:
                        speech = gate.feed(chunk)
                    except Exception:
                        log(traceback.format_exc())
                        gate = None  # voice detector broke: use the loudness check from here on
                        speech = False
                if not gate:
                    speech = float(np.sqrt(np.mean(chunk * chunk))) > c["threshold"]
                st["chunks"].append((chunk, speech))
                if speech:
                    st["voiced"], st["quiet"] = True, 0
                elif st["voiced"]:
                    st["quiet"] += len(chunk)
                    if st["quiet"] / sr >= c["silence"]:  # they stopped talking: the line is finished
                        st["end_quiet"] = st["quiet"]
                        end_line(True)
                try:
                    chunk = q.get_nowait()
                except queue.Empty:
                    chunk = None

            chunks = st["chunks"]
            n = sum(len(ch) for ch, _ in chunks)
            if not st["voiced"]:
                while len(chunks) > 1 and n - len(chunks[0][0]) >= preroll:  # keep a short lead-in only
                    n -= len(chunks.pop(0)[0])
                continue

            if n / sr >= max(c["max_seg"], self.MAX_SEG):
                end_line(False)
                continue

            if c.get("game_mode", True):
                # Game mode: no live text. Re-transcribing the growing sentence every half second
                # was most of LiveSubs' CPU use while people talked; each line is now decoded once.
                continue
            now = time.monotonic()
            if now - last_run < self.asr.live_step or not self.fast_pass_ok():
                continue
            last_run = now
            text, det = self._transcribe(to16k([ch for ch, _ in chunks]), False)
            if text and self._quick_ok(text, det):
                lang, to = self._classify(text, det)
                if self.cfg.get("hide_english") and to is None:
                    continue  # don't flash up a line that's going to be hidden anyway
                if self.utt_id is None:
                    self.utt_id = next(_line_ids)
                self.out.put(("partial", text, lang, self.utt_id))
                if not self.cfg.get("game_mode", True) and self.cfg["show"] != "original" and to:
                    self.translator.preview(self.utt_id, text, lang, to)


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
        u.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        u.SetLayeredWindowAttributes.argtypes = [wintypes.HWND, wintypes.DWORD, wintypes.BYTE, wintypes.DWORD]
        _U32 = u
    return _U32


def _hwnd(win):
    win.update_idletasks()
    return _user32().GetParent(win.winfo_id())


SWP_FLAGS = 0x0001 | 0x0002 | 0x0010 | 0x0200  # NOSIZE | NOMOVE | NOACTIVATE | NOOWNERZORDER


def keep_on_top(*wins):
    """Put windows above everything WITHOUT activating them (Tk's -topmost activates).
    With several windows, the last one ends up on top."""
    for win in wins:
        if sys.platform != "win32":
            win.attributes("-topmost", True)
            win.lift()
            continue
        try:
            _user32().SetWindowPos(_hwnd(win), -1, 0, 0, 0, 0, SWP_FLAGS)  # -1 = HWND_TOPMOST
        except Exception as e:
            log("keep_on_top failed:", e)


SW_HIDE, SW_SHOWNOACTIVATE = 0, 4


def show_window(win, visible):
    """Show or hide a window without activating it (Tk's deiconify would pull focus from the game).
    A hidden window costs nothing; one that's merely transparent still makes Windows blend it over the game."""
    if sys.platform != "win32":
        (win.deiconify if visible else win.withdraw)()
        return
    try:
        _user32().ShowWindow(_hwnd(win), SW_SHOWNOACTIVATE if visible else SW_HIDE)
    except Exception as e:
        log("ShowWindow failed:", e)


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


def set_window_alpha(win, alpha):
    """How solid a window is (0-1). On Windows this must follow game_safe(): a window made layered isn't
    drawn reliably until its layered attributes are set."""
    if sys.platform != "win32":
        win.attributes("-alpha", alpha)
        return
    try:
        _user32().SetLayeredWindowAttributes(_hwnd(win), 0, max(0, min(255, round(alpha * 255))), 0x2)  # LWA_ALPHA
    except Exception as e:
        log("Window transparency failed:", e)


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
        # Two layers, so the Background slider fades only the box and never the text:
        #   back - the dark box; the slider sets how see-through it is
        #   win  - the text, always fully solid; its own background colour is made see-through
        self.back = b = tk.Toplevel(root)
        b.overrideredirect(True)
        b.configure(bg=BG)
        self.win = w = tk.Toplevel(root)
        w.overrideredirect(True)
        self.text_bg = BG
        if sys.platform == "win32":
            try:
                w.attributes("-transparentcolor", TEXT_KEY)
                self.text_bg = TEXT_KEY
            except Exception as e:
                log("See-through text layer unavailable:", e)
        w.configure(bg=self.text_bg)
        self.body = tk.Frame(w, bg=self.text_bg)
        self.body.pack(fill="both", padx=12, pady=(4, 6))
        self.shown = None  # are the layers on screen right now

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
        self._bind(b)  # clicks on the box (between the letters) land on the back layer

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
        self._update_layers()
        self._redraw()

    def _font(self, lang, big, scale=1.0):
        size = max(10, int(self.cfg["font"] * scale))
        family = SCRIPT_FONTS.get(lang, "Segoe UI")
        return (family, size, "bold") if big else (family, max(9, int(size * 0.72)))

    def _update_layers(self):
        """Box see-through as set by the slider, text always solid. Both are taken off screen
        completely while nobody's talking or when hidden, so nothing sits over the game."""
        opacity = max(0.0, min(1.0, float(self.cfg["opacity"])))
        visible = not (self.faded or self.user_hidden)
        locked = self.cfg["click_through"]
        game_safe(self.back, click_through=locked)
        game_safe(self.win, click_through=locked)  # its see-through text background is set by Tk
        set_window_alpha(self.back, opacity)
        show_window(self.back, visible and opacity > 0.02)
        show_window(self.win, visible)
        if visible and not self.shown:
            keep_on_top(self.back, self.win)
        self.shown = visible

    def hide_now(self):
        """Take both layers off screen right away (e.g. so they're not in a screenshot)."""
        show_window(self.back, False)
        show_window(self.win, False)
        self.shown = False

    def layout(self):
        """Box hugs the text (up to the max width) and stays anchored at its bottom-centre point."""
        self.win.update_idletasks()
        h = self.win.winfo_reqheight()
        w = min(self.w, max(120, self.win.winfo_reqwidth()))
        geometry = f"{w}x{h}+{self.cx - w // 2}+{self.bottom - h}"
        self.win.geometry(geometry)
        self.back.geometry(geometry)

    def set_visible(self, visible):
        self.user_hidden = not visible
        self.apply()

    # ---- content ----
    def _label(self, text, font, color, top_pad, marker=None):
        """marker: (speaker number, colour) shown as a coloured tag before the text."""
        if marker is None:
            lbl = self.tk.Label(self.body, text=text, font=font, fg=color, bg=self.text_bg, justify="center",
                                wraplength=self.w - 24)
            lbl.pack(fill="x", pady=(top_pad, 0))
            self._bind(lbl)
            return
        row = self.tk.Frame(self.body, bg=self.text_bg)
        row.pack(pady=(top_pad, 0))
        num, mcol = marker
        tag = self.tk.Label(row, text=f"▍{num + 1}", font=("Segoe UI", max(9, int(font[1] * 0.75)), "bold"),
                            fg=mcol, bg=self.text_bg)
        tag.pack(side="left", anchor="n", padx=(0, 6))
        lbl = self.tk.Label(row, text=text, font=font, fg=color, bg=self.text_bg, justify="left",
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

        to = self._script_of(eng)
        if show == "original" or (not needs and not eng):
            self._label(orig, self._font(lang, True, scale), main, gap, marker)
        elif show == "english":
            if eng:
                self._label(eng, self._font(to, True, scale), eng_col, gap, marker)
            else:  # no translation yet: show what was said, small
                self._label(orig, self._font(lang, False, scale), sub, gap, marker)
        else:
            self._label(orig, self._font(lang, False, scale), sub, gap, marker)
            self._label(eng or "…", self._font(to, True, scale), eng_col if eng else "#6a6a6a", 1)

    def _script_of(self, text):
        """Font language for a translation: English, or e.g. Chinese when captioning into Chinese."""
        if not text:
            return "en"
        to = text_language(text, "en")
        if to == "zh" and self.cfg.get("target") == "zh-Hant":
            to = "yue"  # Traditional characters: Microsoft JhengHei
        return to

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
        self.lines.sort(key=lambda ln: ln["id"])  # a line confirmed late still goes in its spoken place
        del self.lines[:-8]
        self._redraw()
        self._activity()

    def refined(self, line_id, orig, lang, needs_english, spk=None):
        """The second, more careful listen came back: correct the line (or add it if it wasn't shown yet)."""
        ln = self._find(line_id)
        if not ln:
            return self.final(line_id, orig, lang, needs_english, spk)
        ln.update(orig=orig, lang=lang, needs=needs_english)
        if not needs_english:
            ln["eng"] = None
        self._redraw()
        self._activity()

    def retract(self, line_id):
        """It turned out not to be speech (or the line is hidden): take it off the screen."""
        if self.live and self.live.get("id") == line_id:
            self.live = None
        self.lines = [ln for ln in self.lines if ln["id"] != line_id]
        self._redraw()

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
            self._update_layers()

    def tick(self, listening):
        now = time.monotonic()
        hold = self.cfg["hold"]
        if listening and not self.faded and hold > 0 and now - self.last_activity > hold:
            self.faded = True
            self._update_layers()  # nobody's talking: take the captions off screen
        if self.shown and now - self._last_top > 2:
            self._last_top = now
            keep_on_top(self.back, self.win)  # stay above games/videos without stealing focus

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


_HOTKEY_THREAD = [0]  # id of the thread that owns the hotkeys (Windows)
WM_ESC_HOTKEY = 0x8001  # WM_APP + 1: turn the Esc hotkey on (wParam 1) or off (0)
ESC_ID = 5


def start_hotkey(callback):
    """Global hotkeys that work inside games. Calls callback("region"), callback("full") etc. from a thread."""
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
        _HOTKEY_THREAD[0] = ctypes.windll.kernel32.GetCurrentThreadId()
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == 0x0312:  # WM_HOTKEY
                if msg.wParam == ESC_ID:
                    callback("escape")
                elif msg.wParam in keys:
                    callback(keys[msg.wParam][0])
            elif msg.message == WM_ESC_HOTKEY:
                if msg.wParam:
                    user32.RegisterHotKey(None, ESC_ID, 0x4000, 0x1B)  # VK_ESCAPE, no modifiers
                else:
                    user32.UnregisterHotKey(None, ESC_ID)

    threading.Thread(target=run, daemon=True).start()


def set_escape_hotkey(on):
    """Esc closes a translated screenshot while one is showing; the rest of the time the game keeps Esc."""
    if sys.platform != "win32" or not _HOTKEY_THREAD[0]:
        return
    try:
        import ctypes

        ctypes.windll.user32.PostThreadMessageW(_HOTKEY_THREAD[0], WM_ESC_HOTKEY, 1 if on else 0, 0)
    except Exception as e:
        log("Esc hotkey unavailable:", e)


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


class Cancelled(Exception):
    pass


class ScreenJob:
    """One screenshot being translated. origin = where the image sits on screen; full = whole screen."""

    def __init__(self, image, origin, full):
        self.image, self.origin, self.full = image, origin, full
        self.cancelled = False

    def check(self, *_):
        if self.cancelled:
            raise Cancelled()


VISION_MIN_PIXELS = 1024 * 32 * 32  # Qwen3-VL places text boxes reliably from ~1,000 image tokens up
VISION_MAX_PIXELS = 1600 * 32 * 32  # ~1700x960: small game text stays readable, video memory stays modest


def vision_image(image):
    """The screenshot scaled into the size range Qwen3-VL reads best. Returns (data URI, (width, height))."""
    import base64
    import io

    from PIL import Image

    w, h = image.size
    area = w * h
    scale = 1.0
    if area > VISION_MAX_PIXELS:
        scale = (VISION_MAX_PIXELS / area) ** 0.5
    elif area < VISION_MIN_PIXELS:
        scale = min(8.0, (VISION_MIN_PIXELS / area) ** 0.5)  # small boxes: enlarge the text
    if scale != 1.0:
        image = image.resize((max(32, round(w * scale)), max(32, round(h * scale))),
                             Image.LANCZOS if scale < 1 else Image.BICUBIC)
    png = io.BytesIO()
    image.convert("RGB").save(png, "PNG", compress_level=1)
    return "data:image/png;base64," + base64.b64encode(png.getvalue()).decode(), image.size


def spot_text(job, status=log):
    """Every line of text in the screenshot: [{'box': (x1, y1, x2, y2) in screenshot pixels, 'text': str}]."""
    if not model_ready(VISION_MODEL):  # only needed when the text reader isn't available
        ensure_model(VISION_MODEL, status)
        job.check()
    W, H = job.image.size
    uri, (sent_w, sent_h) = vision_image(job.image)
    found = [0]

    def progress(so_far):
        job.check()
        n = so_far.count("bbox_2d")
        if n != found[0]:
            found[0] = n
            status(f"Reading text... {n} found")

    status("Reading text...")
    reply = HOST.chat(VISION_MODEL, [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": uri}},
        {"type": "text", "text": SPOT_PROMPT},
    ]}], max_tokens=4000, temperature=0.1, timeout=240, status=status, on_text=progress)
    job.check()
    raw = parse_spots(reply)
    # Qwen3-VL gives positions on a 0-999 scale; fall back to pixels of the image it was sent if it didn't
    relative = all(max(s["box"]) <= 1000 for s in raw)
    sx, sy = (999, 999) if relative else (sent_w, sent_h)
    spots, seen = [], set()
    for s in raw:
        x1, y1, x2, y2 = s["box"]
        box = tuple(round(min(1.0, max(0.0, v / d)) * size)
                    for v, d, size in ((x1, sx, W), (y1, sy, H), (x2, sx, W), (y2, sy, H)))
        key = (s["text"], tuple(v // 8 for v in box))
        if box[2] - box[0] < 3 or box[3] - box[1] < 3 or key in seen:  # empty or repeated entries
            continue
        seen.add(key)
        spots.append({"box": box, "text": s["text"]})
    return spots


_NO_SPACES = re.compile(r"[　-〿぀-ヿ㐀-鿿豈-﫿＀-￯]")  # Chinese/Japanese
_SENTENCE_END = tuple("。！？!?…:：")


def _runs_on(prev, line):
    """Is `line` the next line of the same sentence as `prev` (a wrapped paragraph or two-line subtitle)?
    Kept strict so menus, lists, chat messages and bilingual subtitles stay separate."""
    ax1, ay1, ax2, ay2 = prev["box"]
    bx1, by1, bx2, by2 = line["box"]
    ha, hb = ay2 - ay1, by2 - by1
    h = (ha + hb) / 2
    a, b = prev["text"].rstrip(), line["text"].strip()
    return (max(ha, hb) <= 1.35 * min(ha, hb)                    # same text size
            and -0.25 * h <= by1 - ay2 <= 0.6 * h                 # directly underneath
            and (abs(bx1 - ax1) <= 1.2 * h or abs((bx1 + bx2) - (ax1 + ax2)) <= 3 * h)  # left- or centre-aligned
            and ax2 - ax1 >= 6 * h                                # a full line, not a short label
            and not a.endswith(_SENTENCE_END)
            and not re.match(r"\s*(\[[^\]]{1,24}\]|\S{1,24}[:：])", b)  # "[Team] name:" starts a new chat message
            and has_foreign_script(a) == has_foreign_script(b)
            and looks_english(a) == looks_english(b))  # never a foreign line + its English subtitle


def group_lines(spots):
    """Lines -> blocks to translate as one: [{'lines': [...], 'text': str, 'box': (x1, y1, x2, y2)}]."""
    blocks = []
    for s in spots:
        b = blocks[-1] if blocks else None
        if b and _runs_on(b["lines"][-1], s):
            glue = "" if _NO_SPACES.match(b["text"][-1]) and _NO_SPACES.match(s["text"][0]) else " "
            b["lines"].append(s)
            b["text"] += glue + s["text"]
            b["box"] = tuple(f(p, q) for f, p, q in zip((min, min, max, max), b["box"], s["box"]))
        else:
            blocks.append({"lines": [s], "text": s["text"], "box": s["box"]})
    return blocks


def screen_needs_translation(text, target="en"):
    if not re.search(r"[^\W\d_]", text):  # no letters: numbers, times, scores, symbols
        return False
    if target == "en":
        return not clearly_english(text)
    return not in_language(text, "", target)


def _same_text(a, b):
    def norm(s):
        return re.sub(r"[\W_]+", "", s.lower())

    return norm(a) == norm(b)


# ---- fast text reader: PaddleOCR models (via RapidOCR) on the CPU, in a helper process ----
OCR_LOG = os.path.join(APP_DIR, "text-reader.log")
_HANGUL_RE = re.compile(r"[\uac00-\ud7af]")


def _ink_width(text):
    """Roughly how many character-heights wide a line of text is."""
    w = 0.0
    for ch in text:
        if re.match(r"[\u1100-\u11ff\u2e80-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]", ch):
            w += 1.0
        elif ch.isspace() or ch in ".,:;!?'\"|[](){}-":
            w += 0.3
        else:
            w += 0.55
    return w


def _ocr_lines(engine, korean, img):
    """[[x1, y1, x2, y2, text, score], ...] for every line of text in an RGB array."""
    res = engine(img)
    out = []
    boxes = res.boxes if res.boxes is not None else []
    for box, text, score in zip(boxes, res.txts or (), res.scores or ()):
        xs, ys = [float(p[0]) for p in box], [float(p[1]) for p in box]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        text, score = " ".join(str(text).split()), float(score)
        # letters the main model doesn't know (Hangul) come back missing: far too little text for the box
        short = _ink_width(text) * max(1.0, y2 - y1) < 0.5 * (x2 - x1)
        if korean is not None and (score < 0.6 or short):
            crop = img[max(0, int(y1) - 2):int(y2) + 2, max(0, int(x1) - 2):int(x2) + 2]
            try:
                r = korean(crop, use_det=False, use_cls=False, use_rec=True)
            except Exception as e:
                print("Korean text reader failed:", e, file=sys.stderr, flush=True)
                r, korean = None, None
            if r is not None and r.txts:
                ktext, kscore = " ".join(str(r.txts[0]).split()), float(r.scores[0])
                if _HANGUL_RE.search(ktext) and kscore >= 0.5 and (kscore > score or short):
                    text, score = ktext, kscore
        if text and score >= 0.5:
            out.append([round(x1), round(y1), round(x2), round(y2), text, round(score, 3)])
    return out


def ocr_server():
    """--ocr-server: reads text off screenshots for the app (one JSON request per line on stdin). It's a
    separate process so its ONNX runtime never meets the copy the speech models use."""
    replies = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr  # whatever the libraries print goes to the log, not into the replies

    def send(obj):
        replies.write(json.dumps(obj, ensure_ascii=True) + "\n")
        replies.flush()

    try:
        from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

        base = {"Global.log_level": "error", "Global.text_score": 0.0, "Global.max_side_len": 4096,
                "Det.limit_type": "max", "Det.limit_side_len": 2560,  # only ever shrink (4K screens)
                "EngineConfig.onnxruntime.intra_op_num_threads": max(1, min(4, (os.cpu_count() or 4) // 2)),
                "EngineConfig.onnxruntime.inter_op_num_threads": 1}
        engine = RapidOCR(params=base)  # Chinese, English, Japanese and Latin-script languages
    except Exception as e:
        send({"ready": False, "error": f"{type(e).__name__}: {e}"})
        return
    korean = {}

    def load_korean():  # Hangul needs its own small model, downloaded once by RapidOCR
        try:
            reader = RapidOCR(params=dict(base, **{
                "Rec.lang_type": LangRec.KOREAN, "Rec.ocr_version": OCRVersion.PPOCRV5,
                "Rec.model_type": ModelType.MOBILE}))
            reader(np.full((48, 160, 3), 255, np.uint8), use_det=False, use_cls=False, use_rec=True)  # loads it
            korean["engine"] = reader
        except Exception as e:
            korean["error"] = f"{type(e).__name__}: {e}"
            print("Korean text reader unavailable:", korean["error"], file=sys.stderr, flush=True)

    loader = threading.Thread(target=load_korean, daemon=True)
    loader.start()
    send({"ready": True})
    for line in sys.stdin:
        try:
            req = json.loads(line)
        except ValueError:
            continue
        if req.get("cmd") == "quit":
            break
        if req.get("cmd") == "prepare":
            loader.join(timeout=600)
            send({"id": req.get("id"), "korean": "engine" in korean, "korean_error": korean.get("error")})
            continue
        try:
            t0 = time.monotonic()
            lines = _ocr_lines(engine, korean.get("engine"), np.load(req["path"]))
            send({"id": req.get("id"), "lines": lines, "ms": int((time.monotonic() - t0) * 1000)})
        except Exception as e:
            send({"id": req.get("id"), "error": f"{type(e).__name__}: {e}"})


class TextReader:
    """Reads text off screenshots in a helper process, on the CPU - about half a second, and nothing has to
    be swapped into video memory: the translating is then done by the model already loaded for captions."""

    def __init__(self):
        self.proc = None
        self.replies = queue.Queue()
        self.lock = threading.Lock()
        self.n = 0

    @staticmethod
    def installed():
        import importlib.util

        try:
            return all(importlib.util.find_spec(m) for m in ("rapidocr", "onnxruntime"))
        except Exception:
            return False

    def _pump(self, pipe, out):
        for raw in iter(pipe.readline, b""):
            out.put(raw)
        out.put(None)

    def _get(self, timeout):
        while True:
            raw = self.replies.get(timeout=timeout)
            if raw is None:
                self.proc = None
                raise RuntimeError("the text reader stopped (details in text-reader.log)")
            try:
                return json.loads(raw)
            except ValueError:
                continue  # not one of its replies

    def _start(self):
        if self.proc is not None and self.proc.poll() is None:
            return
        exe = sys.executable
        console = os.path.join(os.path.dirname(exe), "python.exe")
        if os.path.basename(exe).lower() == "pythonw.exe" and os.path.exists(console):
            exe = console  # pipes are dependable with the console Python (no window is shown)
        err = open_log(OCR_LOG)
        self.replies = queue.Queue()
        self.proc = subprocess.Popen([exe, os.path.abspath(__file__), "--ocr-server"], cwd=APP_DIR,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                                     creationflags=0x08000000 if sys.platform == "win32" else 0)
        err.close()
        _kill_with_us(self.proc)
        threading.Thread(target=self._pump, args=(self.proc.stdout, self.replies), daemon=True).start()
        hello = self._get(timeout=180)
        if not hello.get("ready"):
            self.stop()
            raise RuntimeError(hello.get("error") or "the text reader didn't start")

    def start(self):
        with self.lock:
            self._start()

    def request(self, payload, timeout=60):
        with self.lock:
            self._start()
            self.n += 1
            payload = dict(payload, id=self.n)
            self.proc.stdin.write((json.dumps(payload) + "\n").encode())
            self.proc.stdin.flush()
            while True:
                msg = self._get(timeout)
                if msg.get("id") == payload["id"]:
                    break
        if msg.get("error"):
            raise RuntimeError(msg["error"])
        return msg

    def read(self, image):
        """[{'box': (x1, y1, x2, y2), 'text': str}] for every line of text in a PIL image."""
        import tempfile

        path = os.path.join(tempfile.gettempdir(), f"livesubs-screenshot-{os.getpid()}.npy")
        np.save(path, np.asarray(image.convert("RGB")))
        try:
            msg = self.request({"path": path})
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        log(f"Text reader: {len(msg['lines'])} lines in {msg.get('ms')} ms")
        return [{"box": tuple(ln[:4]), "text": ln[4]} for ln in msg["lines"]]

    def stop(self):
        proc, self.proc = self.proc, None
        if proc is not None and proc.poll() is None:
            try:
                proc.stdin.write(b'{"cmd": "quit"}\n')
                proc.stdin.flush()
                proc.wait(timeout=3)
            except Exception:
                proc.kill()


TEXT_READER = TextReader()


def loaded_model(cfg):
    """The AI model already on the GPU (so screenshot text is translated without swapping anything in)."""
    if HOST.running():
        return HOST.key
    if cfg["translator"] != "off":
        return cfg["translator"]
    # captions aren't being translated: use any model that's already downloaded, else the small one
    return next((k for k in ("qwen3-8b", "qwen3-14b", VISION_MODEL) if model_ready(k)), "qwen3-8b")


def translate_screen_text(texts, cfg, job, status=log, model=VISION_MODEL):
    prompt = screen_prompt(cfg.get("target") or "en")
    topic, words = (cfg.get("topic") or "").strip(), parse_words(cfg.get("words"))
    if topic:
        prompt += f"\nWhat's on screen: {topic}"
    if words:
        prompt += "\nNames and terms that may appear: " + ", ".join(words)
    items = "\n".join(f"{i}. {' '.join(t.split())}" for i, t in enumerate(texts, 1))
    status(f"Translating {len(texts)} line{'' if len(texts) == 1 else 's'}...")
    reply = HOST.chat(model, [{"role": "system", "content": prompt}, {"role": "user", "content": items}],
                      max_tokens=min(3000, 100 + 80 * len(texts)), temperature=0.2, timeout=180,
                      status=status, on_text=job.check)
    job.check()
    return parse_numbered(reply, len(texts))


def read_screenshot(job, cfg, status=log):
    """Finds the text in job.image and translates whatever isn't in the caption language.
    Returns the text blocks; the ones that got an 'eng' (translation) are painted over.
    Fast way: the text reader finds the text and the model already loaded for captions translates it.
    Fallback: Qwen3-VL reads the text (swapped into video memory for it)."""
    spots, model = None, VISION_MODEL
    if TextReader.installed():
        try:
            status("Reading text...")
            spots, model = TEXT_READER.read(job.image), loaded_model(cfg)
        except Exception as e:
            log("Fast text reader failed, using Qwen3-VL instead:", e)
    job.check()
    if spots is None:
        spots = spot_text(job, status)
    blocks = group_lines(spots)
    target = cfg.get("target") or "en"
    todo = [b for b in blocks if screen_needs_translation(b["text"], target)]
    if todo and not model_ready(model):
        ensure_model(model, status)  # e.g. the translator was switched off and nothing is downloaded
        job.check()
    if todo:
        for b, eng in zip(todo, translate_screen_text([b["text"] for b in todo], cfg, job, status, model)):
            if eng and not _same_text(eng, b["text"]):
                b["eng"] = eng
    return blocks


# ---- painting the English onto the screenshot ----
_FONTS = {}


def paint_font(size, text="", prefer=""):
    """Font for the translation painted onto screenshots. prefer="zh-Hant": Traditional Chinese glyphs."""
    if prefer == "zh-Hant" and _HAN.search(text) and not re.search(r"[가-힯぀-ヿ]", text):
        kind, files = "hant", ["msjh.ttc", "msyh.ttc", "NotoSansCJK-Medium.ttc", "NotoSansCJK-Regular.ttc"]
    elif re.search(r"[가-힯]", text):
        kind, files = "ko", ["malgun.ttf", "NotoSansCJK-Medium.ttc", "NotoSansCJK-Regular.ttc"]
    elif re.search(r"[぀-ヿ]", text):
        kind, files = "ja", ["YuGothM.ttc", "meiryo.ttc", "NotoSansCJK-Medium.ttc", "NotoSansCJK-Regular.ttc"]
    elif has_foreign_script(text):
        kind, files = "cjk", ["msyh.ttc", "NotoSansCJK-Medium.ttc", "NotoSansCJK-Regular.ttc"]
    else:
        kind, files = "latin", []
    files = files + ["seguisb.ttf", "segoeui.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"]
    key = (kind, size)
    if key not in _FONTS:
        from PIL import ImageFont

        font = None
        for name in files:
            try:
                font = ImageFont.truetype(name, size)
                break
            except OSError:
                continue
        if font is None:
            try:
                font = ImageFont.load_default(size)
            except TypeError:  # Pillow older than 10.1
                font = ImageFont.load_default()
        _FONTS[key] = font
    return _FONTS[key]


def _luminance(c):
    def lin(v):
        v = float(v) / 255
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(v) for v in c[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a, b):
    lo, hi = sorted((_luminance(a), _luminance(b)))
    return (hi + 0.05) / (lo + 0.05)


def _box_colours(px, box):
    """(background at the top, background at the bottom, text colour) of a box in the screenshot."""
    import numpy as np

    H, W = px.shape[:2]
    x1, y1, x2, y2 = box
    m = 3
    X1, Y1, X2, Y2 = max(0, x1 - m), max(0, y1 - m), min(W, x2 + m), min(H, y2 + m)
    top = px[Y1:max(Y1 + 1, y1), X1:X2].reshape(-1, 3)
    bottom = px[min(y2, Y2 - 1):Y2, X1:X2].reshape(-1, 3)
    sides = np.concatenate([px[Y1:Y2, X1:max(X1 + 1, x1)].reshape(-1, 3),
                            px[Y1:Y2, min(x2, X2 - 1):X2].reshape(-1, 3)])
    bg = np.median(np.concatenate([top, bottom, sides]), axis=0)
    bg_top = np.median(top, axis=0) if len(top) else bg
    bg_bottom = np.median(bottom, axis=0) if len(bottom) else bg
    if np.abs(bg_top - bg_bottom).sum() < 24:  # flat background
        bg_top = bg_bottom = bg
    inner = px[y1:y2, x1:x2].reshape(-1, 3).astype(float)
    fg = None
    if len(inner):
        dist = np.abs(inner - bg).sum(axis=1)
        if dist.max() > 60:  # the strokes of the letters: the pixels furthest from the background colour
            fg = np.median(inner[dist >= 0.6 * dist.max()], axis=0)
    if fg is None or _contrast(fg, bg) < 3.0:  # keep it readable
        fg = (255, 255, 255) if _contrast((255, 255, 255), bg) >= _contrast((16, 16, 16), bg) else (16, 16, 16)
    return tuple(int(v) for v in bg_top), tuple(int(v) for v in bg_bottom), tuple(int(v) for v in fg)


_CJK_CHAR = "\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff"
_WRAP_PIECE = re.compile(f"[{_CJK_CHAR}][，。、！？；：」』）】》〉…,.!?;:)\\]]*|[^{_CJK_CHAR}]+")


def _wrap_units(text):
    """(piece, space before it). Chinese/Japanese can break between any two characters, other text only
    at spaces; closing punctuation stays with the character before it."""
    return [(piece, i > 0 and j == 0)
            for i, word in enumerate(text.split()) for j, piece in enumerate(_WRAP_PIECE.findall(word))]


def _wrap(draw, text, font, width, force=False):
    """Greedy word wrap; force=True also splits words too long for a line on their own."""
    lines, cur = [], ""
    for word, space in _wrap_units(text):
        trial = (cur + (" " if space else "") + word) if cur else word
        if not cur or draw.textlength(trial, font=font) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = word
        while force and len(cur) > 1 and draw.textlength(cur, font=font) > width:
            k = len(cur) - 1
            while k > 1 and draw.textlength(cur[:k], font=font) > width:
                k -= 1
            lines.append(cur[:k])
            cur = cur[k:]
    if cur:
        lines.append(cur)
    return lines or [""]


def _fit(draw, text, box, line_h, n_lines, centred, W, H, prefer=""):
    """The biggest readable font for the English in the box. Text may spill sideways a little (to the right,
    or both ways for centred text) rather than become tiny; if nothing fits, the box grows downward.
    Returns (font, size, lines, line height, painted box)."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    s_hi = int(max(10, min(72, line_h * 0.8)))
    s_lo = int(max(10, min(s_hi, line_h * 0.5)))
    best = None
    for grow in ((1.0, 1.35, 1.7) if n_lines == 1 else (1.0, 1.15)):
        width = min(W - 4, bw * grow)
        for size in range(s_hi, s_lo - 1, -1):
            font = paint_font(size, text, prefer)
            lines = _wrap(draw, text, font, width)
            if (max(draw.textlength(ln, font=font) for ln in lines) <= width
                    and len(lines) * size * 1.2 <= bh * 1.08 + 2):
                # bigger text is better; spilling sideways or onto extra lines is worse
                score = size / line_h - (grow - 1) * 0.35 - max(0, len(lines) - n_lines) * 0.25
                if best is None or score > best[0]:
                    best = (score, size, lines)
                break
    if best:
        _, size, lines = best
    else:
        size = s_lo
        lines = _wrap(draw, text, paint_font(size, text, prefer), min(W - 4, bw * (1.7 if n_lines == 1 else 1.0)),
                      force=True)
    font, lh = paint_font(size, text, prefer), size * 1.2
    need_w = max(draw.textlength(ln, font=font) for ln in lines) + 2 * max(2, size * 0.25)
    if need_w > bw:
        left = (x1 + x2) / 2 - need_w / 2 if centred else x1
        left = max(0, min(W - need_w, left))
        x1, x2 = int(left), int(min(W, left + need_w + 1))
    need_h = len(lines) * lh + size * 0.3
    if need_h > bh:  # lengthen downward (or upward at the bottom edge)
        y2 = int(min(H, y1 + need_h + 1))
        y1 = int(max(0, y2 - need_h - 1))
    return font, size, lines, lh, (x1, y1, x2, y2)


def paint_translation(image, blocks, target="en"):
    """The screenshot with each translated block drawn over its original text, like Google Lens."""
    import numpy as np
    from PIL import Image, ImageDraw

    base = image.convert("RGB")
    px = np.asarray(base).copy()  # colours always come from the untouched screenshot
    out = base.copy()
    W, H = out.size
    draw = ImageDraw.Draw(out)
    todo = []
    for b in blocks:
        if not b.get("eng"):
            continue
        x1, y1, x2, y2 = b["box"]
        line_h = sum(ln["box"][3] - ln["box"][1] for ln in b["lines"]) / len(b["lines"])
        pad = max(2, round(line_h * 0.12))
        box = (max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad))
        top, bottom, fg = _box_colours(px, b["box"])
        centred = abs((x1 + x2) / 2 - W / 2) < W * 0.1  # subtitles, titles, dialog buttons
        font, size, lines, lh, rect = _fit(draw, b["eng"], box, line_h, len(b["lines"]), centred, W, H, target)
        todo.append((rect, top, bottom, fg, font, size, lines, lh, centred))
    # every background first, then all the text, so a box that grew never hides a neighbour's English
    for (x1, y1, x2, y2), top, bottom, *_ in todo:
        w, h = x2 - x1, y2 - y1
        if w < 1 or h < 1:
            continue
        t = np.linspace(0.0, 1.0, h)[:, None]
        rows = np.array(top, float)[None, :] * (1 - t) + np.array(bottom, float)[None, :] * t
        patch = Image.fromarray(np.repeat(rows[:, None, :], w, axis=1).round().astype(np.uint8), "RGB")
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), radius=max(1, min(6, h // 4)), fill=255)
        out.paste(patch, (x1, y1), mask)
    for (x1, y1, x2, y2), _top, _bottom, fg, font, size, lines, lh, centred in todo:
        y = y1 + ((y2 - y1) - len(lines) * lh) / 2 + lh / 2
        for ln in lines:
            if centred:
                x, anchor = (x1 + x2) / 2, "mm"
            else:
                x, anchor = x1 + max(2, size * 0.25), "lm"
            try:
                draw.text((x, y), ln, font=font, fill=fg, anchor=anchor)
            except ValueError:  # bitmap fallback font without anchor support
                draw.text((x1 + 2, y - size / 2), ln, font=font, fill=fg)
            y += lh
    return out


ACCENT = "#4aa3ff"
PANEL = "#161a20"


class LensView:
    """The screenshot shown back exactly where it was taken, with the English painted over the original
    text. Click it, press Esc or press the hotkey again to close it; right-click copies the English."""
    BORDER = 2
    STRIP = 24

    def __init__(self, root, on_close):
        import tkinter as tk

        self.root, self.on_close = root, on_close
        self.win = w = tk.Toplevel(root)
        w.overrideredirect(True)
        w.configure(bg=ACCENT)
        self.c = tk.Canvas(w, highlightthickness=0, bd=0, bg=PANEL, cursor="hand2")
        self.c.pack()
        self.c.bind("<ButtonRelease-1>", lambda e: self.close())
        self.c.bind("<Button-3>", lambda e: self.copy())
        self.is_open = False
        self.photo = None
        self.english = ""
        self.full = False
        self.hint = self.pill = None
        # created once and hidden, so showing it later never takes focus from the game
        w.geometry("1x1+-200+-200")
        game_safe(w)
        set_window_alpha(w, 1.0)
        show_window(w, False)

    def show(self, image, origin, full, hint, english=""):
        from PIL import ImageTk

        c = self.c
        c.delete("all")
        self.photo = ImageTk.PhotoImage(image)
        self.english, self.full = english, full
        iw, ih = image.size
        x, y = origin
        if full:
            b, cw, ch, img_y, gy = 0, iw, ih, 0, y
            self.strip_y = None
        else:
            b = self.BORDER
            cw, ch = max(iw, 290), ih + self.STRIP  # wide enough for the hint underneath
            above = y + ih + b + self.STRIP > self.root.winfo_screenheight()  # no room underneath
            img_y = self.STRIP if above else 0
            self.strip_y = 0 if above else ih
            gy = y - b - (self.STRIP if above else 0)
        c.config(width=cw, height=ch)
        c.create_image(0, img_y, image=self.photo, anchor="nw")
        c.pack_configure(padx=b, pady=b)
        self.pill = self.hint = None
        self.set_hint(hint)
        self.win.geometry(f"{cw + 2 * b}x{ch + 2 * b}+{x - b}+{gy}")
        if not self.is_open:
            show_window(self.win, True)
            keep_on_top(self.win)
            self.is_open = True

    def set_hint(self, text):
        c = self.c
        if self.hint is None:
            if self.full:  # a small label over the top of the screenshot
                self.pill = c.create_rectangle(0, 0, 0, 0, fill=PANEL, outline=ACCENT)
                self.hint = c.create_text(int(c["width"]) // 2, 22, fill="white", font=("Segoe UI", 11, "bold"))
            else:  # a strip under (or over) the picked box
                sy = self.strip_y
                c.create_rectangle(0, sy, int(c["width"]), sy + self.STRIP, fill=PANEL, outline="")
                self.hint = c.create_text(7, sy + self.STRIP // 2, anchor="w", fill="#cfd6e0",
                                          font=("Segoe UI", 9))
        c.itemconfigure(self.hint, text=text)
        if self.pill is not None:
            x1, y1, x2, y2 = c.bbox(self.hint)
            c.coords(self.pill, x1 - 14, y1 - 6, x2 + 14, y2 + 6)

    def copy(self):
        if self.english:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.english)
            self.set_hint("Copied the translation  ·  click to close")

    def close(self):
        if self.is_open:
            self.is_open = False
            show_window(self.win, False)
            self.c.delete("all")
            self.photo = None
            self.on_close()


class Toast:
    """A small note at the top of the screen ("Reading text..."). Never takes focus or clicks."""

    def __init__(self, root):
        import tkinter as tk

        self.root = root
        self.win = w = tk.Toplevel(root)
        w.overrideredirect(True)
        w.configure(bg=ACCENT)
        self.label = tk.Label(w, bg=PANEL, fg="white", font=("Segoe UI", 11, "bold"), padx=16, pady=7)
        self.label.pack(padx=1, pady=1)
        self.visible = False
        self.hide_at = None
        w.geometry("1x1+-200+-200")
        game_safe(w, click_through=True)
        set_window_alpha(w, 0.94)
        show_window(w, False)

    def show(self, text, seconds=None, color="white"):
        self.label.config(text=text, fg=color)
        self.win.update_idletasks()
        ww, wh = self.win.winfo_reqwidth(), self.win.winfo_reqheight()
        self.win.geometry(f"{ww}x{wh}+{(self.root.winfo_screenwidth() - ww) // 2}+24")
        if not self.visible:
            show_window(self.win, True)
            keep_on_top(self.win)
            self.visible = True
        self.hide_at = time.monotonic() + seconds if seconds else None

    def hide(self):
        if self.visible:
            show_window(self.win, False)
            self.visible = False
        self.hide_at = None

    def tick(self):
        if self.hide_at is not None and time.monotonic() > self.hide_at:
            self.hide()


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


SESSION_NAME = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}$")


def saved_sessions():
    """Every session with a chat log on disk (.jsonl and/or .txt), newest first. Only LiveSubs'
    own file names count, so nothing else in the folder is ever touched."""
    try:
        files = os.listdir(HISTORY_DIR)
    except OSError:
        return []
    names = {os.path.splitext(f)[0] for f in files if f.endswith((".jsonl", ".txt"))}
    return sorted((n for n in names if SESSION_NAME.match(n)), reverse=True)


def delete_sessions(names):
    for name in names:
        for ext in (".jsonl", ".txt"):
            try:
                os.remove(os.path.join(HISTORY_DIR, name + ext))
            except FileNotFoundError:
                pass
            except OSError as e:
                log("Couldn't delete an old chat log:", e)


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
    """Keeps every finished line and screenshot translation, and writes them to history/ (one pair
    of files per session). Only the last 2 sessions stay on disk: older chat logs delete themselves."""
    WAIT_FOR_ENGLISH = 20  # seconds to wait for a translation before logging without one

    def __init__(self, cfg):
        self.cfg = cfg
        self.session = time.strftime("%Y-%m-%d %H-%M-%S")
        self.entries = []
        self.pending = {}     # line_id -> entry still waiting for its English
        self.listeners = []   # called with the full entry list whenever it changes
        self.on_disk = False  # this session's log has been started
        self.clean_up()

    def clean_up(self):
        """Deletes the chat logs older than the last KEEP_SESSIONS sessions. This session only counts
        once something from it is saved, so the one before stays readable until then."""
        names = saved_sessions()
        keep = KEEP_SESSIONS - (1 if self.session in names else 0)
        old = [n for n in names if n != self.session][keep:]
        if old:
            delete_sessions(old)
            log(f"Deleted {len(old)} old chat log{'s' if len(old) != 1 else ''}")

    def _write(self, e):
        try:
            os.makedirs(HISTORY_DIR, exist_ok=True)
            base = os.path.join(HISTORY_DIR, self.session)
            with open(base + ".jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
            with open(base + ".txt", "a", encoding="utf-8") as f:
                f.write(format_entry(e) + "\n")
        except Exception as ex:
            log("Couldn't save history:", ex)
            return
        if not self.on_disk:
            self.on_disk = True
            self.clean_up()  # this session is now one of the kept ones

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
        bot.pack(fill="x", side="bottom", before=mid)  # stays visible when the window is made small
        ttk.Button(bot, text="Export...", command=self.export).pack(side="right")

        self.entries, self.shown, self.sessions = [], [], None
        self.history.listeners.append(self._live)
        w.protocol("WM_DELETE_WINDOW", self.close)
        self.refresh_sessions()
        self.reload()

    # ---- data ----
    def refresh_sessions(self):
        """Updates the session list. Returns True if the session being shown was deleted
        (it switches back to this session)."""
        names = [self.THIS] + [s for s in list_sessions() if s != self.history.session]
        if names != self.sessions:
            self.sessions = names
            self.session_cb.config(values=names)
        if self.session_cb.get() not in names:
            self.session_cb.set(self.THIS)
            return True
        return False

    def reload(self):
        sel = self.session_cb.get()
        self.entries = list(self.history.entries) if sel == self.THIS else load_session(sel)
        self.render(scroll_end=True)

    def _live(self, entries):
        if self.refresh_sessions():  # an old log just deleted itself while it was being read
            self.reload()
        elif self.session_cb.get() == self.THIS:
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
                tags = ("eng",)
                if has_foreign_script(e["eng"]):  # e.g. captions translated into Chinese
                    family = script_font(e["eng"])
                    t.tag_configure(f"eng_{family}", font=(family, 12, "bold"))
                    tags = ("eng", f"eng_{family}")
                t.insert("end", e["eng"] + "\n", tags)
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
        if not self.entries:
            t.insert("end", "Nothing here yet - translated lines show up as they happen.", "time")
        if scroll_end:
            t.see("end")
        else:
            t.yview_moveto(y)

    # ---- buttons ----
    def _as_text(self):
        return "\n".join(format_entry(e) for e in self.shown)

    def export(self):
        from tkinter import filedialog

        name = self.session_cb.get()
        default = (self.history.session if name == self.THIS else name) + ".txt"
        path = filedialog.asksaveasfilename(parent=self.win, defaultextension=".txt", initialfile=default,
                                            filetypes=[("Text file", "*.txt"), ("All files", "*.*")])
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._as_text())

    def show(self):
        self.win.deiconify()
        self.win.lift()
        if self.refresh_sessions():
            self.reload()

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
        HEARD_LANGUAGES.extend(self.cfg["heard_languages"])
        self.q = queue.Queue()
        self.asr = None
        self.speakers = None
        self.gate = None
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
        self.lens = LensView(self.root, self._lens_closed)  # translated screenshots
        self.toast = Toast(self.root)
        self.shot_job = None  # screenshot being read/translated right now
        self.build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.load_asr()
        self.load_translator()
        start_hotkey(lambda kind: self.q.put(("screenshot", kind)))
        self.apply_game_mode()
        self.root.after(50, self.poll)
        self.root.after(8000, self._warm_text_reader)

    def _warm_text_reader(self):
        """Start the screenshot text reader in the background, so the first screenshot is quick too."""
        def warm():
            try:
                TEXT_READER.start()
                log("Screenshot text reader ready")
            except Exception as e:
                log("Screenshot text reader unavailable (Qwen3-VL will read screenshots):", e)

        if TextReader.installed():
            threading.Thread(target=warm, daemon=True).start()

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

        # Which language the captions are in
        row(5, "Captions in")
        cf = ttk.Frame(f)
        cf.grid(row=5, column=1, sticky="ew")
        cf.columnconfigure(0, weight=1)
        self.target_cb = ttk.Combobox(cf, values=[n for n, _ in TARGETS], state="readonly")
        self.target_cb.set(next((n for n, code in TARGETS if code == c["target"]), "English"))
        self.target_cb.grid(row=0, column=0, sticky="ew")
        self.target_cb.bind("<<ComboboxSelected>>", lambda e: self.target_changed())
        self.both_var = tk.BooleanVar(value=c["both_ways"])
        self.both_cb = ttk.Checkbutton(cf, variable=self.both_var, command=self.target_changed)
        self.both_cb.grid(row=0, column=1, padx=(8, 0))

        # Help for the recognizer and translator
        row(6, "Topic")
        self.topic_var = tk.StringVar(value=c["topic"])
        te = ttk.Entry(f, textvariable=self.topic_var)
        te.grid(row=6, column=1, sticky="ew")
        row(7, "Custom words")
        self.words_var = tk.StringVar(value=c["words"])
        we = ttk.Entry(f, textvariable=self.words_var)
        we.grid(row=7, column=1, sticky="ew")
        for ent in (te, we):
            ent.bind("<FocusOut>", lambda e: self.hints_changed())
            ent.bind("<Return>", lambda e: self.hints_changed())

        # Language
        row(8, "Spoken language")
        self.lang_cb = ttk.Combobox(f, values=[spoken_label(code) for _, code in LANGUAGES], state="readonly")
        self.lang_cb.set(spoken_label(c["language"] if c["language"] in dict(LANGUAGES).values() else ""))
        self.lang_cb.grid(row=8, column=1, sticky="ew")
        self.lang_cb.bind("<<ComboboxSelected>>", lambda e: self.simple_changed())

        # Show mode
        row(9, "Show")
        self.show_cb = ttk.Combobox(f, values=list(SHOW_MODES), state="readonly")
        self.show_cb.set({v: k for k, v in SHOW_MODES.items()}.get(c["show"], "Original + translation"))
        self.show_cb.grid(row=9, column=1, sticky="ew")
        self.show_cb.bind("<<ComboboxSelected>>", lambda e: self.simple_changed())

        # Which apps to listen to
        row(10, "Listen to")
        lf = ttk.Frame(f)
        lf.grid(row=10, column=1, sticky="ew")
        lf.columnconfigure(0, weight=1)
        self.appmode_cb = ttk.Combobox(lf, values=list(APP_MODES), state="readonly")
        self.appmode_cb.set({v: k for k, v in APP_MODES.items()}.get(c["app_mode"], "All apps"))
        self.appmode_cb.grid(row=0, column=0, sticky="ew")
        self.appmode_cb.bind("<<ComboboxSelected>>", lambda e: self.app_mode_changed())
        self.apps_btn = ttk.Button(lf, text="Apps...", command=self.pick_apps)
        self.apps_btn.grid(row=0, column=1, padx=(6, 0))
        self.apps_var = tk.StringVar()
        ttk.Label(f, textvariable=self.apps_var, foreground="#777").grid(row=11, column=1, sticky="w")

        # Output device (for "All apps")
        row(12, "Output device")
        af = ttk.Frame(f)
        af.grid(row=12, column=1, sticky="ew")
        af.columnconfigure(0, weight=1)
        self.audio_cb = ttk.Combobox(af, state="readonly")
        self.audio_cb.grid(row=0, column=0, sticky="ew")
        self.audio_cb.bind("<<ComboboxSelected>>", lambda e: self.audio_changed())
        ttk.Button(af, text="↻", width=3, command=self.refresh_audio).grid(row=0, column=1, padx=(6, 0))
        self.refresh_audio()

        ttk.Separator(f).grid(row=13, column=0, columnspan=2, sticky="ew", pady=10)

        def slider(i, label, key, lo, hi, cast):
            row(i, label)
            var = tk.DoubleVar(value=c[key])
            ttk.Scale(f, from_=lo, to=hi, variable=var,
                      command=lambda v: self.look_changed(key, cast(float(v)))).grid(row=i, column=1, sticky="ew")
            return var

        self.font_var = slider(14, "Text size", "font", 10, 48, int)
        slider(15, "Background", "opacity", 0.0, 1.0, float)  # 0 = no box, just the text
        slider(16, "Max width", "width", 0.25, 1.0, float)
        slider(17, "Lines on screen", "history", 1, 4, lambda v: int(round(v)))

        self.spk_var = tk.BooleanVar(value=c["speakers"])
        ttk.Checkbutton(f, text="Label different speakers", variable=self.spk_var,
                        command=lambda: self.look_changed("speakers", self.spk_var.get())
                        ).grid(row=18, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.hide_en_var = tk.BooleanVar(value=c["hide_english"])
        ttk.Checkbutton(f, text="Hide lines that don't need translating", variable=self.hide_en_var,
                        command=lambda: self.look_changed("hide_english", self.hide_en_var.get())
                        ).grid(row=19, column=0, columnspan=2, sticky="w")

        self.click_var = tk.BooleanVar(value=c["click_through"])
        ttk.Checkbutton(f, text=f"Lock captions in place   ({HOTKEY_LOCK_LABEL})", variable=self.click_var,
                        command=lambda: self.set_locked(self.click_var.get())
                        ).grid(row=20, column=0, columnspan=2, sticky="w")
        self.overlay.on_lock = self.set_locked
        self.game_var = tk.BooleanVar(value=c["game_mode"])
        ttk.Checkbutton(f, text="Game mode (smoother games - captions appear when each line ends)",
                        variable=self.game_var, command=self.game_mode_changed
                        ).grid(row=21, column=0, columnspan=2, sticky="w")
        self.auto_var = tk.BooleanVar(value=c["autostart"])
        ttk.Checkbutton(f, text="Start subtitles when the app opens", variable=self.auto_var,
                        command=self.simple_changed).grid(row=22, column=0, columnspan=2, sticky="w")

        bf = ttk.Frame(f)
        bf.grid(row=23, column=0, columnspan=2, sticky="ew", pady=(14, 0))
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
                  foreground="#888").grid(row=24, column=0, columnspan=2, pady=(10, 0))
        self._sync_device_box()
        self._sync_app_widgets()
        self._sync_target_widgets()

    def _sync_target_widgets(self):
        target = self.cfg.get("target") or "en"
        if target == "en":
            self.both_cb.config(text="Both ways", state="disabled")
        else:  # e.g. English speech -> Chinese captions, and Chinese speech -> English captions
            self.both_cb.config(text=f"Both ways ({TARGET_SHORT[target]} → English)", state="normal")

    def target_changed(self):
        self.cfg["target"] = dict(TARGETS)[self.target_cb.get()]
        self.cfg["both_ways"] = bool(self.both_var.get())
        save_settings(self.cfg)
        self._sync_target_widgets()  # new lines use it straight away

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

    def _translate_image(self, image, origin, full):
        """Reads the text in the screenshot, translates it and shows the screenshot again with the English
        painted over the original text. A picked box shows straight away and fills in when ready."""
        job = ScreenJob(image, origin, full)
        self.shot_job = job
        if full:
            self.toast.show(f"Reading the screen...   ({HOTKEY_FULL_LABEL} again to cancel)")
        else:
            self.lens.show(image, origin, False, "Translating...   (click or Esc to cancel)")
            set_escape_hotkey(True)

        def work():
            try:
                t0 = time.monotonic()
                blocks = read_screenshot(job, self.cfg, status=lambda m: self.q.put(("screen_status", job, m)))
                done = [b for b in blocks if b.get("eng")]
                picture = paint_translation(image, blocks, self.cfg.get("target") or "en") if done else None
                log(f"Screenshot: {len(blocks)} text blocks, {len(done)} translated "
                    f"({time.monotonic() - t0:.1f}s)")
                self.q.put(("screen_result", job, blocks, picture, None))
            except Cancelled:
                log("Screenshot translation cancelled")
            except Exception as e:
                log(traceback.format_exc())
                self.q.put(("screen_result", job, None, None, str(e)))
            # put the subtitle translator back on the GPU so the next line isn't delayed
            # (unless another screenshot has already started)
            tr = self.cfg["translator"]
            if (tr not in ("off", VISION_MODEL) and self.translator.ready_model
                    and self.shot_job in (None, job)):
                try:
                    HOST.use(tr)
                except Exception as e:
                    log("Couldn't reload translator:", e)

        threading.Thread(target=work, daemon=True).start()

    def show_screen_result(self, job, blocks, picture, err):
        if job is not self.shot_job:
            return  # closed / cancelled meanwhile
        self.shot_job = None
        if err:
            self.lens.close()
            self.toast.show(f"Couldn't translate the screenshot: {err}", 8, "#ff8f8f")
            return
        done = [b for b in blocks if b.get("eng")]
        if not done:
            self.lens.close()
            short = TARGET_SHORT.get(self.cfg.get("target") or "en", "English")
            self.toast.show(f"Nothing to translate - it's all {short} already" if blocks else "No text found", 3)
            return
        english = "\n".join(b["eng"] for b in done)
        self.history.screenshot("\n".join(b["text"] for b in done), english)
        self.toast.hide()
        if job.full:
            hint = (f"{len(done)} translated  ·  click, Esc or {HOTKEY_FULL_LABEL} to close  ·  "
                    "right-click to copy the translation")
        else:
            hint = "Click or Esc to close  ·  right-click copies"
        self.lens.show(picture, job.origin, job.full, hint, english)
        set_escape_hotkey(True)

    def _lens_closed(self):
        set_escape_hotkey(False)
        if self.shot_job is not None:  # closed before the translation was ready
            self.shot_job.cancelled = True
            self.shot_job = None

    def close_screen_translation(self):
        """Closes the translated screenshot, or cancels one in progress. False if there was none."""
        busy = self.shot_job is not None or self.lens.is_open
        if self.shot_job is not None:
            self.shot_job.cancelled = True
            self.shot_job = None
        self.lens.close()
        self.toast.hide()
        return busy

    def _hide_for_shot(self):
        """Make our own windows invisible for a moment so they're not in the screenshot."""
        self.close_screen_translation()  # e.g. the button was used while a translated screenshot was showing
        self.overlay.hide_now()
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
        self._translate_image(image, (0, 0), True)

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
                image, box = args
                self._translate_image(image, box[:2], False)

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
        self.cfg["language"] = {spoken_label(code): code for _, code in LANGUAGES}.get(self.lang_cb.get(), "")
        self.cfg["show"] = SHOW_MODES.get(self.show_cb.get(), "both")
        self.cfg["autostart"] = bool(self.auto_var.get())
        save_settings(self.cfg)
        self.overlay.apply()
        lang = self.cfg["language"]
        if self.cfg["engine"] == "sensevoice" and lang and lang not in SENSEVOICE_LANGS:
            self.set_status(f"SenseVoice can't do {LANG_NAME[lang]} - pick Best accuracy or Whisper", "#d9534f")

    def look_changed(self, key, value):
        self.cfg[key] = value
        self.overlay.apply()
        save_settings(self.cfg)

    def audio_changed(self):
        sel = self.audio_cb.get()
        self.cfg["audio_device"] = "" if sel == DEFAULT_AUDIO else sel
        save_settings(self.cfg)
        self.restart_listening()

    def hints_changed(self):
        topic, words = self.topic_var.get().strip(), self.words_var.get().strip()
        if topic == self.cfg["topic"] and words == self.cfg["words"]:
            return
        reload_asr = parse_words(words) != parse_words(self.cfg["words"]) and self.cfg["engine"] == "qwen3"
        self.cfg["topic"], self.cfg["words"] = topic, words
        save_settings(self.cfg)
        if reload_asr:
            self.load_asr()  # Qwen3-ASR takes its word list when it loads

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
                say = lambda s: self.q.put(("status", s))  # noqa: E731
                asr = load_asr(engine, dev, status=say, game_mode=self.cfg.get("game_mode", True),
                               words=parse_words(self.cfg.get("words")))
                if not self.gate:
                    try:
                        self.gate = SpeechGate(say)
                    except Exception as e:
                        log("Voice detector unavailable, using loudness instead:", e)
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
        self.listener = Listener(self.asr, self.translator, self.cfg, self.q, self.speakers, self.gate)
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
        set_gpu_priority(None, on)
        HOST.set_game_mode(on)
        ASR_HOST.set_game_mode(on)

    def game_mode_changed(self):
        self.cfg["game_mode"] = bool(self.game_var.get())
        save_settings(self.cfg)
        self.apply_game_mode()
        if self.cfg["engine"] in ("sensevoice", "qwen3"):
            self.load_asr()  # Game mode uses Qwen3-ASR alone on the GPU; otherwise SenseVoice adds live text

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
                elif kind in ("final", "refined"):
                    _lid, _orig, _lang, _needs, _spk, _pending = msg[1:7]
                    if kind == "final":
                        self.overlay.final(_lid, _orig, _lang, _needs, _spk)
                    else:
                        self.overlay.refined(_lid, _orig, _lang, _needs, _spk)
                    if not _pending:  # log once the text is settled (after the second listen, if any)
                        self.history.line(_lid, _orig, _lang, _needs, _spk)
                    self.hear_var.set(f"Hearing: {LANG_NAME.get(_lang, _lang)}")
                elif kind == "retract":
                    self.overlay.retract(msg[1])
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
                    if msg[1] in ("full", "region", "escape"):
                        # the same hotkey (or Esc) closes a translated screenshot, or cancels one in progress
                        if not self.close_screen_translation() and msg[1] != "escape":
                            self.start_full_shot() if msg[1] == "full" else self.start_snip()
                    elif msg[1] == "lock":
                        self.set_locked(not self.cfg["click_through"])
                    elif msg[1] == "hide":
                        self.toggle_visible()
                elif kind == "screen_status":
                    job, text = msg[1:]
                    if job is self.shot_job:
                        if job.full:
                            self.toast.show(f"{text}   ({HOTKEY_FULL_LABEL} again to cancel)")
                        elif self.lens.is_open:
                            self.lens.set_hint(text)
                elif kind == "screen_result":
                    self.show_screen_result(*msg[1:])
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
        self.toast.tick()
        self.history.tick()
        self.root.after(50, self.poll)

    def quit(self):
        lst = self.listener
        self.stop_listening()
        self.cfg["heard_languages"] = list(HEARD_LANGUAGES)
        save_settings(self.cfg)
        if lst:
            lst.join(timeout=1.5)
        self.history.flush()
        HOST.stop()  # close the AI engines and free the GPU memory
        ASR_HOST.stop()
        TEXT_READER.stop()
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

    if "--translator" in sys.argv:  # picked in the installer on a first install
        choice = sys.argv[sys.argv.index("--translator") + 1]
        if choice in TRANSLATORS.values():
            cfg["translator"] = choice
            save_settings(cfg)
    ensure_llama_builds(say)
    say("AI engine downloaded.")
    # Only what the default setup uses. Qwen3-VL (screenshots without the text reader) and the CPU copy of
    # Qwen3-ASR (no usable GPU) download by themselves if they're ever needed.
    keys = [ASR_MODEL] + ([cfg["translator"]] if cfg["translator"] != "off" else [])
    for key in keys:
        ensure_model(key, say)
    ensure_sensevoice(say)  # small: live text outside Game mode
    say("AI models downloaded.")

    try:
        asr = load_asr(cfg["engine"], cfg["device"], say, cfg.get("game_mode", True), parse_words(cfg.get("words")))
        say(f"Speech model ready: {asr.name} ({'GPU' if asr.device == 'cuda' else 'CPU'})")
        if asr.kind == "qwen3gpu":
            t0 = time.monotonic()
            asr.listen(np.zeros(WHISPER_SR * 3, np.float32))
            say(f"Speech recognition on the GPU: {(time.monotonic() - t0) * 1000:.0f} ms per line")
        SpeechGate(say)
        say("Voice detector ready.")
        SpeakerTracker(say)
        say("Speaker model ready.")

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

        if TextReader.installed():
            try:
                msg = TEXT_READER.request({"cmd": "prepare"}, timeout=900)  # also fetches the Korean reader
                say("Screenshot text reader ready" + ("" if msg.get("korean") else
                    f" (Korean text unavailable: {msg.get('korean_error')})"))
            except Exception as e:  # screenshots still work: Qwen3-VL reads them instead (slower)
                say(f"Screenshot text reader unavailable ({e}) - Qwen3-VL will read screenshots instead.")
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (520, 180), (24, 28, 36))  # a small game menu in Chinese
        draw = ImageDraw.Draw(img)
        draw.text((40, 30), "开始游戏", font=paint_font(44, "开始游戏"), fill="white")
        draw.text((40, 105), "设置", font=paint_font(44, "设置"), fill=(255, 210, 90))
        t0 = time.monotonic()
        blocks = read_screenshot(ScreenJob(img, (0, 0), True), cfg, say)
        paint_translation(img, blocks)
        found = ", ".join(f"{b['text']} -> {b.get('eng') or '(kept)'}" for b in blocks) or "no text"
        say(f"Screenshot test: {found}  ({time.monotonic() - t0:.1f} s)")
    finally:
        HOST.stop()
        ASR_HOST.stop()
        TEXT_READER.stop()


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
    use_windows_certificates()
    if "--probe-sv-gpu" in sys.argv:
        return probe_sensevoice_gpu()
    if "--ocr-server" in sys.argv:
        return ocr_server()
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
