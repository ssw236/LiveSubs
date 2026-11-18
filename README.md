# LiveSubs

Real-time subtitles and English translation for anything your PC plays: games, streams, videos, voice chat.
Everything runs locally on your own PC. Nothing you hear is sent to an online service.

It's built for mixed-language speech (Mandarin with English, for example), noisy game audio, and playing
fullscreen games while it runs.

## Features

- **Live captions over any app.** The original speech and its English translation appear in a small box that
  stays on top of games and videos without stealing focus.
- **Two-pass speech recognition.** SenseVoice gives an instant first version of each line, then Qwen3-ASR
  listens again and corrects it. This is especially good for mixed Mandarin/English and background noise.
- **Natural translation.** Qwen3 14B, running on your GPU, translates for meaning: slang, tone and swearing
  carry over instead of word-for-word output.
- **Voice detection.** Game sounds and music aren't mistaken for speech, and lines end at real pauses.
- **Speaker colours.** Different voices get their own colour and number.
- **Per-app audio.** Listen to everything, everything except some apps (e.g. Spotify), or only chosen apps.
- **Screenshot translation.** Translate text on screen (menus, chat, signs) with a hotkey.
- **History.** A searchable, timestamped log of every translated line, saved as plain text.
- **Game mode.** Lower CPU/GPU use and priority so games stay smooth.
- **Topic and custom words.** Tell it what you're watching and which names or terms to expect.

Supported speech: Mandarin, Cantonese, Korean, Japanese and English with SenseVoice + Qwen3-ASR (Qwen3-ASR
also handles many other languages), or any language Whisper supports.

## Requirements

- Windows 10 (version 2004 or newer) or Windows 11
- An NVIDIA GPU with 12 GB+ of video memory is recommended for the default translator
  (pick Qwen3 8B in the app if you have less; other GPUs may work through Vulkan)
- About 18 GB of free disk space for the AI models
- Python 3.12 (the installer looks for it and points you to the download if it's missing)

## Install

1. Download this repository (green **Code** button → **Download ZIP**) and unzip it, or `git clone` it.
2. Double-click **Install LiveSubs.bat**. It installs the Python packages, downloads the models and creates
   Desktop and Start menu shortcuts. The first run downloads about 18 GB; if it's interrupted, run it again
   and it continues where it stopped.
3. Open **LiveSubs** from the Desktop shortcut.

In games, use **Windowed Fullscreen** (borderless) display mode. Exclusive fullscreen doesn't allow any
window to appear on top of the game.

## Hotkeys

| Keys | Action |
| --- | --- |
| `Ctrl+Alt+Y` | Translate the whole screen (stays in the game) |
| `Ctrl+Alt+T` | Drag a box around text to translate |
| `Ctrl+Alt+L` | Lock / unlock the captions (locked = click-through) |
| `Ctrl+Alt+H` | Hide / show the captions |

Drag the caption box to move it, scroll on it to resize the text, and right-click it for options.

## Tips for better accuracy

- **Topic:** describe what you're watching, e.g. *"Valorant ranked, Chinese and English teammates"*.
- **Custom words:** list names and terms, separated by commas (agent names, callouts, friends' names).
  These steer both the speech recognition and the translation.
- **Spoken language:** set it if you know the language, instead of auto-detect.
- **Listen to:** choose *Only these apps…* to ignore music and other programs.

## Files it creates

These stay on your PC and are excluded from git by `.gitignore`:

| Path | What it is |
| --- | --- |
| `models/` | Downloaded AI models and the llama.cpp engine |
| `history/` | Your chat logs (`.txt` to read, `.jsonl` for the app) |
| `settings.json` | Your settings |
| `livesubs.log`, `ai-engine.log` | Logs for troubleshooting |

## How it works

```
speakers / chosen apps (WASAPI loopback)
  → Silero VAD: is someone talking?
  → SenseVoice: instant first version of the line
  → Qwen3-ASR: second, more careful listen
  → Qwen3 14B via llama.cpp (GPU): natural English
  → caption box + history
```

Screenshot translation uses Qwen3-VL 8B. Speaker colours use a 3D-Speaker voice-fingerprint model.

## Built with

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx): runs SenseVoice, Qwen3-ASR, Silero VAD and speaker models
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice), Qwen3-ASR, [Silero VAD](https://github.com/snakers4/silero-vad),
  [3D-Speaker](https://github.com/modelscope/3D-Speaker)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) with [Qwen3](https://huggingface.co/Qwen/Qwen3-14B-GGUF) and
  [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [PyAudioWPatch](https://github.com/s0d3s/PyAudioWPatch),
  [pycaw](https://github.com/AndreMiras/pycaw)

The AI models are downloaded from their publishers and each one has its own license.

## License

MIT - see [LICENSE](LICENSE).
