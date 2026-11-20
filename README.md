# LiveSubs

Real-time subtitles and translation for anything your PC plays: games, streams, videos, voice chat.
Everything runs locally on your own PC. Nothing you hear is sent to an online service.

It's built for mixed-language speech (Mandarin with English, for example), noisy game audio, and playing
fullscreen games while it runs.

## Features

- **Live captions over any app.** The original speech and its translation appear in a small box that
  stays on top of games and videos without stealing focus.
- **Captions in your language.** English by default, or Chinese (Simplified or Traditional), Korean,
  Japanese, Spanish and more. *Both ways* translates in both directions at once, e.g. English speech into
  Chinese and Chinese speech into English.
- **Two-pass speech recognition.** SenseVoice gives an instant first version of each line, then Qwen3-ASR
  listens again and corrects it. This is especially good for mixed Mandarin/English and background noise.
- **Natural translation.** Qwen3 14B, running on your GPU, translates for meaning: slang, tone and swearing
  carry over instead of word-for-word output.
- **Voice detection.** Game sounds and music aren't mistaken for speech, and lines end at real pauses.
- **Speaker colours.** Different voices get their own colour and number.
- **Per-app audio.** Listen to everything, everything except some apps (e.g. Spotify), or only chosen apps.
- **Screenshot translation.** Press a hotkey and the screenshot comes back with English painted over the
  foreign text, right where it was (like Google Lens). Works on menus, chat, signs and video subtitles.
- **History.** A searchable, timestamped log of every translated line, saved as plain text.
- **Game mode.** Speech recognition moves onto the graphics card, so the CPU is left to the game (see below).
- **Topic and custom words.** Tell it what you're watching and which names or terms to expect.

Supported speech: Mandarin, Cantonese, Korean, Japanese and English with SenseVoice + Qwen3-ASR (Qwen3-ASR
also handles many other languages), or any language Whisper supports.

## Requirements

- Windows 10 (version 2004 or newer) or Windows 11
- An NVIDIA GPU with 12 GB+ of video memory is recommended for the default translator
  (pick Qwen3 8B in the app if you have less; other GPUs may work through Vulkan)
- About 19 GB of free disk space for the AI models
- Python 3.12 (the installer looks for it and points you to the download if it's missing)

## Install

1. Download this repository (green **Code** button → **Download ZIP**) and unzip it, or `git clone` it.
2. Double-click **Install LiveSubs.bat**. It installs the Python packages, downloads the models and creates
   Desktop and Start menu shortcuts. The first run downloads about 19 GB; if it's interrupted, run it again
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
The **Background** slider fades only the box behind the text; set it to 0 for text with no box.

## Translating text on screen

Press `Ctrl+Alt+Y` for the whole screen, or `Ctrl+Alt+T` and drag a box around some text. The screenshot
appears where it was taken, with English painted over every line that wasn't English already. Lines that
belong to one sentence (a wrapped paragraph or a two-line subtitle) are translated together.

- **Close it:** click it, press `Esc`, or press the same hotkey again (this also cancels one that's still
  being read).
- **Copy the English:** right-click it.
- Every translated screenshot is saved in the history.

## Captions in other languages

Pick the language under **Captions in**. Speech in any other language is translated into it; speech that's
already in it is shown as it is (tick *Hide lines that don't need translating* to leave those out).

Tick **Both ways** to translate in both directions: with captions in Chinese, English speech gets Chinese
captions and Chinese speech gets English captions. That suits a mixed lobby or watching with someone who
reads the other language. Screenshot translation uses the same caption language.

## Tips for better accuracy

- **Topic:** describe what you're watching, e.g. *"Valorant ranked, Chinese and English teammates"*.
- **Custom words:** list names and terms, separated by commas (agent names, callouts, friends' names).
  These steer both the speech recognition and the translation.
- **Spoken language:** set it if you know the language, instead of auto-detect.
- **Listen to:** choose *Only these apps…* to ignore music and other programs.

## Playing games

- Keep **Game mode** on (it's on by default). Games like Valorant are held back by the CPU, not the
  graphics card, so Game mode does the heavy work on the GPU: Qwen3-ASR runs there by itself (through
  llama.cpp, using about 1.5 GB of video memory), and each line is transcribed once, when the speaker
  pauses, instead of live. The CPU only checks whether someone is talking and whose voice it is. All of
  LiveSubs' GPU work runs at lower priority than the game's, and the captions are fully hidden whenever
  nobody is talking.
- Use **Windowed Fullscreen** (borderless) in the game.
- Cap the game's frame rate a little below what your PC can reach (in Valorant: *Limit FPS Always*).
  The headroom keeps frame times steady while LiveSubs translates.
- The screenshot hotkeys swap AI models in video memory, which takes a few seconds of heavy GPU work,
  so avoid them mid-fight.

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
  → SenseVoice: instant first version of the line      (not used in Game mode)
  → Qwen3-ASR via llama.cpp (GPU): careful listen      (CPU version if there's no GPU)
  → Qwen3 14B via llama.cpp (GPU): natural translation (English or your caption language)
  → caption box + history
```

Screenshot translation uses Qwen3-VL 8B. Speaker colours use a 3D-Speaker voice-fingerprint model.

## Built with

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx): runs SenseVoice, Qwen3-ASR, Silero VAD and speaker models
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice), Qwen3-ASR, [Silero VAD](https://github.com/snakers4/silero-vad),
  [3D-Speaker](https://github.com/modelscope/3D-Speaker)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) with [Qwen3](https://huggingface.co/Qwen/Qwen3-14B-GGUF),
  [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF) and
  [Qwen3-ASR](https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [PyAudioWPatch](https://github.com/s0d3s/PyAudioWPatch),
  [pycaw](https://github.com/AndreMiras/pycaw)

The AI models are downloaded from their publishers and each one has its own license.

## License

MIT - see [LICENSE](LICENSE).
