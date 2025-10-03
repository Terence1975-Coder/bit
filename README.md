# MP4 → Word Transcriber

A tiny Streamlit app that turns MP4/MOV/MP3/WAV/M4A into transcripts and Word docs (\*.docx).  
Uses [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) for accurate transcription.

## Features
- Upload audio or video; audio extracted via FFmpeg
- Downloads: **.txt**, **.srt** (subtitles), **.docx** (Word with timecodes)
- Works on CPU (choose *float32* or *int8*). GPU-friendly if your machine supports it.

## Quick start (Windows / PowerShell)
```powershell
# in this folder
.un.ps1
```

This script:
1) creates a venv,  
2) installs requirements,  
3) launches the app at http://localhost:8501

## Manual setup (any OS)
``ash
python -m venv .venv
# Windows:
..venvScriptsactivate
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### FFmpeg
- Easiest: Python package `imageio-ffmpeg` (already in requirements) provides a portable binary.
- Or system install:
  - Windows: `winget install --id Gyan.FFmpeg -e`
  - macOS: `brew install ffmpeg`
  - Debian/Ubuntu: `sudo apt-get install ffmpeg`

### Tips
- **Compute Type**: on CPU use `float32` (most compatible) or `int8` (faster). On GPU, try `float16`.
- **Models**: `small.en` is fast for English; use non-.en models for multilingual audio.
- Add diarization, redaction, or batch processing later if needed.
