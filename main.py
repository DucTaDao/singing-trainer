from time import time
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
import numpy as np
import librosa
import soundfile as sf
import tempfile
import io
from fastapi.responses import FileResponse, StreamingResponse
import shutil
import os
import subprocess
import shutil
import time
import sys
import wave, contextlib

app = FastAPI()

# ✅ CORS (để frontend localhost:5173 gọi được)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# DB
engine = create_engine("sqlite:///test.db")
Session = sessionmaker(bind=engine)
Base = declarative_base()

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_note(m):
    m = int(round(m))
    return f"{NOTE_NAMES[m % 12]}{m//12 - 1}"


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    device_id = Column(String, unique=True)


class Analysis(Base):
    __tablename__ = "analyses"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    f0_min = Column(Float)
    f0_max = Column(Float)
    note_min = Column(String)
    note_max = Column(String)


Base.metadata.create_all(engine)

import subprocess


def convert_to_wav_bytes(input_bytes: bytes, input_ext: str) -> bytes:
    # ffmpeg đọc từ stdin, xuất wav ra stdout
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-f",
        "wav",
        "-ac",
        "1",
        "-ar",
        "44100",
        "pipe:1",
    ]
    p = subprocess.run(
        cmd, input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", errors="ignore"))
    return p.stdout


def load_audio_from_upload(data: bytes, filename: str):
    # 1) best for WAV/FLAC
    try:
        ext = (filename.split(".")[-1] if filename and "." in filename else "").lower()
        if ext in ["webm", "ogg"]:
            data = convert_to_wav_bytes(data, ext)
        y, sr = sf.read(io.BytesIO(data), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y, sr
    except Exception:
        pass

    # 2) fallback: temp file + librosa.load (webm/ogg may need ffmpeg)
    suffix = "." + (filename.split(".")[-1] if filename and "." in filename else "bin")
    with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as f:
        f.write(data)
        f.flush()
        y, sr = librosa.load(f.name, sr=None, mono=True)
        return y, sr


@app.post("/analyze-uuu")
async def analyze_uuu(file: UploadFile, device_id: str = Form(...)):
    db = Session()

    # get or create user
    user = db.query(User).filter_by(device_id=device_id).first()
    if not user:
        user = User(device_id=device_id)
        db.add(user)
        db.commit()
        db.refresh(user)

    raw = await file.read()

    # load audio
    try:
        y, sr = load_audio_from_upload(raw, file.filename or "")
    except Exception as e:
        return {
            "ok": False,
            "error": "Không đọc được audio. Thử WAV (dễ nhất) hoặc cài ffmpeg nếu đang gửi webm/ogg.",
            "details": str(e),
        }

    # pitch (pyin): f0 per frame, unvoiced = NaN
    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
    )

    voiced = f0[~np.isnan(f0)]
    if len(voiced) == 0:
        return {
            "ok": False,
            "error": "Không detect được pitch. Hát to/đều hơn và gần mic.",
        }

    f0_min = float(np.min(voiced))
    f0_max = float(np.max(voiced))
    note_min = midi_to_note(float(librosa.hz_to_midi(f0_min)))
    note_max = midi_to_note(float(librosa.hz_to_midi(f0_max)))
    voiced_ratio = float(len(voiced) / len(f0))

    # save
    analysis = Analysis(
        user_id=user.id,
        f0_min=f0_min,
        f0_max=f0_max,
        note_min=note_min,
        note_max=note_max,
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    return {
        "ok": True,
        "analysis_id": analysis.id,
        "f0_min_hz": round(f0_min, 2),
        "f0_max_hz": round(f0_max, 2),
        "note_min": note_min,
        "note_max": note_max,
        "voiced_ratio": round(voiced_ratio, 3),
    }


def detect_vocal_range(wav_path):
    y, sr = librosa.load(wav_path, sr=None, mono=True)

    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
    )

    voiced = f0[~np.isnan(f0)]
    if len(voiced) == 0:
        raise RuntimeError("No pitch detected in vocals")

    min_midi = np.percentile(librosa.hz_to_midi(voiced), 5)
    max_midi = np.percentile(librosa.hz_to_midi(voiced), 95)
    return float(min_midi), float(max_midi)


def transpose_wav(in_path, out_path, semitones):
    y, sr = librosa.load(in_path, sr=None, mono=True)
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)
    sf.write(out_path, y_shifted, sr)


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


def resolve_demucs_paths(demucs_out: str):
    # robust: demucs_out/<model>/<song>/{vocals.wav,no_vocals.wav}
    model_dirs = [
        d for d in os.listdir(demucs_out) if os.path.isdir(os.path.join(demucs_out, d))
    ]
    if not model_dirs:
        raise RuntimeError("Demucs output missing (no model dir)")
    model_dir = model_dirs[0]

    song_root = os.path.join(demucs_out, model_dir)
    song_dirs = [
        d for d in os.listdir(song_root) if os.path.isdir(os.path.join(song_root, d))
    ]
    if not song_dirs:
        raise RuntimeError("Demucs output missing (no song dir)")
    song_dir = song_dirs[0]

    vocals_path = os.path.join(song_root, song_dir, "vocals.wav")
    inst_path = os.path.join(song_root, song_dir, "no_vocals.wav")

    if not (os.path.exists(vocals_path) and os.path.exists(inst_path)):
        raise RuntimeError("Demucs output missing (vocals/no_vocals not found)")
    return vocals_path, inst_path


def mix_wavs(vocals_path: str, inst_path: str, out_path: str):
    v, sr_v = librosa.load(vocals_path, sr=None, mono=True)
    i, sr_i = librosa.load(inst_path, sr=None, mono=True)

    # ensure same SR
    sr = sr_v
    if sr_i != sr_v:
        i = librosa.resample(i, orig_sr=sr_i, target_sr=sr_v)
        sr = sr_v

    # align length
    n = max(len(v), len(i))
    if len(v) < n:
        v = np.pad(v, (0, n - len(v)))
    if len(i) < n:
        i = np.pad(i, (0, n - len(i)))

    y = v + i
    # avoid clipping
    peak = np.max(np.abs(y)) if len(y) else 1.0
    if peak > 0.99:
        y = y / peak * 0.99

    sf.write(out_path, y, sr)


@app.post("/compute-k")
async def compute_k(file: UploadFile = File(...), device_id: str = Form(...)):
    db = Session()
    try:
        user = db.query(User).filter_by(device_id=device_id).first()
        if not user:
            return {"ok": False, "error": "User chưa analyze giọng"}

        analysis = (
            db.query(Analysis)
            .filter_by(user_id=user.id)
            .order_by(Analysis.id.desc())
            .first()
        )
        if not analysis:
            return {"ok": False, "error": "Chưa có vocal range (hãy analyze trước)"}

        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, file.filename or "input")
            with open(in_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            demucs_out = os.path.join(tmpdir, "demucs")
            os.makedirs(demucs_out, exist_ok=True)

            cmd = [
                sys.executable,
                "-m",
                "demucs",
                "--two-stems",
                "vocals",
                "-o",
                demucs_out,
                in_path,
            ]

            try:
                subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=300,
                    check=True,
                )
            except subprocess.TimeoutExpired:
                return {
                    "ok": False,
                    "error": "Demucs timeout (>300s). Test file ngắn hơn.",
                }
            except subprocess.CalledProcessError as e:
                return {
                    "ok": False,
                    "error": "Demucs failed",
                    "details": (e.stderr or "")[-2000:],
                }

            vocals_path, inst_path = resolve_demucs_paths(demucs_out)

            # song range
            song_min, song_max = detect_vocal_range(vocals_path)
            song_center = (song_min + song_max) / 2

            # user range
            user_min = float(librosa.hz_to_midi(analysis.f0_min))
            user_max = float(librosa.hz_to_midi(analysis.f0_max))
            user_center = (user_min + user_max) / 2

            k = int(round(user_center - song_center))
            k = clamp_int(k, -12, 12)

            return {
                "ok": True,
                "k": k,
                "song_note_min": midi_to_note(song_min),
                "song_note_max": midi_to_note(song_max),
                "user_note_min": analysis.note_min,
                "user_note_max": analysis.note_max,
            }
    finally:
        db.close()


@app.post("/fit-song-to-user")
async def fit_song_to_user(
    file: UploadFile = File(...),
    device_id: str = Form(...),
    k: int = Form(...),
    return_mix: str = Form("true"),
):
    db = Session()
    try:
        user = db.query(User).filter_by(device_id=device_id).first()
        if not user:
            return {"ok": False, "error": "User chưa analyze giọng"}

        analysis = (
            db.query(Analysis)
            .filter_by(user_id=user.id)
            .order_by(Analysis.id.desc())
            .first()
        )
        if not analysis:
            return {"ok": False, "error": "Chưa có vocal range (hãy analyze trước)"}

        k = clamp_int(k, -12, 12)

        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, file.filename or "input")

            with open(in_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            # 1) demucs
            demucs_out = os.path.join(tmpdir, "demucs")
            os.makedirs(demucs_out, exist_ok=True)

            cmd = [
                sys.executable,
                "-m",
                "demucs",
                "--two-stems",
                "vocals",
                "-o",
                demucs_out,
                in_path,
            ]

            try:
                subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=300,
                    check=True,
                )
            except subprocess.TimeoutExpired:
                return {
                    "ok": False,
                    "error": "Demucs timeout (>300s). Test file ngắn hơn.",
                }
            except subprocess.CalledProcessError as e:
                return {
                    "ok": False,
                    "error": "Demucs failed",
                    "details": (e.stderr or "")[-2000:],
                }

            vocals_path, inst_path = resolve_demucs_paths(demucs_out)

            # 2) transpose BOTH
            inst_shifted = os.path.join(tmpdir, f"inst_{k}st.wav")
            vocals_shifted = os.path.join(tmpdir, f"vocals_{k}st.wav")
            out_path = os.path.join(tmpdir, f"out_{k}st.wav")

            transpose_wav(inst_path, inst_shifted, k)
            transpose_wav(vocals_path, vocals_shifted, k)

            # 3) return mix OR only instrumental
            if return_mix.lower() == "true":
                mix_wavs(vocals_shifted, inst_shifted, out_path)
                out_filename = f"song_mix_{k}st.wav"
            else:
                out_path = inst_shifted
                out_filename = f"song_inst_{k}st.wav"

            with open(out_path, "rb") as f:
                out_bytes = f.read()

        headers = {"Content-Disposition": f'attachment; filename="{out_filename}"'}
        return StreamingResponse(
            io.BytesIO(out_bytes), media_type="audio/wav", headers=headers
        )

    finally:
        db.close()


def extract_pitch_contour_midi(wav_path: str, sr_target=16000, hop_length=160):
    y, sr = librosa.load(wav_path, sr=sr_target, mono=True)
    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        hop_length=hop_length,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    f0_midi = []
    for v in f0:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            f0_midi.append(None)
        else:
            f0_midi.append(float(librosa.hz_to_midi(v)))
    return times.tolist(), f0_midi


def postprocess_words(words, extend_last=0.6):
    words = [
        w for w in words if w.get("start") is not None and w.get("end") is not None
    ]
    words.sort(key=lambda w: float(w["start"]))
    for i in range(len(words) - 1):
        words[i]["end"] = float(
            words[i + 1]["start"]
        )  # end = next start (karaoke-style)
    if words:
        words[-1]["end"] = float(words[-1]["end"]) + extend_last
    return words


@app.post("/analyze-song")
async def analyze_song(
    file: UploadFile = File(...),
    language: str = Form("vi"),
    model_size: str = Form("small"),
    vad: str = Form("true"),
    k: int = Form(0),  # ✅ NEW: semitone shift
):

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, file.filename or "input")
            with open(in_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            # 1) demucs tách vocal (giống y hệt /fit-song-to-user)
            demucs_out = os.path.join(tmpdir, "demucs")
            os.makedirs(demucs_out, exist_ok=True)

            cmd = [
                sys.executable,
                "-m",
                "demucs",
                "--two-stems",
                "vocals",
                "-o",
                demucs_out,
                in_path,
            ]
            p = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300,
            )
            if p.returncode != 0:
                return {
                    "ok": False,
                    "error": "Demucs failed",
                    "details": (p.stderr or "")[-2000:],
                }

            # resolve paths
            model_dirs = [
                d
                for d in os.listdir(demucs_out)
                if os.path.isdir(os.path.join(demucs_out, d))
            ]
            model_dir = model_dirs[0]
            song_dirs = [
                d
                for d in os.listdir(os.path.join(demucs_out, model_dir))
                if os.path.isdir(os.path.join(demucs_out, model_dir, d))
            ]
            song_dir = song_dirs[0]
            vocals_path, inst_path = resolve_demucs_paths(demucs_out)

            # ✅ clamp k
            k = clamp_int(k, -12, 12)

            # ✅ shift vocals by k semitones
            vocals_used = vocals_path
            if k != 0:
                vocals_shifted = os.path.join(tmpdir, f"vocals_{k}st.wav")
                transpose_wav(vocals_path, vocals_shifted, k)
                vocals_used = vocals_shifted

            # 2) pitch contour từ vocals (đường ngân) — dùng vocals_used
            times, f0_midi = extract_pitch_contour_midi(
                vocals_used, sr_target=16000, hop_length=160
            )
            valid = [v for v in f0_midi if v is not None]
            y_min = float(min(valid) - 2.0) if valid else 40.0
            y_max = float(max(valid) + 2.0) if valid else 80.0

            # 3) word timestamps từ vocals_used (whisper-timestamped)
            import whisper_timestamped as whisper_ts

            model = whisper_ts.load_model(model_size, device="cpu")
            r = whisper_ts.transcribe(
                model,
                vocals_used,
                language=language,
                vad=(vad.lower() == "true"),
            )

            words = []
            for seg in r.get("segments", []):
                for w in seg.get("words", []):
                    txt = (w.get("text") or "").strip()
                    if not txt:
                        continue
                    words.append(
                        {
                            "text": txt,
                            "start": float(w.get("start", 0.0)),
                            "end": float(w.get("end", 0.0)),
                            "confidence": (
                                float(w.get("confidence", 0.0))
                                if w.get("confidence") is not None
                                else None
                            ),
                        }
                    )
            words = postprocess_words(words, extend_last=0.6)

            return {
                "ok": True,
                "k": k,  # ✅ return k for frontend
                "times": times,
                "f0_midi": f0_midi,
                "y_min": y_min,
                "y_max": y_max,
                "words": words,
                "text": (r.get("text") or "").strip(),
            }

    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Demucs timeout"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
