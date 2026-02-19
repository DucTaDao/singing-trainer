import { useEffect, useMemo, useRef, useState } from "react";

export default function App() {
  // =====================
  // STEP 1: Record UUU
  // =====================
  const recRef = useRef(null);
  const chunksRef = useRef([]);

  const [recording, setRecording] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  // =====================
  // STEP 2: Upload song -> fit
  // =====================
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error1, setError1] = useState("");

  const [outURL, setOutURL] = useState(null);
  const [outName, setOutName] = useState("");
  // --- K (semitones) user chooses ---
  const [kUser, setKUser] = useState(0);
  const [kAuto, setKAuto] = useState(null);
  const [kInfo, setKInfo] = useState("");
  const [computingK, setComputingK] = useState(false);
  // =====================
  // STEP 3: Analyze song (pitch + words)
  // =====================
  const [analyzing, setAnalyzing] = useState(false);
  const [error2, setError2] = useState("");
  const [analysis, setAnalysis] = useState(null);

  const [songURL, setSongURL] = useState(null);
  const songAudioRef = useRef(null);

  const canvasRef = useRef(null);
  const rafRef = useRef(null);

  // smooth time for lyric UI
  const [now, setNow] = useState(0);

  // sync tweak (seconds)
  const [shift, setShift] = useState(0);

  // ✅ LOCKED y-range for the whole track (stable graph)
  const [yRange, setYRange] = useState({ min: 40, max: 80 });

  // ===== LIVE user pitch DOT =====
  const [micOn, setMicOn] = useState(false);
  const micStreamRef = useRef(null);
  const audioCtxRef = useRef(null);
  const analyserRef = useRef(null);
  const pcmRef = useRef(null);

  // latest user midi (smoothed)
  const userMidiRef = useRef(null);
  const userMidiSmoothRef = useRef(null);
  const userHistRef = useRef([]);
  const lastGoodAtRef = useRef(0);
  const dotMidiRef = useRef(null);

  // sweepline params
  const PRE = 2;
  const POST = 8;
  const WINDOW = PRE + POST;

  // canvas layout padding
  const PADL = 48;
  const PADR = 8;
  const PADT = 8;
  const PADB = 8;

  // =====================
  // Lyrics (paste + sync)
  // =====================
  const [lyricsText, setLyricsText] = useState("");
  const [lyricsSynced, setLyricsSynced] = useState(null);
  // lyricsSynced: { words:[{text,start,end,line}], lines:[{startIdx,endIdx,line}], score:{...} }
  const [syncMsg, setSyncMsg] = useState("");

  function getDeviceId() {
    let id = localStorage.getItem("device_id");
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem("device_id", id);
    }
    return id;
  }

  const fileLabel = useMemo(() => {
    if (!file) return "No file chosen";
    const mb = (file.size / (1024 * 1024)).toFixed(2);
    return `${file.name} (${mb} MB)`;
  }, [file]);

  function clearOutput() {
    if (outURL) URL.revokeObjectURL(outURL);
    setOutURL(null);
    setOutName("");
  }

  function clearAnalysis() {
    setAnalysis(null);
    setError2("");
    setShift(0);
    setLyricsSynced(null);
    setSyncMsg("");
  }

  // ===== helper: robust range for y-lock =====
  function robustRange(vals, pLo = 0.05, pHi = 0.95, pad = 2) {
    if (!vals || vals.length === 0) return { min: 40, max: 80 };
    const a = [...vals].sort((x, y) => x - y);
    const lo = a[Math.floor(pLo * (a.length - 1))];
    const hi = a[Math.floor(pHi * (a.length - 1))];

    let min = lo - pad;
    let max = hi + pad;

    if (max - min < 12) {
      const mid = (min + max) / 2;
      min = mid - 6;
      max = mid + 6;
    }
    return { min, max };
  }

  function midiToNoteName(m) {
    const names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    const mm = Math.round(m);
    const name = names[((mm % 12) + 12) % 12];
    const octave = Math.floor(mm / 12) - 1; // MIDI 60 = C4
    return `${name}${octave}`;
  }

  // ===== pitch detect (autocorr MVP) =====
  function hzToMidi(hz) {
    return 69 + 12 * Math.log2(hz / 440);
  }

  function detectPitchHz(timeDomain, sampleRate) {
    const N = timeDomain.length;

    // remove DC + RMS
    let mean = 0;
    for (let i = 0; i < N; i++) mean += timeDomain[i];
    mean /= N;

    let rms = 0;
    const x = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      x[i] = timeDomain[i] - mean;
      rms += x[i] * x[i];
    }
    rms = Math.sqrt(rms / N);
    if (rms < 0.01) return null;

    const minF = 80,
      maxF = 1000;
    const minLag = Math.floor(sampleRate / maxF);
    const maxLag = Math.floor(sampleRate / minF);

    let bestLag = -1;
    let bestVal = 0;

    for (let lag = minLag; lag <= maxLag; lag++) {
      let c = 0;
      for (let i = 0; i < N - lag; i++) c += x[i] * x[i + lag];
      if (c > bestVal) {
        bestVal = c;
        bestLag = lag;
      }
    }

    if (bestLag <= 0) return null;

    const hz = sampleRate / bestLag;
    if (!isFinite(hz) || hz < minF || hz > maxF) return null;
    return hz;
  }

  // =====================
  // Lyrics sync helpers
  // =====================
  function normWord(s) {
    return (s || "")
      .toLowerCase()
      .replace(/[’']/g, "'")
      .replace(/[^a-z0-9\u00C0-\u024F\u1E00-\u1EFF']/gi, " ")
      .trim();
  }

  function tokenizeLyricsWithLines(text) {
    const linesRaw = (text || "").split(/\r?\n/);
    const words = [];
    const lines = [];

    let idx = 0;
    for (let li = 0; li < linesRaw.length; li++) {
      const line = linesRaw[li].trim();
      if (!line) continue;

      const parts = line.split(/\s+/).filter(Boolean);
      const startIdx = idx;

      for (const p of parts) {
        const n = normWord(p);
        if (!n) continue;
        words.push({ raw: p, norm: n, line: li });
        idx++;
      }

      const endIdx = idx - 1;
      if (endIdx >= startIdx) lines.push({ startIdx, endIdx, line: li });
    }

    return { words, lines };
  }

  function tokenizeDetectedWords(words) {
    const out = [];
    for (const w of words || []) {
      const n = normWord(w.text);
      if (!n) continue;
      out.push({ raw: w.text, norm: n, start: w.start, end: w.end });
    }
    return out;
  }

  function alignWordsDP(A, B) {
    // mapAtoB[i] = matched index in B or null
    const n = A.length;
    const m = B.length;

    const dp = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
    const bt = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(null));

    for (let i = 1; i <= n; i++) {
      dp[i][0] = i;
      bt[i][0] = "UP";
    }
    for (let j = 1; j <= m; j++) {
      dp[0][j] = j;
      bt[0][j] = "LEFT";
    }

    for (let i = 1; i <= n; i++) {
      for (let j = 1; j <= m; j++) {
        const costSub = A[i - 1].norm === B[j - 1].norm ? 0 : 1;

        const vDiag = dp[i - 1][j - 1] + costSub;
        const vUp = dp[i - 1][j] + 1;
        const vLeft = dp[i][j - 1] + 1;

        let best = vDiag;
        let move = "DIAG";
        if (vUp < best) {
          best = vUp;
          move = "UP";
        }
        if (vLeft < best) {
          best = vLeft;
          move = "LEFT";
        }

        dp[i][j] = best;
        bt[i][j] = move;
      }
    }

    const mapAtoB = new Array(n).fill(null);
    let i = n,
      j = m;
    while (i > 0 || j > 0) {
      const move = bt[i][j];
      if (move === "DIAG") {
        mapAtoB[i - 1] = j - 1;
        i--;
        j--;
      } else if (move === "UP") {
        mapAtoB[i - 1] = null;
        i--;
      } else {
        j--;
      }
    }

    return mapAtoB;
  }

  function buildSyncedLyrics(pasted, detected, mapAtoB) {
    const n = pasted.words.length;
    const out = pasted.words.map((w) => ({
      text: w.raw,
      norm: w.norm,
      line: w.line,
      start: null,
      end: null,
    }));

    let exact = 0,
      matched = 0;
    for (let i = 0; i < n; i++) {
      const j = mapAtoB[i];
      if (j == null) continue;
      matched++;
      if (pasted.words[i].norm === detected[j]?.norm) exact++;
      out[i].start = detected[j].start;
      out[i].end = detected[j].end;
    }

    // interpolate missing between known neighbors
    const known = [];
    for (let i = 0; i < n; i++) if (out[i].start != null) known.push(i);

    if (known.length >= 2) {
      for (let k = 0; k < known.length - 1; k++) {
        const a = known[k];
        const b = known[k + 1];
        const ta = out[a].start;
        const tb = out[b].start;
        const gap = b - a;
        if (gap <= 1) continue;

        for (let i = a + 1; i < b; i++) {
          const r = (i - a) / gap;
          const t = ta + r * (tb - ta);
          out[i].start = t;
          out[i].end = t + 0.12;
        }
      }
    }

    const firstDet = detected[0]?.start ?? 0;
    const lastDet = detected[detected.length - 1]?.end ?? 0;

    // head fill
    let firstKnown = out.findIndex((w) => w.start != null);
    if (firstKnown === -1) firstKnown = 0;
    for (let i = 0; i < firstKnown; i++) {
      const t = firstDet + i * 0.12;
      out[i].start = t;
      out[i].end = t + 0.12;
    }

    // tail fill
    let lastKnown = -1;
    for (let i = out.length - 1; i >= 0; i--) {
      if (out[i].start != null) {
        lastKnown = i;
        break;
      }
    }
    for (let i = lastKnown + 1; i < out.length; i++) {
      const t = lastDet + (i - (lastKnown + 1)) * 0.12;
      out[i].start = t;
      out[i].end = t + 0.12;
    }

    return {
      words: out,
      lines: pasted.lines,
      score: {
        matched,
        exact,
        total: n,
        exactRate: n ? exact / n : 0,
        matchedRate: n ? matched / n : 0,
      },
    };
  }

  function syncLyrics() {
    setSyncMsg("");
    if (!analysis?.words?.length) {
      setSyncMsg("Analyze song first (need detected words). ");
      return;
    }
    if (!lyricsText.trim()) {
      setSyncMsg("Paste lyrics first.");
      return;
    }

    const pasted = tokenizeLyricsWithLines(lyricsText);
    const detected = tokenizeDetectedWords(analysis.words);

    if (pasted.words.length < 3) {
      setSyncMsg("Lyrics too short.");
      return;
    }
    if (detected.length < 3) {
      setSyncMsg("Detected words too short (bad transcription?).");
      return;
    }

    const mapAtoB = alignWordsDP(pasted.words, detected);
    const synced = buildSyncedLyrics(pasted, detected, mapAtoB);

    setLyricsSynced(synced);
    setSyncMsg(
      `Synced. exact ${(synced.score.exactRate * 100).toFixed(1)}% | matched ${(synced.score.matchedRate * 100).toFixed(1)}%`
    );
  }

  // ===== mic control =====
  async function startMic() {
    if (micOn) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          channelCount: 1,
        },
      });
      micStreamRef.current = stream;

      const AC = window.AudioContext || window.webkitAudioContext;
      const ctx = new AC();
      audioCtxRef.current = ctx;

      const src = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.0;
      analyserRef.current = analyser;

      src.connect(analyser);
      pcmRef.current = new Float32Array(analyser.fftSize);

      userMidiRef.current = null;
      userMidiSmoothRef.current = null;
      userHistRef.current = [];
      dotMidiRef.current = null;
      lastGoodAtRef.current = 0;

      setMicOn(true);
    } catch (e) {
      setError2(e?.message || "Mic permission denied");
    }
  }

  function stopMic() {
    setMicOn(false);
    try {
      analyserRef.current = null;
      pcmRef.current = null;

      if (audioCtxRef.current) {
        audioCtxRef.current.close().catch(() => { });
        audioCtxRef.current = null;
      }
      if (micStreamRef.current) {
        micStreamRef.current.getTracks().forEach((t) => t.stop());
        micStreamRef.current = null;
      }
    } catch { }
  }

  useEffect(() => {
    return () => stopMic();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // update user pitch continuously (no rerenders)
  useEffect(() => {
    let raf = 0;

    const HOLD_MS = 180;
    const HIST_N = 7;
    const MAX_STEP = 0.9;
    const EMA_FAST = 0.35;
    const EMA_SLOW = 0.12;

    function median(arr) {
      const a = [...arr].sort((x, y) => x - y);
      return a[Math.floor(a.length / 2)];
    }

    function clamp(v, lo, hi) {
      return Math.max(lo, Math.min(hi, v));
    }

    function tick() {
      const analyser = analyserRef.current;
      const pcm = pcmRef.current;
      const ctx = audioCtxRef.current;

      if (micOn && analyser && pcm && ctx) {
        analyser.getFloatTimeDomainData(pcm);
        const hz = detectPitchHz(pcm, ctx.sampleRate);

        const nowMs = performance.now();

        if (hz) {
          const m = hzToMidi(hz);

          const hist = userHistRef.current;
          hist.push(m);
          if (hist.length > HIST_N) hist.shift();

          const mMed = median(hist);

          const prevFast = userMidiRef.current;
          const fast = prevFast == null ? mMed : prevFast + EMA_FAST * (mMed - prevFast);
          userMidiRef.current = fast;

          const prevSlow = userMidiSmoothRef.current;
          let smooth = prevSlow == null ? fast : prevSlow + EMA_SLOW * (fast - prevSlow);

          if (prevSlow != null) {
            const delta = smooth - prevSlow;
            smooth = prevSlow + clamp(delta, -MAX_STEP, MAX_STEP);
          }

          userMidiSmoothRef.current = smooth;
          dotMidiRef.current = smooth;
          lastGoodAtRef.current = nowMs;
        } else {
          if (nowMs - lastGoodAtRef.current > HOLD_MS) {
            userMidiRef.current = null;
            userMidiSmoothRef.current = null;
            dotMidiRef.current = null;
            userHistRef.current = [];
          }
        }
      }

      raf = requestAnimationFrame(tick);
    }

    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [micOn]);

  useEffect(() => {
    if (!file) {
      if (songURL) URL.revokeObjectURL(songURL);
      setSongURL(null);
      clearAnalysis();
      clearOutput();
      return;
    }
    const url = URL.createObjectURL(file);
    setSongURL(url);
    return () => URL.revokeObjectURL(url);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file]);

  // update now
  useEffect(() => {
    const id = setInterval(() => {
      const a = songAudioRef.current;
      if (a) setNow(a.currentTime || 0);
    }, 50);
    return () => clearInterval(id);
  }, []);

  // =====================
  // Record UUU
  // =====================
  async function start() {
    setError("");
    setResult(null);
    chunksRef.current = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      const options = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? { mimeType: "audio/webm;codecs=opus" }
        : {};

      const rec = new MediaRecorder(stream, options);
      recRef.current = rec;

      rec.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      rec.onstop = async () => {
        try {
          const blob = new Blob(chunksRef.current, {
            type: rec.mimeType || "audio/webm",
          });

          const fd = new FormData();
          fd.append("file", blob, "uuu.webm");
          fd.append("device_id", getDeviceId());

          const res = await fetch("http://localhost:8000/analyze-uuu", {
            method: "POST",
            body: fd,
          });

          const data = await res.json();
          if (!data.ok) setError(data.error || "Analyze failed");
          else setResult(data);
        } catch (e) {
          setError(e?.message || "Analyze failed (backend on?)");
        } finally {
          stream.getTracks().forEach((t) => t.stop());
        }
      };

      rec.start();
      setRecording(true);
    } catch (e) {
      setError(e?.message || "Mic permission denied");
    }
  }

  function stop() {
    recRef.current?.stop();
    setRecording(false);
  }
  async function computeK() {
    setError1("");
    setKInfo("");
    setKAuto(null);

    if (!result) {
      setError1("Record UUU first so we know your range");
      return;
    }
    if (!file) {
      setError1("Choose a song file first");
      return;
    }

    setComputingK(true);
    try {
      const fd = new FormData();
      fd.append("file", file, file.name);
      fd.append("device_id", getDeviceId());

      // ✅ backend endpoint mới: chỉ tính K + trả info (chưa transpose)
      const res = await fetch("http://localhost:8000/compute-k", {
        method: "POST",
        body: fd,
      });

      const j = await res.json();
      if (!j.ok) throw new Error(j.error || "Compute K failed");

      setKAuto(j.k);
      setKUser(j.k); // ✅ auto-fill slider
      setKInfo(
        `Auto K = ${j.k} | Song: ${j.song_note_min}→${j.song_note_max} | You: ${j.user_note_min}→${j.user_note_max}`
      );
    } catch (e) {
      setError1(e?.message || "Compute K failed");
    } finally {
      setComputingK(false);
    }
  }

  // =====================
  // Fit song -> audio response
  // =====================
  async function uploadAndProcess() {
    setError1("");
    clearOutput();

    if (!result) {
      setError1("Record UUU first so we know your range");
      return;
    }
    if (!file) {
      setError1("Choose a song file first");
      return;
    }

    setUploading(true);
    try {
      const fd = new FormData();
      fd.append("file", file, file.name);
      fd.append("device_id", getDeviceId());
      fd.append("k", String(kUser));

      const res = await fetch("http://localhost:8000/fit-song-to-user", {
        method: "POST",
        body: fd,
      });

      if (!res.ok) {
        const txt = await res.text().catch(() => "");
        throw new Error(txt || "Process failed");
      }

      const ct = res.headers.get("content-type") || "";
      if (!ct.includes("audio")) {
        const txt = await res.text().catch(() => "");
        throw new Error("Not audio response: " + txt);
      }

      const dispo = res.headers.get("content-disposition") || "";
      const match = dispo.match(/filename=\"?([^\";]+)\"?/i);
      const filename = match?.[1] || "song_fitted.wav";

      const outBlob = await res.blob();
      if (outBlob.size < 2000) throw new Error("Output too small: " + outBlob.size + " bytes");

      const url = URL.createObjectURL(outBlob);
      setOutURL(url);
      setOutName(filename);
    } catch (e) {
      setError1(e?.message || "Upload failed (backend on?)");
    } finally {
      setUploading(false);
    }
  }

  function downloadOutput() {
    if (!outURL) return;
    const a = document.createElement("a");
    a.href = outURL;
    a.download = outName || "song_fitted.wav";
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  // =====================
  // Analyze song -> JSON (pitch + word timing)
  // =====================
  async function analyzeSong() {
    setError2("");
    clearAnalysis();

    if (!file) {
      setError2("Choose a song file first");
      return;
    }

    setAnalyzing(true);
    try {
      const fd = new FormData();
      fd.append("file", file, file.name);
      fd.append("language", "vi");
      fd.append("model_size", "base");
      fd.append("vad", "true");

      const res = await fetch("http://localhost:8000/analyze-song", {
        method: "POST",
        body: fd,
      });

      const ct = res.headers.get("content-type") || "";
      if (!res.ok) {
        const txt = await res.text().catch(() => "");
        throw new Error(txt || "Analyze failed");
      }
      if (!ct.includes("application/json")) {
        const txt = await res.text().catch(() => "");
        throw new Error("Expected JSON, got: " + ct + " " + txt);
      }

      const j = await res.json();
      if (!j.ok) throw new Error(j.error || "Analyze failed");

      const vals = (j.f0_midi || []).filter((v) => v != null && isFinite(v));
      setYRange(robustRange(vals, 0.05, 0.95, 2));

      setAnalysis(j);
    } catch (e) {
      setError2(e?.message || "Analyze failed");
    } finally {
      setAnalyzing(false);
    }
  }

  // ===== words shifted (detected words) =====
  const wordsShifted = useMemo(() => {
    if (!analysis?.words) return [];
    return analysis.words.map((w) => ({
      ...w,
      start: w.start + shift,
      end: w.end + shift,
    }));
  }, [analysis, shift]);

  // ===== choose words for UI: prefer synced lyrics =====
  const wordsForUI = useMemo(() => {
    if (lyricsSynced?.words?.length) return lyricsSynced.words;
    return wordsShifted;
  }, [lyricsSynced, wordsShifted]);

  const wordsForUIShifted = useMemo(() => {
    return (wordsForUI || []).map((w) => ({
      ...w,
      start: (w.start ?? 0) + shift,
      end: (w.end ?? 0) + shift,
    }));
  }, [wordsForUI, shift]);

  // window times
  const t0 = Math.max(0, now - PRE);
  const t1 = t0 + WINDOW;

  const visibleWords = useMemo(() => {
    return wordsForUIShifted.filter((w) => (w.end ?? 0) >= t0 && (w.start ?? 0) <= t1);
  }, [wordsForUIShifted, t0, t1]);

  const activeVisibleIdx = useMemo(() => {
    for (let i = 0; i < visibleWords.length; i++) {
      if (now >= visibleWords[i].start && now < visibleWords[i].end) return i;
    }
    return -1;
  }, [visibleWords, now]);

  // =====================
  // Canvas draw sweepline
  // =====================
  useEffect(() => {
    function draw() {
      const cvs = canvasRef.current;
      const audio = songAudioRef.current;
      if (!cvs || !audio || !analysis) {
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      const ctx = cvs.getContext("2d");
      const W = cvs.width;
      const H = cvs.height;

      const times = analysis.times || [];
      const f0 = analysis.f0_midi || [];

      const t = audio.currentTime || 0;
      const win0 = Math.max(0, t - PRE);
      const win1 = win0 + WINDOW;

      const plotW = Math.max(1, W - PADL - PADR);
      function xOf(tt) {
        return PADL + ((tt - win0) / (win1 - win0)) * plotW;
      }

      let yMin = yRange.min;
      let yMax = yRange.max;
      if (yMax - yMin < 1e-6) yMax = yMin + 1;

      const plotH = Math.max(1, H - PADT - PADB);
      function yOf(mm) {
        return PADT + (1 - (mm - yMin) / (yMax - yMin)) * plotH;
      }

      ctx.clearRect(0, 0, W, H);

      // grid
      ctx.save();
      const mStart = Math.ceil(yMin);
      const mEnd = Math.floor(yMax);

      for (let m = mStart; m <= mEnd; m++) {
        const y = yOf(m);
        const pc = ((m % 12) + 12) % 12;
        const isC = pc === 0;

        ctx.beginPath();
        ctx.moveTo(PADL, y);
        ctx.lineTo(W - PADR, y);
        ctx.lineWidth = isC ? 1.2 : 0.6;
        ctx.stroke();

        const shouldLabel = isC || pc === 9;
        if (shouldLabel) {
          ctx.font = "12px system-ui";
          ctx.fillText(midiToNoteName(m), 6, y - 3);
        }
      }
      ctx.restore();

      // reference curve
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < times.length; i++) {
        const tt = times[i];
        if (tt < win0 || tt > win1) continue;
        const mm = f0[i];
        if (mm == null) {
          started = false;
          continue;
        }
        const x = xOf(tt);
        const y = yOf(mm);
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.lineWidth = 2;
      ctx.stroke();

      // highlight current word segment (use wordsForUIShifted for consistency)
      let ws = null,
        we = null;
      for (let i = 0; i < wordsForUIShifted.length; i++) {
        const w = wordsForUIShifted[i];
        if (t >= w.start && t < w.end) {
          ws = w.start;
          we = w.end;
          break;
        }
      }

      if (ws != null && we != null) {
        ctx.beginPath();
        let s2 = false;
        for (let i = 0; i < times.length; i++) {
          const tt = times[i];
          if (tt < win0 || tt > win1) continue;
          if (tt < ws || tt > we) continue;
          const mm = f0[i];
          if (mm == null) {
            s2 = false;
            continue;
          }
          const x = xOf(tt);
          const y = yOf(mm);
          if (!s2) {
            ctx.moveTo(x, y);
            s2 = true;
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.lineWidth = 5;
        ctx.stroke();
      }

      // playhead line
      const playX = PADL + (PRE / WINDOW) * plotW;
      ctx.beginPath();
      ctx.moveTo(playX, PADT);
      ctx.lineTo(playX, H - PADB);
      ctx.lineWidth = 1;
      ctx.stroke();

      // user dot
      if (micOn) {
        const mUser = dotMidiRef.current;
        if (mUser != null) {
          const dotX = playX;
          const dotY = yOf(mUser);
          ctx.beginPath();
          ctx.arc(dotX, dotY, 6, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      rafRef.current = requestAnimationFrame(draw);
    }

    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analysis, shift, yRange, micOn, wordsForUIShifted]);

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#f5f5f5",
        padding: 24,
        fontFamily: "system-ui",
        color: "black",
      }}
    >
      <h1 style={{ marginBottom: 24 }}>🎤 Pitch Trainer</h1>

      {/* STEP 1 */}
      <div
        style={{
          background: "white",
          borderRadius: 12,
          padding: 20,
          marginBottom: 24,
          boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
        }}
      >
        <h2 style={{ marginBottom: 12 }}>① Record “UUU” (Detect your range)</h2>

        {!recording ? <button onClick={start}>Start</button> : <button onClick={stop}>Stop</button>}

        {error && <p style={{ color: "red", marginTop: 8 }}>{error}</p>}

        {result && (
          <div style={{ marginTop: 12 }}>
            <p>
              <b>Note range:</b> {result.note_min} → {result.note_max}
            </p>
            <p>
              <b>Frequency:</b> {result.f0_min_hz} → {result.f0_max_hz} Hz
            </p>
          </div>
        )}
      </div>

      {/* STEP 2 */}
      <div
        style={{
          background: "white",
          borderRadius: 12,
          padding: 20,
          marginBottom: 24,
          boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
          opacity: result ? 1 : 0.6,
        }}
      >
        <h2 style={{ marginBottom: 12 }}>② Upload Song → Get Fitted Version</h2>

        {!result && <p style={{ fontSize: 14, color: "#666" }}>⛔ Record “UUU” first to detect your vocal range</p>}

        <div style={{ marginBottom: 10 }}>
          <input
            type="file"
            accept="audio/*"
            disabled={!result}
            onChange={(e) => {
              setError1("");
              clearOutput();
              clearAnalysis();
              setKUser(0);
              setKAuto(null);
              setKInfo("");
              setComputingK(false);
              setFile(e.target.files?.[0] || null);
            }}
          />
          <div style={{ marginTop: 6, fontSize: 14 }}>{fileLabel}</div>
        </div>
        <div style={{ marginTop: 10, padding: 10, border: "1px solid #eee", borderRadius: 10 }}>
          <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
            <button onClick={computeK} disabled={computingK || !file || !result}>
              {computingK ? "Computing K..." : "Compute K (Auto)"}
            </button>

            <span style={{ fontSize: 13, color: "#666" }}>
              {kInfo || "Click Compute K to get recommendation, then adjust K."}
            </span>
          </div>

          <div style={{ marginTop: 10 }}>
            <label>
              K (semitones): <b>{kUser}</b>{" "}
              <input
                type="range"
                min={-12}
                max={12}
                step={1}
                value={kUser}
                onChange={(e) => setKUser(parseInt(e.target.value, 10))}
                style={{ width: 280, verticalAlign: "middle" }}
              />
            </label>

            <button
              style={{ marginLeft: 10 }}
              onClick={() => setKUser(0)}
              disabled={!file || !result}
            >
              Reset
            </button>

            {kAuto != null && (
              <button
                style={{ marginLeft: 10 }}
                onClick={() => setKUser(kAuto)}
                disabled={!file || !result}
              >
                Use Auto ({kAuto})
              </button>
            )}
          </div>
        </div>

        <button onClick={uploadAndProcess} disabled={uploading || !file || !result}>
          {uploading ? "Processing..." : "Upload & Process"}
        </button>

        {error1 && <p style={{ color: "red", marginTop: 8 }}>{error1}</p>}

        {outURL && (
          <div style={{ marginTop: 16 }}>
            <p>
              <b>Output:</b> {outName}
            </p>
            <audio controls src={outURL} style={{ width: "100%" }} />
            <div style={{ marginTop: 8 }}>
              <button onClick={downloadOutput}>Download</button>
            </div>
          </div>
        )}
      </div>

      {/* STEP 3 */}
      <div
        style={{
          background: "white",
          borderRadius: 12,
          padding: 20,
          boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
          opacity: file ? 1 : 0.6,
        }}
      >
        <h2 style={{ marginBottom: 12 }}>③ Analyze Song (Sweepline: pitch + lyrics)</h2>

        {!file && <p style={{ fontSize: 14, color: "#666" }}>⛔ Choose a song file first</p>}

        <button onClick={analyzeSong} disabled={analyzing || !file}>
          {analyzing ? "Analyzing..." : "Analyze (JSON)"}
        </button>

        {error2 && <p style={{ color: "red", marginTop: 8, whiteSpace: "pre-wrap" }}>{error2}</p>}

        {songURL && (
          <div style={{ marginTop: 12 }}>
            <p style={{ margin: 0, fontSize: 14, color: "#666" }}>Play song — canvas + lyrics will move (sweepline).</p>
            <audio ref={songAudioRef} controls src={songURL} style={{ width: "100%" }} />
          </div>
        )}

        {analysis && (
          <>
            {/* mic controls */}
            <div style={{ marginTop: 10, display: "flex", gap: 10, alignItems: "center" }}>
              {!micOn ? <button onClick={startMic}>Start Mic (User Dot)</button> : <button onClick={stopMic}>Stop Mic</button>}
              <span style={{ fontSize: 13, color: "#666" }}>Hát lên: chấm sẽ nhảy theo cao độ tại vạch dọc.</span>
            </div>

            {/* lyric shift */}
            <div style={{ marginTop: 12 }}>
              <label>
                Lyric sync shift: <b>{shift.toFixed(2)}s</b>{" "}
                <input
                  type="range"
                  min={-1}
                  max={1}
                  step={0.05}
                  value={shift}
                  onChange={(e) => setShift(parseFloat(e.target.value))}
                />
              </label>
            </div>

            {/* paste lyrics + sync */}
            <div style={{ marginTop: 14 }}>
              <div style={{ fontSize: 14, color: "#666", marginBottom: 6 }}>
                Paste lyrics here (official). Then click Sync to align with detected timing.
              </div>

              <textarea
                value={lyricsText}
                onChange={(e) => setLyricsText(e.target.value)}
                placeholder={"Paste lyrics...\n(one line per lyric line is best)"}
                style={{ width: "100%", minHeight: 140, padding: 10, border: "1px solid #ddd", borderRadius: 8 }}
              />

              <div style={{ marginTop: 8, display: "flex", gap: 10, alignItems: "center" }}>
                <button onClick={syncLyrics}>Sync Lyrics</button>
                <button
                  onClick={() => {
                    setLyricsSynced(null);
                    setSyncMsg("Using detected lyrics again.");
                  }}
                >
                  Use Detected
                </button>
                <span style={{ fontSize: 13, color: "#666" }}>{syncMsg}</span>
              </div>
            </div>

            {/* canvas */}
            <div style={{ marginTop: 12 }}>
              <canvas ref={canvasRef} width={900} height={260} style={{ border: "1px solid #ddd", width: "100%" }} />
            </div>

            {/* lyrics window */}
            {visibleWords.length > 0 && (
              <div style={{ marginTop: 12, lineHeight: 1.9, fontSize: 18 }}>
                {visibleWords.map((w, i) => (
                  <span key={`${w.start}-${i}`} style={{ fontWeight: i === activeVisibleIdx ? 900 : 400, marginRight: 8 }}>
                    {w.text}
                  </span>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
