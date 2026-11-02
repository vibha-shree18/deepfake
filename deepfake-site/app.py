
import datetime
import hashlib
import os
import tempfile
import time
import uuid

import cv2
import librosa
import numpy as np
import timm
import torch
from flask import Flask, jsonify, request
from torchvision import transforms
from transformers import Wav2Vec2ForSequenceClassification

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")
device = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# What the website shows about each model (numbers are from our own test runs)
MODEL_INFO = {
    "image": dict(name="ViT-Base (patch 16, 224 px)", dataset="140k Real and Fake Faces",
                  accuracy="86.7%", test_size="20,000 unseen images",
                  limit="Trained on StyleGAN faces. Fakes from newer generators may be judged Real."),
    "video": dict(name="ViT-Base on face frames", dataset="FaceForensics++ subset (400 videos)",
                  accuracy="76.7%", test_size="60 unseen videos",
                  limit="Small training set. Videos from modern AI generators are often judged Real."),
    "audio": dict(name="Wav2Vec2-Base", dataset="ASVspoof 2019 (LA)",
                  accuracy="98.2%", test_size="2,000 unseen clips",
                  limit="Trained on studio recordings. Real voices recorded on a phone or browser microphone may be judged Fake."),
}


# ---------- load the three trained models ----------
def load_vit(file_name):
    m = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=2)
    m.load_state_dict(torch.load(os.path.join(MODELS, file_name), map_location=device))
    return m.to(device).eval()


def load_audio():
    m = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=2)
    m.load_state_dict(torch.load(os.path.join(MODELS, "best_wav2vec.pth"), map_location=device))
    return m.to(device).eval()


print("Loading models on", device, "...")
image_model = load_vit("best_vit_v2.pth")
video_model = load_vit("best_vit_video.pth")
audio_model = load_audio()
print("All models ready.")

# ---------- preprocessing (same as training) ----------
to_tensor_norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
face_det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def crop_face(bgr, margin):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = face_det.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return bgr, False
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    m = int(margin * max(w, h))
    return bgr[max(0, y - m):y + h + m, max(0, x - m):x + w + m], True


def to_tensor(bgr):
    rgb = cv2.cvtColor(cv2.resize(bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    return to_tensor_norm(rgb)


@torch.no_grad()
def fake_probs_images(model, tensors):
    x = torch.stack(tensors).to(device)
    return model(x).softmax(1)[:, 0].cpu().tolist()   # class 0 = fake


# ---------- one function per file type: returns (fake_probability, note, details) ----------
def predict_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Could not read this image.")
    p = fake_probs_images(image_model, [to_tensor(img)])[0]
    return p, "Checked one image", {}


def predict_video(path):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError("Could not read this video.")
    tensors, faces_found = [], 0
    for idx in np.linspace(0, max(total - 2, 0), 10).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        face, found = crop_face(frame, 0.2)
        faces_found += int(found)
        tensors.append(to_tensor(face))
    cap.release()
    if not tensors:
        raise ValueError("No frames could be read from this video.")
    scores = fake_probs_images(video_model, tensors)
    p = float(np.mean(scores))
    details = {"frames": len(tensors), "faces_found": faces_found,
               "frame_scores": [round(s * 100, 1) for s in scores]}
    return p, f"Averaged over {len(tensors)} frames", details


@torch.no_grad()
def predict_audio(path):
    wav, _ = librosa.load(path, sr=16000)
    if len(wav) < 16000:
        raise ValueError("Audio is too short. Use at least 1 second.")
    size = 16000 * 4
    chunks = []
    for start in range(0, len(wav), size)[:5]:
        c = wav[start:start + size]
        if len(c) < 16000 and chunks:
            continue
        c = np.pad(c, (0, size - len(c)))
        c = (c - c.mean()) / (c.std() + 1e-7)
        chunks.append(torch.tensor(c, dtype=torch.float32))
    x = torch.stack(chunks).to(device)
    probs = audio_model(x).logits.softmax(1)[:, 0].cpu().tolist()
    p = float(np.mean(probs))
    return p, f"Averaged over {len(chunks)} audio clip(s)", {"clip_scores": [round(s * 100, 1) for s in probs]}


# ---------- website ----------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024


@app.route("/")
def home():
    return PAGE


@app.route("/predict", methods=["POST"])
def predict():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="Choose a file first."), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext in IMAGE_EXT:
        kind, fn = "image", predict_image
    elif ext in VIDEO_EXT:
        kind, fn = "video", predict_video
    elif ext in AUDIO_EXT:
        kind, fn = "audio", predict_audio
    else:
        return jsonify(error="This file type is not supported. Use an image, video or audio file."), 400

    t0 = time.time()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as t:
        f.save(t)
        path = t.name
    try:
        sha = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                sha.update(chunk)
        size = os.path.getsize(path)
        p, note, details = fn(path)
    except Exception as e:
        return jsonify(error=str(e)), 500
    finally:
        os.remove(path)

    d = abs(p - 0.5)
    confidence = "HIGH" if d >= 0.3 else "MEDIUM" if d >= 0.15 else "LOW"
    label = "Likely AI-Generated" if p >= 0.6 else "Likely Real" if p <= 0.4 else "Inconclusive"
    return jsonify(id=str(uuid.uuid4()), kind=kind, filename=f.filename, size=size, sha256=sha.hexdigest(),
                   analyzed_at=datetime.datetime.now().isoformat(timespec="seconds"),
                   elapsed=round(time.time() - t0, 1), ai_probability=round(p * 100, 1),
                   real_probability=round((1 - p) * 100, 1), confidence=confidence, label=label,
                   note=note, model=MODEL_INFO[kind], details=details)


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Deepfake Detector</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500&family=Manrope:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap">
<style>
:root{--bg:#070B14;--card:#0D1322;--line:#1C2438;--text:#EEF1F8;--mut:#8C97B2;--acc:#6C5CE7;--acc2:#8B5CF6;--cy:#06B6D4;--good:#19C48B;--bad:#F0476B;--warn:#F5A623;--track:#1A2136;--navbg:rgba(7,11,20,.82);--tile:rgba(0,0,0,.25)}
[data-theme=light]{--bg:#F4F6FB;--card:#FFFFFF;--line:#DCE1EE;--text:#10162A;--mut:#566180;--good:#0E9F6E;--bad:#D6264F;--warn:#B7791F;--track:#E6EAF4;--navbg:rgba(244,246,251,.85);--tile:rgba(16,22,42,.04)}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--text);font-family:Manrope,'Segoe UI',system-ui,sans-serif;line-height:1.55;-webkit-font-smoothing:antialiased}
h1,h2,h3,h4{font-family:'Space Grotesk',Manrope,sans-serif;margin:0;letter-spacing:-.02em}
p{margin:0}.mut{color:var(--mut)}.mono{font-family:'IBM Plex Mono',monospace}
.wrap{max-width:1216px;margin:0 auto;padding:0 24px}.narrow{max-width:860px;margin:0 auto;padding:0 24px}
button{font-family:inherit}
.nav{position:sticky;top:0;z-index:20;background:var(--navbg);backdrop-filter:blur(12px);border-bottom:1px solid var(--line)}
.nav .wrap{display:flex;align-items:center;justify-content:space-between;height:68px;gap:12px}
.logo{display:flex;align-items:center;gap:12px;font:700 20px 'Space Grotesk',sans-serif;cursor:pointer;border:0;background:none;color:var(--text)}
.lg{width:38px;height:38px;border-radius:11px;background:linear-gradient(135deg,#6C5CE7,#C084FC);display:grid;place-items:center;color:#fff;box-shadow:0 6px 18px rgba(108,92,231,.4)}
.links{display:flex;gap:4px}.links button{border:0;background:none;color:var(--mut);font:600 15px Manrope;padding:10px 16px;border-radius:10px;cursor:pointer}
.links button.on{background:rgba(108,92,231,.16);color:#A99BFF}[data-theme=light] .links button.on{color:#4B3FD0}
.icb{width:42px;height:42px;border-radius:11px;border:1px solid var(--line);background:none;color:var(--text);display:grid;place-items:center;cursor:pointer}
.btn{border:0;border-radius:11px;padding:14px 24px;font:600 15px Manrope;cursor:pointer;display:inline-flex;align-items:center;gap:10px;color:var(--text)}
.btn-p{background:var(--acc);color:#fff;box-shadow:0 8px 26px rgba(108,92,231,.38)}.btn-g{background:transparent;border:1px solid var(--line)}.btn-w{background:#fff;color:#4338CA}
.btn:focus-visible,.icb:focus-visible,.links button:focus-visible,.tab:focus-visible{outline:3px solid #A99BFF;outline-offset:2px}
.view{display:none}.view.on{display:block}
.hero{text-align:center;padding:96px 24px 72px;background:radial-gradient(60% 55% at 50% 0%,rgba(108,92,231,.3),transparent 70%),linear-gradient(var(--line) 1px,transparent 1px) 0 0/48px 48px,linear-gradient(90deg,var(--line) 1px,transparent 1px) 0 0/48px 48px;background-blend-mode:normal}
.badge{display:inline-flex;gap:8px;align-items:center;border:1px solid var(--line);background:var(--card);border-radius:99px;padding:8px 16px;font:500 12px 'IBM Plex Mono',monospace;letter-spacing:.12em;color:var(--mut)}
.hero h1{font-size:clamp(40px,7vw,76px);line-height:1.02;margin:28px auto 22px;max-width:900px;font-weight:700}
.grad{background:linear-gradient(90deg,#7C6CF5,#A855F7 55%,#06B6D4);-webkit-background-clip:text;background-clip:text;color:transparent}
.hero p.lead{font-size:19px;color:var(--mut);max-width:640px;margin:0 auto 34px}
.sec{padding:56px 0}.eyebrow{font:500 12px 'IBM Plex Mono',monospace;letter-spacing:.2em;color:#7C8CC8;text-align:center}
.sec h2.t{font-size:38px;text-align:center;margin:8px 0 36px}
.card{background:var(--card);border:1px solid var(--line);border-radius:24px;padding:28px}
.g3{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}.g4{display:grid;grid-template-columns:repeat(4,1fr);gap:20px}.g2{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.ico{width:56px;height:56px;border-radius:15px;display:grid;place-items:center;color:#fff;margin-bottom:34px}
.i1{background:linear-gradient(135deg,#6C5CE7,#A78BFA)}.i2{background:linear-gradient(135deg,#C026D3,#E879F9)}.i3{background:linear-gradient(135deg,#0EA5E9,#22D3EE)}
.card h3{font-size:22px;margin-bottom:8px}.card .tag{display:inline-block;margin-top:14px;font:500 12px 'IBM Plex Mono',monospace;color:var(--mut);border:1px solid var(--line);border-radius:99px;padding:4px 12px}
.step{position:relative;padding-top:34px}.step .n{position:absolute;top:-12px;left:24px;background:var(--acc);color:#fff;border-radius:99px;font:700 12px Manrope;padding:2px 10px}
.step .si{width:48px;height:48px;border-radius:12px;background:rgba(108,92,231,.14);color:#8E7FFF;display:grid;place-items:center;margin-bottom:18px}
.cta{background:linear-gradient(135deg,#5B4FE0,#9333EA 50%,#06B6D4);border-radius:32px;padding:72px 24px;text-align:center;color:#fff;margin:24px 0 80px}
.cta h2{font-size:42px;margin-bottom:12px}.cta p{opacity:.9;max-width:520px;margin:0 auto 28px}
.tabs{display:inline-flex;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:6px;gap:4px}
.tab{padding:11px 30px;border-radius:10px;border:0;background:none;color:var(--mut);font:600 15px Manrope;cursor:pointer;display:flex;gap:8px;align-items:center}.tab.on{background:var(--bg);color:var(--text)}
.drop{border:2px dashed #5B4FD6;border-radius:28px;background:rgba(108,92,231,.06);padding:60px 24px;text-align:center;margin-top:28px}.drop.over{background:rgba(108,92,231,.16)}
.drop .ico{margin:0 auto 22px;width:64px;height:64px;margin-bottom:22px}.chips{display:flex;gap:10px;justify-content:center;margin-top:20px;flex-wrap:wrap}
.chip{font:500 12px 'IBM Plex Mono',monospace;border:1px solid var(--line);border-radius:99px;padding:5px 14px;color:var(--mut)}
.prev{max-width:100%;max-height:170px;border-radius:12px;margin:0 auto 14px;display:block}audio.prev{width:320px}
.fine{text-align:center;font-size:13px;color:var(--mut);margin-top:22px}.err{display:none;margin-top:18px;border:1px solid var(--bad);color:var(--bad);border-radius:14px;padding:14px 18px;text-align:center}
.spin{width:112px;height:112px;margin:0 auto 28px;position:relative;color:#8E7FFF;display:grid;place-items:center}.spin svg.r{position:absolute;inset:0;animation:rot 1.1s linear infinite}
@keyframes rot{to{transform:rotate(360deg)}}
.st{display:flex;gap:16px;align-items:center;padding:16px 18px;border:1px solid var(--line);border-radius:16px;background:var(--card);margin-bottom:12px;color:var(--mut)}
.st .b{width:40px;height:40px;border-radius:11px;background:var(--track);display:grid;place-items:center}
.st.done{border-color:rgba(25,196,139,.4);background:rgba(25,196,139,.06);color:var(--text)}.st.done .b{background:var(--good);color:#04130C}
.st.run{border-color:var(--acc);color:var(--text)}.st.run .b{background:var(--acc);color:#fff}.st.run .b svg{animation:rot 1s linear infinite}
.vc{display:grid;grid-template-columns:1.1fr .9fr;gap:24px;align-items:center;border-radius:28px;padding:38px;border:1px solid var(--line)}
.vc.good{background:rgba(25,196,139,.07);border-color:rgba(25,196,139,.35)}.vc.bad{background:rgba(240,71,107,.07);border-color:rgba(240,71,107,.35)}.vc.warn{background:rgba(245,166,35,.07);border-color:rgba(245,166,35,.35)}
.pill{display:inline-flex;gap:8px;align-items:center;border-radius:99px;padding:8px 16px;font:600 14px Manrope;color:#04130C}.pill.good{background:var(--good)}.pill.bad{background:var(--bad);color:#fff}.pill.warn{background:var(--warn)}
.vt{font-size:46px;line-height:1.05;margin:18px 0 12px}.vt.good{color:var(--good)}.vt.bad{color:var(--bad)}.vt.warn{color:var(--warn)}
.tiles{display:flex;gap:12px;margin-top:22px;flex-wrap:wrap}.tile{border:1px solid var(--line);background:var(--tile);border-radius:14px;padding:12px 22px;text-align:center;min-width:100px}.tile small{display:block;color:var(--mut);font-size:13px}.tile b{font:700 20px 'Space Grotesk',sans-serif}
.ring{width:100%;max-width:260px;display:block;margin:0 auto}
.sig{border:1px solid var(--line);border-radius:16px;padding:16px 20px;background:var(--card);margin-top:12px}.sig h4{font:600 15px Manrope;margin-bottom:4px}.sig p{color:var(--mut);font-size:15px}
.bars{display:flex;align-items:flex-end;gap:6px;height:80px;margin-top:12px}.bars i{flex:1;background:linear-gradient(var(--acc2),var(--acc));border-radius:5px 5px 0 0;min-height:3px}
.hdr{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:22px;flex-wrap:wrap}
.warnbox{border:1px solid rgba(245,166,35,.45);background:rgba(245,166,35,.06);border-radius:16px;padding:18px 22px;display:flex;gap:14px;color:var(--mut);font-size:15px;margin-top:20px}.warnbox svg{color:var(--warn);flex:none;margin-top:2px}
.hash{font:500 12px 'IBM Plex Mono',monospace;color:var(--mut);margin-top:18px;word-break:break-all;display:flex;gap:10px;align-items:center}
.kv{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:18px}.kv div{border:1px solid var(--line);border-radius:14px;padding:12px 16px;background:var(--card)}.kv small{display:block;font:500 11px 'IBM Plex Mono',monospace;letter-spacing:.1em;color:var(--mut);text-transform:uppercase}.kv span{font-weight:600;word-break:break-all;font-size:14px}
details{border-bottom:1px solid var(--line);padding:18px 0}summary{cursor:pointer;font-weight:600;display:flex;justify-content:space-between;list-style:none}summary::-webkit-details-marker{display:none}details p{color:var(--mut);margin-top:10px;font-size:15px}
table{width:100%;border-collapse:collapse;font-size:15px}th,td{text-align:left;padding:12px 14px;border-bottom:1px solid var(--line)}th{font:500 12px 'IBM Plex Mono',monospace;letter-spacing:.1em;color:var(--mut);text-transform:uppercase}
.hi{display:flex;gap:14px;align-items:center;justify-content:space-between;border:1px solid var(--line);background:var(--card);border-radius:16px;padding:14px 18px;margin-top:12px;flex-wrap:wrap}
footer{border-top:1px solid var(--line);padding:40px 0;color:var(--mut);font-size:14px}footer .wrap{display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap}
@media(max-width:860px){.g3,.g4,.g2,.vc,.kv{grid-template-columns:1fr}.links button{padding:8px 10px;font-size:14px}.vt{font-size:36px}.cta h2{font-size:30px}.logo span{display:none}}
@media print{:root,[data-theme=dark]{--bg:#fff;--card:#fff;--line:#cfd5e3;--text:#10162A;--mut:#4a5470;--good:#0E9F6E;--bad:#C21F47;--warn:#A86A12;--track:#e6eaf4;--tile:#f3f5fa}
.nav,footer,.noprint{display:none!important}body{-webkit-print-color-adjust:exact;print-color-adjust:exact}.vc,.sig,.kv div{break-inside:avoid}}
</style>
</head>
<body data-theme="dark">
<header class="nav"><div class="wrap">
  <button class="logo" onclick="go('home')" aria-label="Home"><span class="lg" id="lg"></span><span>Deepfake Detector</span></button>
  <nav class="links" aria-label="Main">
    <button data-v="home" onclick="go('home')">Home</button><button data-v="analyze" onclick="go('analyze')">Analyze</button>
    <button data-v="history" onclick="go('history')">History</button><button data-v="about" onclick="go('about')">About</button>
  </nav>
  <button class="icb" id="theme" aria-label="Switch light or dark theme"></button>
</div></header>

<main>
<section class="view" id="v-home">
  <div class="hero">
    <span class="badge" id="b1"></span>
    <h1>Detect <span class="grad">Deepfakes</span> in Images, Video and Audio.</h1>
    <p class="lead">Upload a file and get a Real or Fake score from fine-tuned ViT and Wav2Vec2 models, with the signals behind it.</p>
    <div style="display:flex;gap:14px;justify-content:center;flex-wrap:wrap"><button class="btn btn-p" onclick="go('analyze')">Analyze Content <span id="ar1"></span></button><button class="btn btn-g" onclick="go('about')">Learn How It Works</button></div>
  </div>
  <div class="wrap sec"><div class="g3" id="feat"></div></div>
  <div class="wrap sec"><div class="eyebrow">WORKFLOW</div><h2 class="t">How It Works</h2><div class="g4" id="steps"></div></div>
  <div class="wrap"><div class="cta"><h2>Verify authenticity in seconds</h2><p>Upload your first file and get a full breakdown with a downloadable report.</p><button class="btn btn-w" onclick="go('analyze')">Start Analyzing</button></div></div>
</section>

<section class="view" id="v-analyze"><div class="narrow" style="padding-top:48px;padding-bottom:64px">
  <div style="text-align:center"><div class="eyebrow">ANALYZE</div><h2 style="font-size:36px;margin:8px 0 24px">Check a file</h2>
  <div class="tabs" role="tablist" id="tabs"></div></div>
  <div class="drop" id="drop"></div>
  <input id="file" type="file" hidden>
  <div class="err" id="err" role="alert"></div>
  <p class="fine">AI detection results are estimates and should not be treated as definitive proof of authorship or authenticity.</p>
</div></section>

<section class="view" id="v-analyzing"><div class="narrow" style="padding-top:64px;padding-bottom:64px;max-width:620px;text-align:center">
  <div class="spin"><svg class="r" viewBox="0 0 112 112" fill="none"><circle cx="56" cy="56" r="52" stroke="var(--track)" stroke-width="4"/><path d="M56 4a52 52 0 0152 52" stroke="#6C5CE7" stroke-width="4" stroke-linecap="round"/></svg><span id="cpu"></span></div>
  <h2 style="font-size:34px">Analyzing your file...</h2><p class="mut" style="margin:8px 0 28px">Running the detection model. Videos can take up to a minute on a laptop.</p>
  <div id="stl" style="text-align:left"></div>
</div></section>

<section class="view" id="v-result"><div class="narrow" id="res" style="padding-top:44px;padding-bottom:64px"></div></section>
<section class="view" id="v-report"><div class="narrow" id="rep" style="padding-top:44px;padding-bottom:64px"></div></section>

<section class="view" id="v-history"><div class="narrow" style="padding-top:48px;padding-bottom:64px">
  <div class="hdr"><div><div class="eyebrow" style="text-align:left">HISTORY</div><h2 style="font-size:34px">Past analyses</h2></div><button class="btn btn-g" id="clr">Clear all</button></div>
  <p class="mut" style="font-size:14px">Saved only in this browser. Delete any item at any time.</p><div id="hist"></div>
</div></section>

<section class="view" id="v-about"><div class="wrap" style="padding-top:48px;padding-bottom:40px">
  <div class="eyebrow">ABOUT</div><h2 class="t">Three models, one website</h2>
  <div class="card" style="overflow:auto"><table><thead><tr><th>Media</th><th>Model</th><th>Trained on</th><th>Test accuracy</th></tr></thead><tbody>
  <tr><td>Image</td><td>ViT-Base</td><td>140k Real and Fake Faces</td><td>86.7% (20,000 images)</td></tr>
  <tr><td>Video</td><td>ViT-Base on 10 face frames</td><td>FaceForensics++ subset, 400 videos</td><td>76.7% (60 videos)</td></tr>
  <tr><td>Audio</td><td>Wav2Vec2-Base</td><td>ASVspoof 2019 (LA)</td><td>98.2% (2,000 clips)</td></tr></tbody></table></div>
  <div class="g2" style="margin-top:20px">
    <div class="card" style="border-color:rgba(245,166,35,.4)"><h3>Accuracy Disclaimer</h3><p class="mut">Detection is probabilistic and can be wrong. Each model works best on content like its training data. Videos from new AI generators and voices recorded on phones or browsers are often misjudged. Results are estimates, not proof.</p></div>
    <div class="card"><h3>Privacy First</h3><p class="mut" style="display:grid;gap:12px"><span>Files are processed on this computer in temporary storage.</span><span>Uploads are deleted right after analysis.</span><span>Your files are never used for training.</span></p></div>
  </div>
  <div style="max-width:720px;margin:56px auto 0"><div class="eyebrow">QUESTIONS</div><h2 class="t" style="margin-bottom:12px">Frequently Asked</h2>
  <details><summary>How accurate is the detection?<span class="chev"></span></summary><p>On unseen test data: 86.7% for images, 76.7% for videos and 98.2% for audio. On content unlike the training data, accuracy drops.</p></details>
  <details><summary>What file formats are supported?<span class="chev"></span></summary><p>Images: JPG, PNG, WEBP, BMP. Video: MP4, MOV, AVI, WEBM, MKV. Audio: MP3, WAV, FLAC, OGG, M4A. Up to 300 MB.</p></details>
  <details><summary>Are my uploaded files stored?<span class="chev"></span></summary><p>No. The file is deleted after analysis. Only the result is kept, in your browser history.</p></details>
  <details><summary>Do you use my files to train models?<span class="chev"></span></summary><p>No. The models were trained once on public datasets.</p></details></div>
</div></section>
</main>

<footer><div class="wrap"><span>&copy; 2026 Deepfake Detector. A Deep Learning P2BL project.</span><span>Detection results are estimates, not definitive proof of authenticity.</span></div></footer>

<script>
const $=s=>document.querySelector(s);
const IC={shield:'<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/><path d="M9 12l2 2 4-4"/>',upload:'<path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a1 1 0 001 1h14a1 1 0 001-1v-3"/>',cpu:'<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3"/>',check:'<path d="M5 12l5 5L20 7"/>',file:'<path d="M14 3H7a2 2 0 00-2 2v14a2 2 0 002 2h10a2 2 0 002-2V8z"/><path d="M14 3v5h5M9 13h6M9 17h6"/>',image:'<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="1.8"/><path d="M21 16l-5-5-8 9"/>',video:'<rect x="3" y="5" width="14" height="14" rx="2"/><path d="M17 10l4-2v8l-4-2z"/>',music:'<path d="M9 18V6l11-2v12"/><circle cx="6" cy="18" r="3"/><circle cx="17" cy="16" r="3"/>',sun:'<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',moon:'<path d="M20 14A8 8 0 019.5 4 8 8 0 1020 14z"/>',trash:'<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>',download:'<path d="M12 4v12M7 11l5 5 5-5M4 20h16"/>',arrow:'<path d="M5 12h14M13 6l6 6-6 6"/>',back:'<path d="M19 12H5M11 6l-6 6 6 6"/>',refresh:'<path d="M20 11a8 8 0 00-14-4L4 9M4 5v4h4M4 13a8 8 0 0014 4l2-2M20 19v-4h-4"/>',info:'<circle cx="12" cy="12" r="9"/><path d="M12 8h.01M11 12h1v5h1"/>',x:'<path d="M6 6l12 12M18 6L6 18"/>',chev:'<path d="M6 9l6 6 6-6"/>',loader:'<path d="M12 3a9 9 0 019 9"/>',fp:'<path d="M7 11a5 5 0 0110 0v3M12 11v8M9 14v4M15 17v3M4 15a8 8 0 0116 0"/>'};
const ic=(n,s=20)=>`<svg width="${s}" height="${s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${IC[n]}</svg>`;
const esc=t=>String(t).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const TABS={image:{l:'Image',ic:'image',acc:'.jpg,.jpeg,.png,.webp,.bmp',ch:'JPG, PNG, WEBP, BMP',ext:['jpg','jpeg','png','webp','bmp']},video:{l:'Video',ic:'video',acc:'.mp4,.avi,.mov,.mkv,.webm',ch:'MP4, MOV, AVI, WEBM',ext:['mp4','avi','mov','mkv','webm']},audio:{l:'Audio',ic:'music',acc:'.wav,.mp3,.flac,.ogg,.m4a',ch:'MP3, WAV, FLAC, OGG',ext:['wav','mp3','flac','ogg','m4a']}};
let tab='image',file=null,last=null;
const kindOf=n=>{const e=(n.split('.').pop()||'').toLowerCase();return Object.keys(TABS).find(k=>TABS[k].ext.includes(e))||null};
const fmtSize=b=>b>1048576?(b/1048576).toFixed(2)+' MB':Math.max(1,Math.round(b/1024))+' KB';

/* static bits */
$('#lg').innerHTML=ic('shield',22);$('#cpu').innerHTML=ic('cpu',40);$('#ar1').innerHTML=ic('arrow',18);
$('#b1').innerHTML=ic('shield',14)+' DEEP LEARNING AUTHENTICITY ANALYSIS';
document.querySelectorAll('.chev').forEach(e=>e.innerHTML=ic('chev',18));
$('#feat').innerHTML=[['i1','image','Image Detection','Checks faces in JPG, PNG, WEBP and more with a fine-tuned Vision Transformer.','Test accuracy 86.7%'],['i2','video','Video Detection','Samples 10 frames, crops the face in each and averages the scores.','Test accuracy 76.7%'],['i3','music','Audio Detection','Finds synthetic speech with a fine-tuned Wav2Vec2 model.','Test accuracy 98.2%']].map(f=>`<div class="card"><div class="ico ${f[0]}">${ic(f[1],26)}</div><h3>${f[2]}</h3><p class="mut">${f[3]}</p><span class="tag">${f[4]}</span></div>`).join('');
$('#steps').innerHTML=[['upload','Upload','Drag and drop an image, video or audio file.'],['cpu','Analyze','The matching model scores the file.'],['shield','Verify','Get an AI probability with the signals behind it.'],['file','Report','Download the findings as a PDF report.']].map((s,i)=>`<div class="card step"><span class="n">${i+1}</span><div class="si">${ic(s[0],22)}</div><h3 style="font-size:19px">${s[1]}</h3><p class="mut" style="font-size:15px">${s[2]}</p></div>`).join('');

/* theme */
function setTheme(t){document.body.dataset.theme=t;$('#theme').innerHTML=ic(t==='dark'?'sun':'moon',20);try{localStorage.setItem('dd_theme',t)}catch(e){}}
$('#theme').onclick=()=>setTheme(document.body.dataset.theme==='dark'?'light':'dark');
try{setTheme(localStorage.getItem('dd_theme')||'dark')}catch(e){setTheme('dark')}

/* router */
const go=v=>{location.hash=v};
function show(){let v=(location.hash||'#home').slice(1);if(!['home','analyze','analyzing','result','report','history','about'].includes(v))v='home';
  if((v==='result'||v==='report')&&!last)v='analyze';if(v==='analyzing'&&!busy)v='analyze';
  document.querySelectorAll('.view').forEach(e=>e.classList.toggle('on',e.id==='v-'+v));
  document.querySelectorAll('.links button').forEach(b=>b.classList.toggle('on',b.dataset.v===(v==='result'||v==='report'||v==='analyzing'?'analyze':v)));
  if(v==='analyze')drawDrop();if(v==='history')drawHist();if(v==='result')$('#res').innerHTML=resultPage(last);if(v==='report')$('#rep').innerHTML=reportPage(last);window.scrollTo(0,0)}
window.onhashchange=show;let busy=false;

/* analyze page */
function drawTabs(){$('#tabs').innerHTML=Object.keys(TABS).map(k=>`<button class="tab ${k===tab?'on':''}" role="tab" onclick="setTab('${k}')">${ic(TABS[k].ic,17)} ${TABS[k].l}</button>`).join('')}
function setTab(k){tab=k;file=null;drawDrop()}
function drawDrop(){drawTabs();const T=TABS[tab];$('#file').accept=T.acc;$('#err').style.display='none';const d=$('#drop');
  if(!file){d.innerHTML=`<div class="ico i1">${ic('upload',30)}</div><h3 style="font-size:26px">Drag &amp; drop your ${tab} here</h3><p class="mut" style="margin:6px 0 22px">or click to browse from your device</p><button class="btn btn-p" onclick="$('#file').click()">Browse Files</button><div class="chips"><span class="chip">${T.ch}</span><span class="chip">Max 300 MB</span></div>`;return}
  const u=URL.createObjectURL(file);const pv=tab==='image'?`<img class="prev" src="${u}" alt="Selected image">`:tab==='video'?`<video class="prev" src="${u}" controls></video>`:`<audio class="prev" src="${u}" controls></audio>`;
  d.innerHTML=`${pv}<h3 style="font-size:22px;word-break:break-all">${esc(file.name)}</h3><p class="mut" style="margin:6px 0 20px">${fmtSize(file.size)}</p><button class="btn btn-p" onclick="analyze()">Analyze Content</button> <button class="btn btn-g" onclick="file=null;drawDrop()">${ic('x',16)} Remove</button>`}
function pick(f){if(!f)return;const k=kindOf(f.name),e=$('#err');if(!k){file=null;drawDrop();e.textContent='This file type is not supported. Use an image, video or audio file.';e.style.display='block';return}
  if(f.size>300*1048576){e.textContent='File is larger than 300 MB.';e.style.display='block';return}tab=k;file=f;drawDrop()}
$('#file').onchange=e=>pick(e.target.files[0]);$('#file').onclick=e=>{e.target.value=''};
const dz=$('#drop');dz.onclick=e=>{if(!file&&!e.target.closest('button'))$('#file').click()};
dz.ondragover=e=>{e.preventDefault();dz.classList.add('over')};dz.ondragleave=()=>dz.classList.remove('over');dz.ondrop=e=>{e.preventDefault();dz.classList.remove('over');pick(e.dataTransfer.files[0])};

/* analyzing */
const STEPS=['Uploading file','Extracting content','Running AI detection','Checking authenticity indicators','Generating report'];
function steps(n){$('#stl').innerHTML=STEPS.map((s,i)=>`<div class="st ${i<n?'done':i===n?'run':''}"><span class="b">${i<n?ic('check',20):i===n?ic('loader',20):ic('file',18)}</span>${s}</div>`).join('')}
function analyze(){if(!file)return;busy=true;steps(0);go('analyzing');let i=0;const tm=setInterval(()=>{if(i<3){i++;steps(i)}},1800);
  const fd=new FormData();fd.append('file',file);const x=new XMLHttpRequest();x.open('POST','/predict');
  const fail=m=>{clearInterval(tm);busy=false;go('analyze');const e=$('#err');e.textContent=m;e.style.display='block'};
  x.onload=()=>{let r;try{r=JSON.parse(x.responseText)}catch(e){return fail('The server returned an unexpected answer.')}
    if(x.status!==200)return fail(r.error||'Something went wrong.');clearInterval(tm);steps(5);last=r;save(r);setTimeout(()=>{busy=false;go('result')},500)};
  x.onerror=()=>fail('Could not reach the server. Is app.py still running?');x.send(fd)}

/* result builders */
const tone=r=>r.label.includes('AI')?'bad':r.label.includes('Real')?'good':'warn';
const desc=r=>({bad:`The model scored this ${r.kind} as likely AI-generated or manipulated.`,good:`The model found this ${r.kind} more likely to be authentic.`,warn:'The score is close to the middle, so the model cannot decide.'})[tone(r)];
function ring(a,t){const c=`var(--${t})`;return `<svg viewBox="0 0 220 220" class="ring" role="img" aria-label="AI probability ${a} percent"><circle cx="110" cy="110" r="96" fill="none" stroke="var(--track)" stroke-width="14"/><circle cx="110" cy="110" r="96" fill="none" stroke="${c}" stroke-width="14" stroke-linecap="round" stroke-dasharray="${(a/100*603.2).toFixed(1)} 603.2" transform="rotate(-90 110 110)"/><text x="110" y="96" text-anchor="middle" font-family="IBM Plex Mono" font-size="11" letter-spacing="3" fill="var(--mut)">AI PROBABILITY</text><text x="110" y="142" text-anchor="middle" font-family="Space Grotesk" font-size="50" font-weight="700" fill="${c}">${Math.round(a)}%</text></svg>`}
function verdict(r){const t=tone(r);return `<div class="vc ${t}"><div><span class="pill ${t}">${ic(t==='good'?'check':'info',16)} ${r.label}</span><h2 class="vt ${t}">${r.label}</h2><p class="mut">${desc(r)}</p><div class="tiles"><div class="tile"><small>AI</small><b style="color:var(--bad)">${r.ai_probability}%</b></div><div class="tile"><small>Real</small><b style="color:var(--good)">${r.real_probability}%</b></div><div class="tile"><small>Confidence</small><b>${r.confidence}</b></div></div></div><div>${ring(r.ai_probability,t)}</div></div>`}
function bars(r){const s=r.details.frame_scores||r.details.clip_scores;if(!s)return '';return `<div class="sig"><h4>${r.kind==='video'?'AI probability per frame':'AI probability per audio clip'}</h4><div class="bars">${s.map((v,i)=>`<i style="height:${Math.max(v,3)}%" title="${r.kind==='video'?'Frame':'Clip'} ${i+1}: ${v}%"></i>`).join('')}</div><p style="font-size:13px;margin-top:8px">Taller bars mean the model found that part more likely AI-generated.</p></div>`}
function signals(r){const m=r.model,d=r.details,s=[['Model used',`${m.name}, fine-tuned on ${m.dataset}.`],['Accuracy on unseen test data',`${m.accuracy} on ${m.test_size}.`]];
  if(r.kind==='image')s.push(['Input','The full image is resized to 224 x 224 pixels and scored once.']);
  if(r.kind==='video')s.push(['Frames analysed',`${d.frames} frames sampled evenly across the video. A face was found in ${d.faces_found} of them.`]);
  if(r.kind==='audio')s.push(['Clips analysed',`${d.clip_scores.length} clip(s) of up to 4 seconds at 16 kHz.`]);
  s.push(['How the label is decided',`The model gives an AI probability of ${r.ai_probability}%. Above 60% is labelled Likely AI-Generated, below 40% Likely Real, and anything between is Inconclusive.`],['Known limit of this model',m.limit]);
  return s.map(x=>`<div class="sig"><h4>${x[0]}</h4><p>${esc(x[1])}</p></div>`).join('')+bars(r)}
function summary(r){const s=r.details.frame_scores||r.details.clip_scores;let t=`The ${r.kind} model gave this file an AI probability of ${r.ai_probability}% (${r.real_probability}% real).`;
  if(s&&s.length>1)t+=` Individual ${r.kind==='video'?'frames':'clips'} ranged from ${Math.min(...s)}% to ${Math.max(...s)}%.`;return t+' This summary comes from the model score and not from a visual or audio inspection of the content.'}
const warn=`<div class="warnbox">${ic('info',20)}<span>AI detection is probabilistic and can produce false positives and false negatives. Results are estimates and should not be treated as definitive proof of authorship or authenticity.</span></div>`;
function body(r){return `${verdict(r)}<div style="margin-top:24px">${signals(r)}</div><div class="sig"><h4>Technical Analysis Summary</h4><p>${esc(summary(r))}</p></div><div class="hash">${ic('fp',16)} SHA-256: ${r.sha256}</div>${warn}`}
function resultPage(r){return `<div class="hdr"><h2 style="font-size:32px">Analysis Result</h2><button class="btn btn-g" onclick="file=null;go('analyze')">${ic('refresh',16)} Analyze Another File</button></div>${body(r)}<div style="margin-top:24px"><button class="btn btn-p" onclick="go('report')">View Detailed Report</button></div>`}
function reportPage(r){const at=new Date(r.analyzed_at).toLocaleString();return `<div class="hdr noprint"><button class="btn btn-g" onclick="go('result')">${ic('back',16)} Back</button><button class="btn btn-p" onclick="window.print()">${ic('download',16)} Download Report (PDF)</button></div><p class="noprint mut" style="font-size:13px;margin:-8px 0 16px;text-align:right">In the print window, set Destination to Save as PDF.</p><div class="mono mut" style="font-size:12px;letter-spacing:.2em">DETAILED REPORT</div><h2 style="font-size:34px;margin:6px 0;word-break:break-all">${esc(r.filename)}</h2><div class="mono mut" style="font-size:12px;margin-bottom:22px">Analysis ID: ${r.id}</div>${verdict(r)}<div class="kv"><div><small>File type</small><span>${r.kind}</span></div><div><small>File name</small><span>${esc(r.filename)}</span></div><div><small>File size</small><span>${fmtSize(r.size)}</span></div><div><small>Analyzed</small><span>${at}</span></div></div><h3 style="margin:28px 0 4px;font-size:20px">Detection Signals</h3><p class="mut" style="font-size:14px">These are values observed during analysis, not definitive proof.</p>${signals(r)}<div class="sig"><h4>Technical Analysis Summary</h4><p>${esc(summary(r))}</p></div><div class="hash">${ic('fp',16)} SHA-256: ${r.sha256}</div>${warn}`}

/* history */
const getH=()=>{try{return JSON.parse(localStorage.getItem('dd_hist')||'[]')}catch(e){return[]}};
const putH=h=>{try{localStorage.setItem('dd_hist',JSON.stringify(h.slice(0,50)))}catch(e){}};
function save(r){const h=getH();h.unshift(r);putH(h)}
function drawHist(){const h=getH();$('#hist').innerHTML=h.length?h.map((r,i)=>`<div class="hi"><div style="min-width:0"><b style="word-break:break-all">${esc(r.filename)}</b><div class="mut" style="font-size:13px">${r.kind} &middot; ${new Date(r.analyzed_at).toLocaleString()}</div></div><div style="display:flex;gap:10px;align-items:center"><b style="color:var(--${tone(r)})">${r.label} &middot; ${r.ai_probability}%</b><button class="btn btn-g" style="padding:8px 14px" onclick="openH(${i})">Report</button><button class="icb" style="width:38px;height:38px" aria-label="Delete" onclick="delH(${i})">${ic('trash',16)}</button></div></div>`).join(''):'<p class="mut" style="margin-top:28px">No analyses yet.</p>'}
const openH=i=>{last=getH()[i];go('report')};
const delH=i=>{const h=getH();h.splice(i,1);putH(h);drawHist()};
$('#clr').onclick=()=>{putH([]);drawHist()};
show();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)