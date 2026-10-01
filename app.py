import io
import os
import shutil
import subprocess
import tempfile
import time

import cv2
import easyocr
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# =============================================================
# Page config
# =============================================================
st.set_page_config(
    page_title="PlateScan · License Plate Recognition",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================
# Styling
# =============================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Barlow:wght@400;500;600&display=swap');

:root {
    --ink: #0d1320;
    --panel: #151d2e;
    --panel-2: #1c2639;
    --line: #2a364d;
    --text: #e8ecf4;
    --muted: #8d99b0;
    --plate-yellow: #f5c400;
    --ok: #3ddc97;
}

html, body, [class*="css"], .stApp {
    font-family: 'Barlow', sans-serif;
    color: var(--text);
}
.stApp { background: var(--ink); }

#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1280px; }

h1, h2, h3, h4 {
    font-family: 'Barlow Condensed', sans-serif !important;
    letter-spacing: 0.01em;
    font-weight: 700 !important;
    color: var(--text) !important;
}

/* ---------- Hero ---------- */
.hero {
    display: flex; align-items: center; justify-content: space-between;
    gap: 2rem; flex-wrap: wrap;
    padding: 2rem 2.2rem; margin-bottom: 1.6rem;
    background: linear-gradient(135deg, var(--panel) 0%, var(--panel-2) 100%);
    border: 1px solid var(--line); border-radius: 18px;
}
.hero h1 {
    font-size: 3.2rem; line-height: 1; margin: 0 0 .6rem 0;
}
.hero p { color: var(--muted); font-size: 1.05rem; max-width: 520px; margin: 0; }

/* The one memorable element: a real plate */
.plate {
    display: inline-flex; flex-direction: column; align-items: center;
    background: #fff; color: #111;
    border: 4px solid #111; border-radius: 12px;
    padding: .35rem 1.3rem .45rem;
    box-shadow: 0 0 0 3px var(--plate-yellow), 0 10px 30px rgba(0,0,0,.45);
    min-width: 190px;
}
.plate .plate-text {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700; font-size: 2.6rem; letter-spacing: .12em; line-height: 1.1;
}
.plate .plate-sub {
    font-size: .68rem; letter-spacing: .08em; color: #555; font-weight: 600;
}
.plate.small { min-width: 0; padding: .2rem .9rem .3rem; border-width: 3px; }
.plate.small .plate-text { font-size: 1.7rem; }

/* ---------- Metrics ---------- */
.metric-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 1rem; margin: 1.2rem 0; }
.metric {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 14px; padding: 1rem 1.2rem;
}
.metric .v {
    font-family: 'Barlow Condensed', sans-serif; font-size: 2.1rem;
    font-weight: 700; line-height: 1.1;
}
.metric .l { color: var(--muted); font-size: .9rem; }

/* ---------- Result card ---------- */
.result {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 14px; padding: 1rem 1.2rem; margin-bottom: .8rem;
}
.bar { height: 6px; border-radius: 6px; background: var(--line); overflow: hidden; margin-top: .6rem; }
.bar > div { height: 100%; background: var(--ok); }
.bar.mid > div { background: var(--plate-yellow); }
.bar.low > div { background: #ff6b6b; }
.conf-label { color: var(--muted); font-size: .85rem; margin-top: .4rem; }

/* ---------- Empty state ---------- */
.empty {
    border: 1.5px dashed var(--line); border-radius: 14px;
    padding: 2.2rem; text-align: center; color: var(--muted);
}

/* ---------- Streamlit widgets ---------- */
section[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] * { color: var(--text); }

.stTabs [data-baseweb="tab-list"] { gap: .4rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] {
    font-family: 'Barlow Condensed', sans-serif; font-size: 1.25rem; font-weight: 600;
    padding: .6rem 1.2rem; border-radius: 10px 10px 0 0;
}
.stTabs [aria-selected="true"] { color: var(--plate-yellow) !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--plate-yellow) !important; }

[data-testid="stFileUploaderDropzone"] {
    background: var(--panel); border: 1.5px dashed var(--line); border-radius: 14px;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--plate-yellow); }

.stButton > button, .stDownloadButton > button {
    border-radius: 10px; font-weight: 600; padding: .6rem 1.4rem;
    border: 1px solid var(--line); background: var(--panel-2); color: var(--text);
}
.stButton > button[kind="primary"] {
    background: var(--plate-yellow); color: #111; border: none;
}
.stButton > button[kind="primary"]:hover { background: #ffd62e; color: #000; }
.stDownloadButton > button:hover { border-color: var(--plate-yellow); color: var(--plate-yellow); }

.stProgress > div > div > div > div { background-color: var(--plate-yellow); }
img { border-radius: 12px; }

@media (max-width: 640px) {
    .hero h1 { font-size: 2.3rem; }
    .plate .plate-text { font-size: 2rem; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =============================================================
# Models
# =============================================================
@st.cache_resource(show_spinner="Loading detection and OCR models…")
def load_models():
    model = YOLO("best.pt")
    reader = easyocr.Reader(["en"])
    return model, reader


model, reader = load_models()

# =============================================================
# Core functions
# =============================================================
BOX_RGB = (245, 196, 0)  # plate yellow


def clean_text(text: str) -> str:
    return " ".join(text.upper().split())


def read_plate(plate, scale=3):
    """Read plate text with EasyOCR. Returns (text, confidence)."""
    if plate is None or plate.size == 0:
        return None, 0.0

    plate = cv2.resize(plate, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    results = reader.readtext(plate)
    if not results:
        return None, 0.0

    # Join all text pieces left-to-right (plates are often split in 2 parts)
    results = sorted(results, key=lambda r: r[0][0][0])
    text = clean_text(" ".join(r[1] for r in results))
    conf = float(np.mean([r[2] for r in results]))
    return text, conf


def draw_label(img, text, x1, y1, color):
    """Box label with a filled background so it stays readable."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, 0.8, 2)
    top = max(y1 - th - base - 8, 0)
    cv2.rectangle(img, (x1, top), (x1 + tw + 12, top + th + base + 8), color, -1)
    cv2.putText(img, text, (x1 + 6, top + th + 3), font, 0.8, (17, 17, 17), 2, cv2.LINE_AA)


def detect_and_read(frame, conf, imgsz, scale, color):
    """Run YOLO + OCR on one frame. Draws on the frame in place."""
    result = model.predict(source=frame, conf=conf, imgsz=imgsz, verbose=False)[0]
    h, w = frame.shape[:2]
    found = []

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2].copy()
        text, ocr_conf = read_plate(crop, scale)

        found.append({
            "plate": text or "Not read",
            "readable": text is not None,
            "ocr_conf": ocr_conf,
            "det_conf": float(box.conf[0]),
            "crop": crop,
            "box": (x1, y1, x2, y2),
        })

        label = f"{text} ({ocr_conf:.0%})" if text else "Plate"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        draw_label(frame, label, x1, y1, color)

    return found


def to_png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def make_browser_playable(src_path):
    """Re-encode to H.264 if ffmpeg is available (mp4v won't play in browsers)."""
    if not shutil.which("ffmpeg"):
        return src_path
    dst = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    cmd = [
        "ffmpeg", "-y", "-i", src_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return dst
    except Exception:
        return src_path


# =============================================================
# UI helpers
# =============================================================
def plate_html(text, small=False, sub="LICENSE PLATE"):
    cls = "plate small" if small else "plate"
    sub_html = f'<span class="plate-sub">{sub}</span>' if sub else ""
    return f'<div class="{cls}"><span class="plate-text">{text}</span>{sub_html}</div>'


def conf_class(c):
    return "" if c >= 0.75 else ("mid" if c >= 0.5 else "low")


def metrics_html(items):
    cells = "".join(
        f'<div class="metric"><div class="v">{v}</div><div class="l">{l}</div></div>'
        for v, l in items
    )
    return f'<div class="metric-row">{cells}</div>'


def empty_state(msg):
    st.markdown(f'<div class="empty">{msg}</div>', unsafe_allow_html=True)


# =============================================================
# Sidebar settings
# =============================================================
with st.sidebar:
    st.markdown("### Settings")
    det_conf = st.slider(
        "Detection sensitivity", 0.05, 0.90, 0.15, 0.05,
        help="Lower finds more plates but may include false matches.",
    )
    imgsz = st.select_slider(
        "Detection image size", options=[640, 800, 960, 1280], value=960,
        help="Larger sizes find small plates better but run slower.",
    )
    ocr_scale = st.slider(
        "OCR zoom", 1, 5, 3,
        help="Enlarges each plate before reading. Higher can improve accuracy on small plates.",
    )
    min_ocr = st.slider(
        "Minimum text confidence", 0.0, 1.0, 0.0, 0.05,
        help="Hide plates whose text is read with lower confidence.",
    )
    st.markdown("#### Video")
    frame_skip = st.slider(
        "Analyze every Nth frame", 1, 10, 2,
        help="Skipped frames reuse the last result. Higher is faster.",
    )
    st.caption("Tip: if plates are missed, lower the sensitivity or raise the image size.")

# =============================================================
# Hero
# =============================================================
st.markdown(
    f"""
    <div class="hero">
        <div>
            <h1>Read any plate in seconds</h1>
            <p>Upload a photo or a video. PlateScan finds each license plate and reads the number for you.</p>
        </div>
        {plate_html("ABC 1234", sub="PLATESCAN")}
    </div>
    """,
    unsafe_allow_html=True,
)

tab_img, tab_vid = st.tabs(["Image", "Video"])

# =============================================================
# IMAGE TAB
# =============================================================
with tab_img:
    img_file = st.file_uploader(
        "Drop an image here", type=["jpg", "jpeg", "png"], key="img_up"
    )

    if not img_file:
        empty_state("Upload a JPG or PNG to get started.")
    else:
        image_np = np.array(Image.open(img_file).convert("RGB"))

        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("#### Original")
            st.image(image_np, use_container_width=True)

        run = st.button("Detect license plates", type="primary", key="img_run")

        if run:
            with st.spinner("Finding and reading plates…"):
                t0 = time.time()
                out = image_np.copy()
                detections = detect_and_read(out, det_conf, imgsz, ocr_scale, BOX_RGB)
                elapsed = time.time() - t0

            shown = [d for d in detections if d["ocr_conf"] >= min_ocr]

            with right:
                st.markdown("#### Result")
                st.image(out, use_container_width=True)
                st.download_button(
                    "Download result image",
                    to_png_bytes(out),
                    file_name="plate_result.png",
                    mime="image/png",
                )

            readable = [d for d in shown if d["readable"]]
            avg = np.mean([d["ocr_conf"] for d in readable]) if readable else 0
            st.markdown(
                metrics_html([
                    (len(shown), "Plates found"),
                    (len(readable), "Plates read"),
                    (f"{avg:.0%}", "Average text confidence"),
                    (f"{elapsed:.1f}s", "Processing time"),
                ]),
                unsafe_allow_html=True,
            )

            st.markdown("#### Detected plates")
            if not shown:
                empty_state(
                    "No license plates found. Try lowering the detection sensitivity "
                    "in the sidebar, or upload a clearer image."
                )
            for i, d in enumerate(shown, 1):
                c1, c2 = st.columns([1, 2], gap="medium")
                with c1:
                    st.image(d["crop"], use_container_width=True)
                with c2:
                    st.markdown(
                        f"""
                        <div class="result">
                            {plate_html(d["plate"], small=True, sub=f"PLATE {i}")}
                            <div class="bar {conf_class(d['ocr_conf'])}"><div style="width:{d['ocr_conf']*100:.0f}%"></div></div>
                            <div class="conf-label">Text confidence {d['ocr_conf']:.0%}
                            &nbsp;·&nbsp; Detection confidence {d['det_conf']:.0%}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# =============================================================
# VIDEO TAB
# =============================================================
with tab_vid:
    vid_file = st.file_uploader(
        "Drop a video here", type=["mp4", "avi", "mov", "mkv"], key="vid_up"
    )

    if not vid_file:
        empty_state("Upload an MP4, AVI, MOV or MKV video to get started.")
    else:
        st.markdown("#### Original")
        st.video(vid_file)

        if st.button("Process video", type="primary", key="vid_run"):
            in_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            in_tmp.write(vid_file.getvalue())
            in_tmp.close()
            raw_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

            cap = cv2.VideoCapture(in_tmp.name)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            writer = cv2.VideoWriter(
                raw_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )

            progress = st.progress(0.0, text="Starting…")
            best = {}  # plate text -> best OCR confidence
            last_boxes = []  # (box, label) reused on skipped frames
            frame_no = 0
            t0 = time.time()
            BOX_BGR = BOX_RGB[::-1]

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_no % frame_skip == 0:
                    found = detect_and_read(frame, det_conf, imgsz, ocr_scale, BOX_BGR)
                    last_boxes = []
                    for d in found:
                        if d["readable"] and d["ocr_conf"] >= min_ocr:
                            best[d["plate"]] = max(best.get(d["plate"], 0), d["ocr_conf"])
                        label = (f"{d['plate']} ({d['ocr_conf']:.0%})"
                                 if d["readable"] else "Plate")
                        last_boxes.append((d["box"], label))
                else:
                    for (x1, y1, x2, y2), label in last_boxes:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_BGR, 3)
                        draw_label(frame, label, x1, y1, BOX_BGR)

                writer.write(frame)
                frame_no += 1

                if total > 0 and frame_no % 3 == 0:
                    progress.progress(
                        min(frame_no / total, 1.0),
                        text=f"Processing frame {frame_no} of {total}",
                    )

            cap.release()
            writer.release()
            progress.progress(1.0, text="Done")
            elapsed = time.time() - t0

            final_path = make_browser_playable(raw_out)
            with open(final_path, "rb") as f:
                video_bytes = f.read()

            st.markdown("#### Processed video")
            st.video(video_bytes)
            st.download_button(
                "Download processed video",
                video_bytes,
                file_name="plate_result.mp4",
                mime="video/mp4",
            )
            if final_path == raw_out and not shutil.which("ffmpeg"):
                st.caption(
                    "Install ffmpeg on the server for smoother in-browser playback. "
                    "The download works either way."
                )

            st.markdown(
                metrics_html([
                    (len(best), "Unique plates"),
                    (frame_no, "Frames processed"),
                    (f"{elapsed:.0f}s", "Processing time"),
                ]),
                unsafe_allow_html=True,
            )

            st.markdown("#### Detected plates")
            if not best:
                empty_state(
                    "No readable license plates found. Try lowering the detection "
                    "sensitivity or analyzing every frame."
                )
            else:
                cols = st.columns(3, gap="medium")
                ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
                for i, (text, c) in enumerate(ranked):
                    with cols[i % 3]:
                        st.markdown(
                            f"""
                            <div class="result">
                                {plate_html(text, small=True, sub="")}
                                <div class="bar {conf_class(c)}"><div style="width:{c*100:.0f}%"></div></div>
                                <div class="conf-label">Best confidence {c:.0%}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            for p in {in_tmp.name, raw_out, final_path}:
                try:
                    os.remove(p)
                except OSError:
                    pass