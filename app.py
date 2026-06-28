"""
Local Gradio demo for MVInverse (multi-view inverse rendering).

Equivalent to the HuggingFace Space `maddog241/mvinverse-demo`, adapted to run
locally on this repo's uv (cu128) environment:
  - inference runs under fp16 autocast (matches inference.py; fits a 32GB GPU),
  - the bundled `examples/<scene>/` image folders are auto-encoded to short mp4s
    on first launch so they can be used as Video examples (the upstream Space
    ships mp4 examples; this repo ships image folders).

Run:  uv run python app.py    →  http://localhost:7860
"""
import os
import zipfile

import cv2
import numpy as np
import torch
import gradio as gr
from PIL import Image
from torchvision import transforms

from mvinverse.models.mvinverse import MVInverse

# ZeroGPU decorator only on HF Spaces; no-op locally.
if os.environ.get("SPACE_ID"):
    from spaces import GPU
    gpu_decorator = GPU(duration=120)
else:
    def gpu_decorator(func):
        return func

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_FRAMES = 16
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ABS_EXAMPLES_DIR = os.path.join(BASE_DIR, "examples")
EXAMPLES_MP4_DIR = os.path.join(BASE_DIR, "examples_mp4")

CSS = ".gradio-container { max-width: 1400px; margin: 0 auto !important; }"

TITLE = """
# **MVInverse: Feed-forward Multi-view Inverse Rendering in Seconds**
Recover **albedo, metallic, roughness, normal and shading** from
**multi-view images or video**. Project page: https://maddog241.github.io/mvinverse-page/
"""

# ---------------------------------------------------------------------------
# Model (loaded once at startup)
# ---------------------------------------------------------------------------
print("Loading MVInverse...")
model = MVInverse.from_pretrained("maddog241/mvinverse").to(DEVICE).eval()
print("Model loaded.")


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------
def build_example_videos():
    """Encode each bundled examples/<scene>/ image folder into a short mp4 so it
    can be offered as a Video example (idempotent / cached on disk)."""
    os.makedirs(EXAMPLES_MP4_DIR, exist_ok=True)
    vids = []
    if not os.path.isdir(ABS_EXAMPLES_DIR):
        return vids
    for name in sorted(os.listdir(ABS_EXAMPLES_DIR)):
        d = os.path.join(ABS_EXAMPLES_DIR, name)
        if not os.path.isdir(d):
            continue
        imgs = sorted(f for f in os.listdir(d) if f.lower().endswith(("jpg", "jpeg", "png")))
        if not imgs:
            continue
        out = os.path.join(EXAMPLES_MP4_DIR, f"{name}.mp4")
        if not os.path.exists(out):
            try:
                first = cv2.imread(os.path.join(d, imgs[0]))
                h, w = first.shape[:2]
                vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 5, (w, h))
                for im in imgs:
                    frame = cv2.imread(os.path.join(d, im))
                    if frame is None:
                        continue
                    if frame.shape[:2] != (h, w):
                        frame = cv2.resize(frame, (w, h))
                    vw.write(frame)
                vw.release()
            except Exception as e:  # noqa: BLE001
                print(f"[examples] failed to build {out}: {e}")
                continue
        if os.path.exists(out):
            vids.append(out)
    return vids


def preprocess_images(pil_images):
    imgs = []
    for img in pil_images:
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > 1024:
            s = 1024 / max(w, h)
            w, h = int(w * s), int(h * s)
        w, h = w // 14 * 14, h // 14 * 14
        imgs.append(img.resize((w, h)))
    t = transforms.ToTensor()
    return torch.stack([t(i) for i in imgs], dim=0)


def load_from_zip(path):
    imgs = []
    with zipfile.ZipFile(path, "r") as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith(("png", "jpg", "jpeg")))
        for n in names[:MAX_FRAMES]:
            with z.open(n) as f:
                imgs.append(Image.open(f).copy())
    return imgs


def load_from_video(path, stride):
    cap = cv2.VideoCapture(path)
    frames, idx = [], 0
    while cap.isOpened() and len(frames) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        idx += 1
    cap.release()
    return frames


def to_pil(x, normal=False):
    if normal:
        x = x * 0.5 + 0.5
    return Image.fromarray((x.float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@gpu_decorator
@torch.no_grad()
def run_inference(input_type, single, multi, zip_f, video, stride):
    if input_type == "Single Image":
        if single is None:
            raise gr.Error("Please provide an image.")
        imgs = [single]
    elif input_type == "Multiple Images":
        if not multi:
            raise gr.Error("Please provide images.")
        imgs = [Image.open(f.name).copy() for f in multi][:MAX_FRAMES]
    elif input_type == "ZIP":
        if zip_f is None:
            raise gr.Error("Please provide a .zip of images.")
        imgs = load_from_zip(zip_f.name)
    elif input_type == "Video":
        if not video:
            raise gr.Error("Please provide a video.")
        imgs = load_from_video(video, int(stride))
    else:
        raise gr.Error("Invalid input type")

    if not imgs:
        raise gr.Error("No frames found in the input.")

    imgs = preprocess_images(imgs).to(DEVICE)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        res = model(imgs[None])

    albedo, metallic, roughness, normal, shading = [], [], [], [], []
    for i in range(res["albedo"].shape[1]):
        albedo.append(to_pil(res["albedo"][0, i]))
        metallic.append(to_pil(res["metallic"][0, i].squeeze(-1)))
        roughness.append(to_pil(res["roughness"][0, i].squeeze(-1)))
        normal.append(to_pil(res["normal"][0, i], True))
        shading.append(to_pil(res["shading"][0, i]))
    return albedo, metallic, roughness, normal, shading


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
examples_list = [[v, "Video"] for v in build_example_videos()]

with gr.Blocks(title="MVInverse Demo") as demo:
    gr.Markdown(TITLE)

    with gr.Row():
        with gr.Column(scale=1):
            input_type = gr.Radio(
                ["Single Image", "Multiple Images", "ZIP", "Video"],
                value="Video",
                label="🛠️ Input Type",
            )
            single = gr.Image(type="pil", visible=False, label="Single Image")
            multi = gr.File(file_count="multiple", visible=False, label="Multiple Images")
            zip_f = gr.File(file_types=[".zip"], visible=False, label="Zip File")
            video = gr.Video(visible=True, format="mp4", label="Input Video")
            stride = gr.Slider(1, 10, 1, step=1, visible=True, label="Stride")

            def switch(t):
                return (
                    gr.update(visible=t == "Single Image"),
                    gr.update(visible=t == "Multiple Images"),
                    gr.update(visible=t == "ZIP"),
                    gr.update(visible=t == "Video"),
                    gr.update(visible=t == "Video"),
                )

            input_type.change(switch, input_type, [single, multi, zip_f, video, stride])
            run = gr.Button("🚀 Run", variant="primary")

            if examples_list:
                gr.Markdown("### 🎥 Examples")
                gr.Examples(
                    examples=examples_list,
                    inputs=[video, input_type],
                    label="Click to try:",
                    cache_examples=False,
                    run_on_click=False,
                )

        with gr.Column(scale=3):
            gr.Markdown("### ▶️ Results")
            albedo = gr.Gallery(columns=4, height=260, preview=False, object_fit="contain", label="Albedo")
            with gr.Row():
                metallic = gr.Gallery(columns=4, height=260, preview=False, object_fit="contain", label="Metallic")
                roughness = gr.Gallery(columns=4, height=260, preview=False, object_fit="contain", label="Roughness")
            with gr.Row():
                normal = gr.Gallery(columns=4, height=260, preview=False, object_fit="contain", label="Normal")
                shading = gr.Gallery(columns=4, height=260, preview=False, object_fit="contain", label="Shading")

    run.click(
        run_inference,
        [input_type, single, multi, zip_f, video, stride],
        [albedo, metallic, roughness, normal, shading],
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=CSS,
        allowed_paths=[BASE_DIR, "/tmp", ABS_EXAMPLES_DIR, EXAMPLES_MP4_DIR],
    )
