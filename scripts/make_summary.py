#!/usr/bin/env python
"""
Build a per-frame summary image for MVInverse outputs, including a reconstruction.

Each summary tiles the input frame, a reconstruction, and the predicted maps
with labels. MVInverse outputs PNG maps (albedo / metallic / roughness /
normal / shading) per frame, so this is done in PIL/numpy.

Reconstruction = albedo (.) shading, the diffuse recomposition stated in the
MVInverse paper ("the diffuse image is obtained as the product of albedo and
diffuse shading"). Albedo is trained with a scale-invariant loss, so the product
is only defined up to a per-channel scale; we recover that scale in closed form
(least squares to the input, the paper's s*) before display. This reproduces the
diffuse part of the input; specular/metallic effects are not captured (the paper
defines no full PBR re-render, and there is no rendering loss in training).

Layout (default cols=2), row 1 = Input | Reconstructed:
    [ Input        | Reconstructed ]
    [ Albedo       | Shading       ]
    [ Normal       | Metallic      ]
    [ Roughness    | (pad)         ]

Multi-image batch + input scaling: every input frame is resized to the map
resolution the model actually produced (longest side <=1024, multiple of 14),
so the input/reconstruction line up pixel-for-pixel with the maps. The script
prints a verification block (counts + per-frame sizes + recon stats) so the
scaling/alignment can be checked.

Usage:
  uv run python scripts/make_summary.py \
      --input_dir examples/Courtroom \
      --maps_dir outputs/smoke/Courtroom \
      --out_dir outputs/smoke/Courtroom_summary [--cols 2]
"""
import argparse
import os
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

MAPS = ["albedo", "metallic", "roughness", "normal", "shading"]
IMG_EXT = (".jpg", ".jpeg", ".png")


def find_font(size):
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def label(img, text, font):
    d = ImageDraw.Draw(img)
    box = d.textbbox((0, 0), text, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    d.rectangle([0, 0, tw + 18, th + 14], fill=(0, 0, 0))
    d.text((9, 4), text, fill=(255, 255, 255), font=font)
    return img


def to_rgb(im, size):
    return im.convert("RGB").resize(size, Image.BILINEAR) if im.size != size else im.convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="dir of the original multi-view input images")
    ap.add_argument("--maps_dir", required=True, help="dir of MVInverse output maps ({idx}_{map}.png)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cols", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # frame ids from albedo maps, sorted
    ids = sorted(
        m.group(1)
        for f in os.listdir(args.maps_dir)
        if (m := re.match(r"(\d+)_albedo\.png$", f))
    )
    inputs = sorted(f for f in os.listdir(args.input_dir) if f.lower().endswith(IMG_EXT))

    print(f"[verify] frames(maps)={len(ids)} input_images={len(inputs)}", flush=True)
    if len(ids) != len(inputs):
        print(f"[verify][WARN] count mismatch: {len(ids)} maps vs {len(inputs)} inputs; "
              f"pairing the first {min(len(ids), len(inputs))} by sorted order.", flush=True)
    n_pair = min(len(ids), len(inputs))

    panels_order = ["Input", "Reconstructed (A×D)", "Albedo", "Shading", "Normal", "Metallic", "Roughness"]

    for k in range(n_pair):
        idx = ids[k]
        maps = {m: Image.open(os.path.join(args.maps_dir, f"{idx}_{m}.png")) for m in MAPS}
        W, H = maps["albedo"].size  # model/map resolution

        in_name = inputs[k]
        in_im = Image.open(os.path.join(args.input_dir, in_name))
        in_native = in_im.size
        inp = to_rgb(in_im, (W, H))  # <- input scaling to map resolution

        albedo = np.asarray(to_rgb(maps["albedo"], (W, H)), np.float32) / 255.0
        shading = np.asarray(to_rgb(maps["shading"], (W, H)), np.float32) / 255.0
        inp_np = np.asarray(inp, np.float32) / 255.0
        # Diffuse recomposition albedo (.) shading. Albedo is scale-invariant, so
        # solve the per-channel scale in closed form (least squares to the input,
        # the paper's s*) so the recomposition matches the input's exposure.
        raw = albedo * shading
        scale = (inp_np * raw).sum((0, 1)) / ((raw * raw).sum((0, 1)) + 1e-6)
        recon_np = np.clip(raw * scale, 0, 1)
        recon = Image.fromarray((recon_np * 255).astype(np.uint8))

        panels = {
            "Input": inp,
            "Reconstructed (A×D)": recon,
            "Albedo": to_rgb(maps["albedo"], (W, H)),
            "Shading": to_rgb(maps["shading"], (W, H)),
            "Normal": to_rgb(maps["normal"], (W, H)),
            "Metallic": to_rgb(maps["metallic"], (W, H)),
            "Roughness": to_rgb(maps["roughness"], (W, H)),
        }

        font = find_font(max(14, H // 18))
        cols = args.cols
        rows = (len(panels_order) + cols - 1) // cols
        montage = Image.new("RGB", (cols * W, rows * H), (40, 40, 40))
        for i, name in enumerate(panels_order):
            p = panels[name].copy()
            label(p, name, font)
            r, c = divmod(i, cols)
            montage.paste(p, (c * W, r * H))

        out_path = os.path.join(args.out_dir, f"{idx}_summary.png")
        montage.save(out_path)
        print(f"[verify] frame {idx}: map={W}x{H}  input '{in_name}' native={in_native[0]}x{in_native[1]} "
              f"-> resized={W}x{H}  recon scale(RGB)=[{scale[0]:.2f},{scale[1]:.2f},{scale[2]:.2f}] "
              f"recon[min/mean/max]={recon_np.min():.3f}/{recon_np.mean():.3f}/{recon_np.max():.3f}  -> {out_path}",
              flush=True)

    print(f"[verify] wrote {n_pair} summaries to {args.out_dir} (grid {args.cols} cols x "
          f"{(len(panels_order)+args.cols-1)//args.cols} rows)", flush=True)


if __name__ == "__main__":
    main()
