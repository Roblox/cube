"""Gradio demo for the multi-mesh shape denoiser.

Launches a local Gradio app where the user can upload a `.glb` mesh, type a
list of part names, and run the same inference pipeline as
`examples/run_inference.py` / `test.ipynb`. The resulting per-part meshes are
assembled into a single colored `trimesh.Scene` and rendered with
`gr.Model3D`.

Usage:

    python examples/gradio_demo.py \
        --config configs/shape_denoiser_multimesh.yaml \
        --checkpoint /path/to/multimesh_checkpoint.pt

The pipeline is loaded once at startup; each request only runs the
encode/denoise/decode passes.
"""

from __future__ import annotations

import argparse
import colorsys
import glob
import json
import logging
import os
import random
import tempfile
import traceback
from typing import List, Optional, Sequence, Tuple

import gradio as gr
import numpy as np
import torch
import trimesh

from cube_part.pipelines import PartShapeDenoiserPipeline, ShapeInput
from cube_part.utils.mesh import load_mesh, sample_surface

logger = logging.getLogger("cube_part.gradio_demo")

MAX_PARTS = 8

DEFAULT_EXAMPLES_JSON = "examples/inputs/examples.json"
DEFAULT_EXAMPLES_RENDER_DIR = "examples/inputs"

DEFAULT_HF_REPO_ID = "Roblox/cubepart"
DEFAULT_HF_DIT_FILENAME = "multi_part_dit.safetensors"
DEFAULT_HF_VAE_FILENAME = "vae.safetensors"
DEFAULT_HF_LOCAL_DIR = "weights"

DEFAULT_PARTS = (
    "body\n"
    "left front wheel\n"
    "right front wheel\n"
    "left rear wheel\n"
    "right rear wheel\n"
)


def _is_lfs_pointer(path: str) -> bool:
    """Return True if `path` looks like a Git LFS pointer rather than a real
    binary (GLB).

    LFS pointer files are tiny (< 1 KiB) and begin with the literal line
    ``version https://git-lfs.github.com/spec/v1``. Treating them as real
    meshes silently breaks downstream loaders, so the demo skips them with
    a clear warning instead.
    """
    try:
        if os.path.getsize(path) > 1024:
            return False
        with open(path, "rb") as f:
            head = f.read(64)
        return head.startswith(b"version https://git-lfs.github.com/spec")
    except OSError:
        return False


def _find_render_image(stem: str, renders_dir: Optional[str]) -> Optional[str]:
    """Locate a thumbnail render image for an example mesh.

    Matches, in order:
      1. ``{renders_dir}/{stem}.png|.jpg|.jpeg|.webp``
      2. ``{renders_dir}/render_{stem}_*.png|.jpg|.jpeg|.webp``
      3. ``{renders_dir}/{stem}_*.png|.jpg|.jpeg|.webp``

    Returns the first hit (sorted lexicographically when multiple match) so
    the renders directory can keep its current ``render_<stem>_<timestamp>``
    naming and we still pick a deterministic file.
    """
    if not renders_dir or not os.path.isdir(renders_dir):
        return None
    exts = ("png", "jpg", "jpeg", "webp")
    for ext in exts:
        direct = os.path.join(renders_dir, f"{stem}.{ext}")
        if os.path.exists(direct):
            return direct
    for pattern in (f"render_{stem}_*", f"{stem}_*"):
        for ext in exts:
            matches = sorted(glob.glob(os.path.join(renders_dir, f"{pattern}.{ext}")))
            if matches:
                return matches[0]
    return None


def _load_examples(
    examples_json: Optional[str],
    renders_dir: Optional[str] = None,
) -> List[Tuple[str, str, Optional[str]]]:
    """Load `(mesh_path, parts_text, render_path)` triples from a JSON manifest.

    The manifest maps a mesh filename (resolved relative to the JSON's
    directory) to a list of part names, e.g.::

        {
          "robot.glb": ["body", "left arm", "right arm"]
        }

    A thumbnail render is auto-discovered under ``renders_dir`` via
    :func:`_find_render_image`; ``None`` is returned in the triple when no
    matching image exists.

    Entries whose mesh file does not exist are skipped with a warning so the
    demo still launches when only a subset of meshes are checked in.
    """
    if not examples_json:
        return []
    if not os.path.exists(examples_json):
        logger.warning("Examples JSON %r not found; skipping examples.", examples_json)
        return []

    try:
        with open(examples_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to parse examples JSON %r: %s", examples_json, exc)
        return []

    if not isinstance(raw, dict):
        logger.warning(
            "Examples JSON %r must be an object mapping mesh filename -> "
            "list of part names; got %s.",
            examples_json,
            type(raw).__name__,
        )
        return []

    base_dir = os.path.dirname(os.path.abspath(examples_json))
    resolved_renders_dir = (
        os.path.abspath(renders_dir) if renders_dir else None
    )
    triples: List[Tuple[str, str, Optional[str]]] = []
    for mesh_name, parts in raw.items():
        if not isinstance(parts, (list, tuple)) or not all(
            isinstance(p, str) for p in parts
        ):
            logger.warning(
                "Skipping example %r: parts must be a list of strings.", mesh_name
            )
            continue
        mesh_path = (
            mesh_name if os.path.isabs(mesh_name) else os.path.join(base_dir, mesh_name)
        )
        if not os.path.exists(mesh_path):
            logger.warning(
                "Skipping example %r: mesh file %r not found.", mesh_name, mesh_path
            )
            continue
        if _is_lfs_pointer(mesh_path):
            logger.warning(
                "Skipping example %r: %r is a Git LFS pointer, not a real "
                "GLB. Run `git lfs install && git lfs pull` in the repo to "
                "fetch the actual mesh binaries.",
                mesh_name,
                mesh_path,
            )
            continue
        stem = os.path.splitext(os.path.basename(mesh_name))[0]
        render_path = _find_render_image(stem, resolved_renders_dir)
        if resolved_renders_dir and render_path is None:
            logger.warning(
                "No render thumbnail found for example %r under %r.",
                mesh_name,
                resolved_renders_dir,
            )
        triples.append((mesh_path, ",\n".join(parts), render_path))

    return triples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-mesh denoiser Gradio demo")
    parser.add_argument(
        "--config",
        default="configs/shape_denoiser_multimesh.yaml",
        help="Multi-mesh YAML config",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Path to the multi-mesh checkpoint (.pt / .safetensors). "
            "If omitted, the checkpoint is downloaded from the Hugging Face "
            f"repo {DEFAULT_HF_REPO_ID!r} into --hf-local-dir."
        ),
    )
    parser.add_argument(
        "--vae-checkpoint",
        default=None,
        help=(
            "Optional override for the shape VAE checkpoint (.safetensors). "
            "If --checkpoint is also omitted, the VAE is downloaded from the "
            "same Hugging Face repo. Otherwise the path from the YAML config "
            "is used."
        ),
    )
    parser.add_argument(
        "--hf-repo-id",
        default=DEFAULT_HF_REPO_ID,
        help="Hugging Face model repo to fall back to when --checkpoint is omitted.",
    )
    parser.add_argument(
        "--hf-dit-filename",
        default=DEFAULT_HF_DIT_FILENAME,
        help="Filename of the multi-part DiT checkpoint inside the HF repo.",
    )
    parser.add_argument(
        "--hf-vae-filename",
        default=DEFAULT_HF_VAE_FILENAME,
        help="Filename of the shape VAE checkpoint inside the HF repo.",
    )
    parser.add_argument(
        "--hf-local-dir",
        default=DEFAULT_HF_LOCAL_DIR,
        help="Local directory to snapshot-download HF weights into.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio share link"
    )
    parser.add_argument(
        "--examples-json",
        default=DEFAULT_EXAMPLES_JSON,
        help=(
            "Path to a JSON file mapping mesh filename -> list of part names. "
            "Mesh paths are resolved relative to the JSON's directory. "
            "Pass an empty string to disable examples."
        ),
    )
    parser.add_argument(
        "--examples-render-dir",
        default=DEFAULT_EXAMPLES_RENDER_DIR,
        help=(
            "Directory containing thumbnail renders shown in the examples "
            "gallery. For each mesh, the demo looks for either "
            "`{stem}.png` (or .jpg/.jpeg/.webp) or `render_{stem}_*.{ext}`. "
            "Pass an empty string to disable thumbnails."
        ),
    )
    return parser.parse_args()


def _palette(n: int) -> np.ndarray:
    """Return `n` visually-distinct RGB colors as uint8 in [0, 255]."""
    colors = []
    for i in range(max(n, 1)):
        h = (i / max(n, 1)) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.55, 0.95)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return np.array(colors, dtype=np.uint8)


def _parse_parts(raw: str) -> List[str]:
    """Split user-entered part names (newline or comma separated)."""
    if raw is None:
        return []
    parts: List[str] = []
    for line in raw.replace(",", "\n").splitlines():
        name = line.strip()
        if name:
            parts.append(name)
    return parts


def _scene_to_glb(scene: trimesh.Scene) -> str:
    out_dir = tempfile.mkdtemp(prefix="cube_part_demo_")
    out_path = os.path.join(out_dir, "parts.glb")
    scene.export(out_path)
    return out_path


def _legend_html(
    names: Sequence[str],
    colors: np.ndarray,
    kept_mask: Sequence[bool],
) -> str:
    """Render a colored-swatch legend mapping part name -> color."""
    rows = []
    for name, color, kept in zip(names, colors, kept_mask):
        r, g, b = int(color[0]), int(color[1]), int(color[2])
        swatch = (
            f"<span style='display:inline-block;width:18px;height:18px;"
            f"border-radius:4px;background:rgb({r},{g},{b});"
            f"border:1px solid rgba(255,255,255,0.15);margin-right:8px;"
            f"vertical-align:middle;box-shadow:0 0 0 1px rgba(0,0,0,0.4);'></span>"
        )
        if kept:
            label = (
                f"<span style='vertical-align:middle;color:#e5e7eb;'>{name}</span>"
            )
        else:
            label = (
                f"<span style='vertical-align:middle;opacity:0.45;color:#e5e7eb;"
                f"text-decoration:line-through;' title='no geometry'>"
                f"{name}</span>"
            )
        rows.append(
            f"<div style='display:flex;align-items:center;"
            f"margin:4px 12px 4px 0;'>{swatch}{label}</div>"
        )
    return (
        "<div style='display:flex;flex-wrap:wrap;"
        "font-family:system-ui,sans-serif;font-size:14px;color:#e5e7eb;'>"
        + "".join(rows)
        + "</div>"
    )


def _message_html(msg: str) -> str:
    return (
        "<div style='font-family:system-ui,sans-serif;font-size:14px;"
        f"color:#cbd5e1;opacity:0.85;'>{msg}</div>"
    )


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def _build_runner(pipe: PartShapeDenoiserPipeline):
    """Return a closure that Gradio can call with UI values."""

    def run(
        mesh_path: Optional[str],
        parts_text: str,
        guidance_scale: float,
        num_inference_steps: int,
        resolution_base: float,
        num_samples: int,
        chunk_size: int,
        seed: int,
    ) -> Tuple[Optional[str], str]:
        if mesh_path is None or not os.path.exists(mesh_path):
            return None, _message_html("Please upload a `.glb` mesh.")

        parts = _parse_parts(parts_text)
        if not parts:
            return None, _message_html("Please enter at least one part name.")
        if len(parts) > MAX_PARTS:
            return None, _message_html(
                f"Too many parts ({len(parts)}). Maximum is {MAX_PARTS}."
            )

        try:
            mesh, _, _ = load_mesh(mesh_path)
            _seed_everything(int(seed))
            surface = sample_surface(mesh, num_samples=int(num_samples))
            surface = (
                torch.from_numpy(surface)
                .to(pipe.device)
                .unsqueeze(0)
                .float()
            )
            latents, _ = pipe.encode_shape(surface)

            part_meshes = pipe.input_to_part_shape(
                ShapeInput(prompt=[parts], latents=latents),
                guidance_scale=float(guidance_scale),
                num_inference_steps=int(num_inference_steps),
                resolution_base=float(resolution_base),
                chunk_size=int(chunk_size),
                seed=int(seed),
                timeshift=4.0,
                scheduler_type="dpm_solver"
            )

            palette = _palette(len(parts))
            scene = trimesh.Scene()
            kept_mask: List[bool] = []
            for i, (vertices, faces) in enumerate(part_meshes):
                name = parts[i] if i < len(parts) else f"part_{i}"
                if (
                    vertices is not None
                    and faces is not None
                    and vertices.shape[0] > 0
                    and faces.shape[0] > 0
                ):
                    submesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                    submesh.visual.face_colors = palette[i % len(palette)]
                    scene.add_geometry(submesh, geom_name=f"part_{i}_{name}")
                    kept_mask.append(True)
                else:
                    kept_mask.append(False)

            if len(scene.geometry) == 0:
                return None, _message_html("No parts produced any geometry.")

            out_path = _scene_to_glb(scene)
            return out_path, _legend_html(parts, palette, kept_mask)
        except Exception:
            return None, _message_html(
                "<pre style='white-space:pre-wrap;'>"
                f"Inference failed:\n{traceback.format_exc()}"
                "</pre>"
            )

    return run


DARK_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=(gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"),
).set(
    body_background_fill="#0b0f17",
    body_background_fill_dark="#0b0f17",
    background_fill_primary="#111827",
    background_fill_primary_dark="#111827",
    background_fill_secondary="#0f172a",
    background_fill_secondary_dark="#0f172a",
    block_background_fill="#111827",
    block_background_fill_dark="#111827",
    block_border_color="#1f2937",
    block_border_color_dark="#1f2937",
    block_label_background_fill="#0f172a",
    block_label_background_fill_dark="#0f172a",
    block_label_text_color="#e5e7eb",
    block_label_text_color_dark="#e5e7eb",
    block_title_text_color="#f1f5f9",
    block_title_text_color_dark="#f1f5f9",
    body_text_color="#e5e7eb",
    body_text_color_dark="#e5e7eb",
    body_text_color_subdued="#94a3b8",
    body_text_color_subdued_dark="#94a3b8",
    border_color_accent="#1f2937",
    border_color_accent_dark="#1f2937",
    border_color_primary="#1f2937",
    border_color_primary_dark="#1f2937",
    input_background_fill="#0f172a",
    input_background_fill_dark="#0f172a",
    input_border_color="#1f2937",
    input_border_color_dark="#1f2937",
    button_primary_background_fill="#2563eb",
    button_primary_background_fill_dark="#2563eb",
    button_primary_background_fill_hover="#1d4ed8",
    button_primary_background_fill_hover_dark="#1d4ed8",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="#1f2937",
    button_secondary_background_fill_dark="#1f2937",
    button_secondary_text_color="#e5e7eb",
    button_secondary_text_color_dark="#e5e7eb",
)

FORCE_DARK_JS_TEMPLATE = """
function refresh() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
        return;
    }
    const meshPaths = __MESH_PATHS__;
    if (!meshPaths || !meshPaths.length) return;
    // Conservative prefetch: probe once for the correct Gradio file-serving
    // prefix, then walk meshPaths sequentially with a small delay so we
    // never saturate the browser's per-host socket pool (~6) or race the
    // user's first click. Responses land in the HTTP cache, making the
    // subsequent example click materially faster.
    const probe = async () => {
        for (const prefix of ['/gradio_api/file=', '/file=']) {
            try {
                const r = await fetch(prefix + meshPaths[0],
                                      {method: 'HEAD', credentials: 'same-origin'});
                if (r.ok) return prefix;
            } catch (_) {}
        }
        return null;
    };
    const run = async () => {
        await new Promise(r => setTimeout(r, 3000));
        const prefix = await probe();
        if (!prefix) return;
        for (const p of meshPaths) {
            try {
                await fetch(prefix + p, {credentials: 'same-origin'});
            } catch (_) {}
            await new Promise(r => setTimeout(r, 250));
        }
    };
    run();
}
"""


def _build_force_dark_js(example_paths: Sequence[str]) -> str:
    """Inject the example mesh paths into the page-load JS for prefetching."""
    return FORCE_DARK_JS_TEMPLATE.replace(
        "__MESH_PATHS__", json.dumps(list(example_paths))
    )

DARK_CSS = """
.gradio-container { background: #0b0f17 !important; }
footer { display: none !important; }

.prose code,
.markdown code,
.gradio-container code:not(pre code) {
    background: #1f2937 !important;
    color: #93c5fd !important;
    border: 1px solid #334155 !important;
    border-radius: 4px !important;
    padding: 1px 6px !important;
    font-size: 0.9em !important;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;
}

.prose pre,
.markdown pre,
.gradio-container pre {
    background: #0f172a !important;
    color: #e5e7eb !important;
    border: 1px solid #1f2937 !important;
    border-radius: 6px !important;
    padding: 10px 12px !important;
}
.prose pre code,
.markdown pre code,
.gradio-container pre code {
    background: transparent !important;
    color: inherit !important;
    border: none !important;
    padding: 0 !important;
}

/* gr.Examples — dark-theme overrides, scoped to our explicit elem_id so
   this is robust to Gradio's internal class-name churn. */
#cubepart_examples,
#cubepart_examples * {
    background-color: transparent !important;
    color: #e5e7eb !important;
    border-color: #1f2937 !important;
}
#cubepart_examples {
    background-color: #111827 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 6px !important;
    padding: 8px !important;
}
#cubepart_examples table {
    background-color: #111827 !important;
    border-collapse: collapse !important;
    width: 100% !important;
}
#cubepart_examples thead,
#cubepart_examples thead tr,
#cubepart_examples thead th {
    background-color: #0f172a !important;
    color: #f1f5f9 !important;
    border-color: #1f2937 !important;
    font-weight: 600 !important;
}
#cubepart_examples tbody tr,
#cubepart_examples tbody td {
    background-color: #111827 !important;
    color: #e5e7eb !important;
    border-color: #1f2937 !important;
}
#cubepart_examples tbody tr:nth-child(even),
#cubepart_examples tbody tr:nth-child(even) td {
    background-color: #0f172a !important;
}
#cubepart_examples tbody tr:hover,
#cubepart_examples tbody tr:hover td {
    background-color: #1e293b !important;
    cursor: pointer !important;
}
#cubepart_examples button {
    background-color: #1f2937 !important;
    color: #e5e7eb !important;
    border: 1px solid #334155 !important;
}
#cubepart_examples button:hover {
    background-color: #334155 !important;
}
/* Some Gradio versions wrap cell text in <span> / <div> with their own bg. */
#cubepart_examples td span,
#cubepart_examples td div,
#cubepart_examples th span,
#cubepart_examples th div {
    background-color: transparent !important;
    color: inherit !important;
}

/* Examples gallery — dark-theme thumbnails with hover/selected highlight. */
#cubepart_examples_gallery {
    background-color: #111827 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 6px !important;
    padding: 6px !important;
    max-height: none !important;
    min-height: 0 !important;
    height: auto !important;
    overflow: visible !important;
}
#cubepart_examples_gallery .grid-wrap,
#cubepart_examples_gallery .preview {
    background-color: transparent !important;
    max-height: none !important;
    min-height: 0 !important;
    height: auto !important;
    overflow: visible !important;
    padding: 0 !important;
}
/* Strip any min-height Gradio applies to the gallery's internal wrappers,
   so the panel shrink-wraps the single row of thumbnails. */
#cubepart_examples,
#cubepart_examples > div,
#cubepart_examples_gallery,
#cubepart_examples_gallery > div {
    min-height: 0 !important;
    height: auto !important;
}
/* Override Gradio's `grid-template-columns: repeat(N, minmax(0, 1fr))` (which
   stretches items evenly across the row) with auto-fill at a fixed width so
   items pack from the left and wrap onto new rows when needed. */
#cubepart_examples_gallery .grid-container {
    display: grid !important;
    grid-template-columns: repeat(auto-fill, 140px) !important;
    grid-auto-rows: 140px !important;
    justify-content: start !important;
    gap: 8px !important;
}
/* Each direct child of the grid is a thumbnail wrapper; size it explicitly
   so the inner img/canvas fills exactly the 140x140 slot. */
#cubepart_examples_gallery .grid-container > * {
    width: 140px !important;
    height: 140px !important;
}
#cubepart_examples_gallery .thumbnail-item,
#cubepart_examples_gallery .thumbnail-small {
    background-color: #0f172a !important;
    border: 1px solid #1f2937 !important;
    border-radius: 6px !important;
    overflow: hidden !important;
    width: 140px !important;
    height: 140px !important;
    transition: transform 0.12s ease, border-color 0.12s ease,
                box-shadow 0.12s ease !important;
    cursor: pointer !important;
}
#cubepart_examples_gallery .thumbnail-item img,
#cubepart_examples_gallery .thumbnail-small img {
    width: 100% !important;
    height: 100% !important;
    object-fit: cover !important;
}
#cubepart_examples_gallery .thumbnail-item:hover,
#cubepart_examples_gallery .thumbnail-small:hover {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.35) !important;
    transform: translateY(-1px);
}
#cubepart_examples_gallery .thumbnail-item.selected,
#cubepart_examples_gallery .thumbnail-small.selected {
    border-color: #60a5fa !important;
    box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.55) !important;
}
#cubepart_examples_gallery .caption {
    background: rgba(15, 23, 42, 0.85) !important;
    color: #e5e7eb !important;
    font-size: 12px !important;
    padding: 2px 6px !important;
}
"""


def build_ui(
    pipe: PartShapeDenoiserPipeline,
    examples: Sequence[Tuple[str, str, Optional[str]]] = (),
) -> gr.Blocks:
    run = _build_runner(pipe)

    title = "CubePart: An Open-Vocabulary Part-Controllable 3D Generator"
    with gr.Blocks(title=title, fill_width=True) as demo:
        gr.Markdown(
            f"""
            # {title}

            [Project Page](https://cubepart.github.io/) &nbsp;|&nbsp;
            [GitHub Repo]() &nbsp;|&nbsp;
            [Paper](https://cubepart.github.io/)

            ### About this demo
            This is a demo of multi-part mesh generation of CubePart.
            Upload a `.glb` mesh, list the part names you want, and the
            CubePart model will decomposes the input mesh into the specified parts.

            ### Guidelines
            To get optimal results, please follow the guidelines below:

            1. The input mesh should be **canonically aligned** (+Y up, +Z forward).
            2. A watertight, single-surface mesh is preferred. Meshes with duplicated inner/outer shells (common in some AI-generated meshes) often degrade quality.
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Group():
                    input_mesh = gr.Model3D(
                        label="Input mesh (.glb)",
                        clear_color=[0.043, 0.059, 0.090, 1.0],
                        height="28em",
                    )
                    parts_box = gr.Textbox(
                        label="Part names (one per line or comma-separated, max 8)",
                        value=DEFAULT_PARTS,
                        lines=6,
                    )
                    with gr.Accordion("Sampling settings", open=False):
                        guidance_scale = gr.Slider(
                            1.0, 15.0, value=7.5, step=0.1, label="Guidance scale"
                        )
                        num_inference_steps = gr.Slider(
                            1, 100, value=50, step=1, label="Diffusion steps"
                        )
                        resolution_base = gr.Slider(
                            6.0,
                            10.0,
                            value=8.5,
                            step=0.5,
                            label="Marching-cubes log2 resolution",
                        )
                        num_samples = gr.Slider(
                            16_000,
                            256_000,
                            value=128_000,
                            step=16_000,
                            label="Surface samples for encoding",
                        )
                        chunk_size = gr.Slider(
                            1024,
                            65_536,
                            value=8192,
                            step=1024,
                            label="Decode chunk size",
                        )
                        seed = gr.Number(value=42, precision=0, label="Seed")
                with gr.Row():
                    run_btn = gr.Button("Run inference", variant="primary")

            with gr.Column(scale=3):
                output_mesh = gr.Model3D(
                    label="Colored part scene (.glb)",
                    clear_color=[0.043, 0.059, 0.090, 1.0],
                    height="45em",
                    interactive=False,
                )
                legend = gr.HTML(label="Legend")

        if examples:
            gallery_items: List[Tuple[str, str]] = []
            example_payloads: List[Tuple[str, str]] = []
            for mesh_path, parts_text, render_path in examples:
                stem = os.path.splitext(os.path.basename(mesh_path))[0]
                thumb = render_path if render_path else mesh_path
                gallery_items.append((thumb, stem))
                example_payloads.append((mesh_path, parts_text))

            with gr.Group(elem_id="cubepart_examples"):
                gr.Markdown("### Examples — click a thumbnail to load it")
                examples_gallery = gr.Gallery(
                    value=gallery_items,
                    label=None,
                    show_label=False,
                    columns=8,
                    object_fit="cover",
                    height="auto",
                    allow_preview=False,
                    elem_id="cubepart_examples_gallery",
                )

            def _on_example_select(
                evt: gr.SelectData,
            ) -> Tuple[str, str]:
                idx = int(evt.index) if evt.index is not None else 0
                idx = max(0, min(idx, len(example_payloads) - 1))
                mesh_path, parts_text = example_payloads[idx]
                return mesh_path, parts_text

            examples_gallery.select(
                _on_example_select,
                outputs=[input_mesh, parts_box],
            )

        run_btn.click(
            run,
            inputs=[
                input_mesh,
                parts_box,
                guidance_scale,
                num_inference_steps,
                resolution_base,
                num_samples,
                chunk_size,
                seed,
            ],
            outputs=[output_mesh, legend],
        )

    return demo


def _resolve_checkpoints(args: argparse.Namespace) -> Tuple[str, Optional[str]]:
    """Return `(checkpoint_path, vae_checkpoint_path)`.

    If the user passed `--checkpoint`, honor local paths exactly as before.
    Otherwise, snapshot-download the configured HF model repo (default:
    `Roblox/cubepart`) into `--hf-local-dir` and resolve the two filenames.
    """
    if args.checkpoint:
        return args.checkpoint, args.vae_checkpoint

    from huggingface_hub import snapshot_download

    print(
        f"--checkpoint not provided; downloading weights from "
        f"Hugging Face repo {args.hf_repo_id!r} into {args.hf_local_dir!r} ..."
    )
    local_dir = snapshot_download(
        repo_id=args.hf_repo_id,
        local_dir=args.hf_local_dir,
        allow_patterns=[args.hf_dit_filename, args.hf_vae_filename],
    )
    ckpt = os.path.join(local_dir, args.hf_dit_filename)
    vae = os.path.join(local_dir, args.hf_vae_filename)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"Expected DiT checkpoint at {ckpt!r} after snapshot_download "
            f"from {args.hf_repo_id!r}."
        )
    if not os.path.exists(vae):
        raise FileNotFoundError(
            f"Expected VAE checkpoint at {vae!r} after snapshot_download "
            f"from {args.hf_repo_id!r}."
        )
    print(f"Resolved DiT checkpoint: {ckpt}")
    print(f"Resolved VAE checkpoint: {vae}")
    return ckpt, vae


def main() -> None:
    args = parse_args()

    checkpoint_path, vae_checkpoint_path = _resolve_checkpoints(args)

    print(f"Loading PartShapeDenoiserPipeline from {checkpoint_path} ...")
    pipe = PartShapeDenoiserPipeline(
        config_path=args.config,
        checkpoint_path=checkpoint_path,
        vae_checkpoint_path=vae_checkpoint_path,
        device=args.device,
        extract_geometry_fn_name="extract_geometry_coarse_to_fine",
    )
    print("Pipeline ready.")

    examples = _load_examples(
        args.examples_json or None,
        args.examples_render_dir or None,
    )
    if examples:
        n_with_thumb = sum(1 for _, _, r in examples if r)
        print(
            f"Loaded {len(examples)} example(s) from {args.examples_json} "
            f"({n_with_thumb} with thumbnails from {args.examples_render_dir!r})."
        )
    else:
        print(
            "No examples loaded "
            f"(examples_json={args.examples_json!r}); demo will start without examples."
        )

    demo = build_ui(pipe, examples)

    # Conservative prefetch of example meshes after page load: one request at
    # a time, single URL prefix, kicks in only after the page is idle. Tries
    # the Gradio-4 prefix first; falls back to legacy on 404. Intentionally
    # avoids saturating the browser's per-host socket pool (cap ~6) and never
    # races against the user's first click.
    #
    # NOTE: We deliberately do *not* add server-side gzip middleware here:
    # GZipMiddleware breaks Range requests, which `gr.Model3D` (three.js
    # GLTFLoader) uses to stream .glb files. Enabling it causes example
    # clicks to hang on the input viewer. GLB binary buffers compress poorly
    # anyway, so the cost/risk wasn't worth it.
    example_paths = [os.path.abspath(p) for p, _, _ in examples]
    force_dark_js = _build_force_dark_js(example_paths)

    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        theme=DARK_THEME,
        css=DARK_CSS,
        js=force_dark_js,
    )


if __name__ == "__main__":
    main()
