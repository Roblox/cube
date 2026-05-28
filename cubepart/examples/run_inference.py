"""End-to-end inference example for cube_part.

Given an input mesh and a list of part names, encode the mesh to a latent
with the shape VAE, then run the multi-part diffusion pipeline to obtain
one mesh per part.

Download the pretrained weights first (see top-level README), e.g.::

    huggingface-cli download Roblox/cubepart --local-dir weights

Usage::

    python examples/run_inference.py \
        --config configs/shape_denoiser_multimesh.yaml \
        --checkpoint weights/multi_part_dit.safetensors \
        --vae-checkpoint weights/vae.safetensors \
        --mesh examples/inputs/jellyfish_car.glb \
        --parts "body, front right wheel, front left wheel, rear right wheel, rear left wheel, exhaust pipe, headlights, gun" \
        --output outputs/jellyfish_car_parts
"""

from __future__ import annotations

import argparse
import colorsys
import os

import numpy as np
import torch
import trimesh

from cube_part.pipelines import PartShapeDenoiserPipeline, ShapeInput
from cube_part.utils.mesh import load_mesh, sample_surface


def _palette(n: int) -> np.ndarray:
    """Return `n` visually-distinct RGB colors as uint8 in [0, 255]."""
    colors = []
    for i in range(max(n, 1)):
        h = (i / max(n, 1)) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.55, 0.95)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return np.array(colors, dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="cube_part parts inference example")
    parser.add_argument("--config", required=True, help="multi-mesh YAML config")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Path to the multi-part DiT checkpoint "
            "(e.g. `weights/multi_part_dit.safetensors`). "
            "Run `huggingface-cli download Roblox/cubepart --local-dir weights` "
            "to fetch the pretrained weights."
        ),
    )
    parser.add_argument(
        "--vae-checkpoint",
        default=None,
        help=(
            "Optional override for the shape VAE checkpoint "
            "(e.g. `weights/vae.safetensors`). If omitted, the path from the "
            "YAML config is used."
        ),
    )
    parser.add_argument("--mesh", required=True, help="path to the input mesh")
    parser.add_argument(
        "--parts",
        required=True,
        type=lambda s: [p.strip() for p in s.split(",") if p.strip()],
        help=(
            "Comma-separated part names to segment, "
            'e.g. "seat, backrest, legs"'
        ),
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="directory to write the per-part meshes",
    )
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--resolution-base", type=float, default=8.5)
    parser.add_argument(
        "--scheduler", default="dpm_solver", choices=["euler", "heun", "dpm_solver"]
    )
    parser.add_argument("--timeshift", type=float, default=4.0)
    parser.add_argument("--num-samples", type=int, default=128_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipe = PartShapeDenoiserPipeline(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        vae_checkpoint_path=args.vae_checkpoint,
        device=args.device,
        extract_geometry_fn_name="extract_geometry_coarse_to_fine",
    )

    mesh, _, _ = load_mesh(args.mesh)
    surface = sample_surface(mesh, num_samples=args.num_samples)
    surface = (
        torch.from_numpy(surface).to(pipe.device).unsqueeze(0).float()
    )
    latents, _ = pipe.encode_shape(surface)

    os.makedirs(args.output, exist_ok=True)

    part_meshes = pipe.input_to_part_shape(
        ShapeInput(prompt=[args.parts], latents=latents),
        guidance_scale=args.guidance_scale,
        resolution_base=args.resolution_base,
        scheduler_type=args.scheduler,
        timeshift=args.timeshift,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        output_mesh=True
    )

    palette = _palette(len(args.parts))
    scene = trimesh.Scene()
    for i, (verts, faces) in enumerate(part_meshes):
        if verts is None:
            continue
        safe_name = args.parts[i].replace(" ", "_")
        submesh = trimesh.Trimesh(verts, faces)
        out = os.path.join(args.output, f"part_{i:02d}_{safe_name}.glb")
        submesh.export(out)
        print(f"Saved part {i}: {out}")

        colored = submesh.copy()
        colored.visual.face_colors = palette[i % len(palette)]
        scene.add_geometry(colored, geom_name=f"part_{i:02d}_{safe_name}")

    if len(scene.geometry) > 0:
        combined_out = os.path.join(args.output, "parts.glb")
        scene.export(combined_out)
        print(f"Saved combined parts scene: {combined_out}")


if __name__ == "__main__":
    main()
