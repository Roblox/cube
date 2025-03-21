import argparse
import os

import torch
import trimesh

from cube3d.inference.engine import Engine, EngineFast
from cube3d.mesh_utils.postprocessing import (
    PYMESHLAB_AVAILABLE,
    create_pymeshset,
    postprocess_mesh,
    save_mesh,
)
from cube3d.renderer import renderer

def generate_mesh(
    engine,
    prompt,
    output_dir,
    output_name,
    resolution_base=8.0,
    disable_postprocess=False,
    top_p=None,
):
    mesh_v_f = engine.t2s(
        [prompt],
        use_kv_cache=True,
        resolution_base=resolution_base,
        top_p=top_p,
    )
    vertices, faces = mesh_v_f[0][0], mesh_v_f[0][1]
    obj_path = os.path.join(output_dir, f"{output_name}.obj")
    if PYMESHLAB_AVAILABLE:
        ms = create_pymeshset(vertices, faces)
        if not disable_postprocess:
            target_face_num = max(10000, int(faces.shape[0] * 0.1))
            print(f"Postprocessing mesh to {target_face_num} faces")
            postprocess_mesh(ms, target_face_num, obj_path)

        save_mesh(ms, obj_path)
    else:
        print(
            "WARNING: pymeshlab is not available, using trimesh to export obj and skipping optional post processing."
        )
        mesh = trimesh.Trimesh(vertices, faces)
        mesh.export(obj_path)

    return obj_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cube shape generation script")
    parser.add_argument(
        "--config-path",
        type=str,
        default="cube3d/configs/open_model.yaml",
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/",
        help="Path to the output directory to store .obj and .gif files",
    )
    parser.add_argument(
        "--gpt-ckpt-path",
        type=str,
        required=True,
        help="Path to the main GPT checkpoint file.",
    )
    parser.add_argument(
        "--shape-ckpt-path",
        type=str,
        required=True,
        help="Path to the shape encoder/decoder checkpoint file.",
    )
    parser.add_argument(
        "--fast-inference",
        help="Use optimized inference",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for generating a 3D mesh",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Float < 1: Keep smallest set of tokens with cumulative probability ≥ top_p. Default None: deterministic generation.",
    )
    parser.add_argument(
        "--render-gif",
        help="Render a turntable gif of the mesh",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--disable-postprocessing",
        help="Disable postprocessing on the mesh. This will result in a mesh with more faces.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--resolution-base",
        type=float,
        default=8.0,
        help="Resolution base for the shape decoder.",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    # Initialize engine based on fast_inference flag
    if args.fast_inference:
        print(
            "Using cuda graphs, this will take some time to warmup and capture the graph."
        )
        engine = EngineFast(
            args.config_path, args.gpt_ckpt_path, args.shape_ckpt_path, device=device
        )
        print("Compiled the graph.")
    else:
        engine = Engine(
            args.config_path, args.gpt_ckpt_path, args.shape_ckpt_path, device=device
        )
    
    # Generate meshes based on input source
    obj_path = generate_mesh(
        engine,
        args.prompt,
        args.output_dir,
        "output",
        args.resolution_base,
        args.disable_postprocessing,
        args.top_p,
    )
    if args.render_gif:
        gif_path = renderer.render_turntable(obj_path, args.output_dir)
        print(f"Rendered turntable gif for {args.prompt} at `{gif_path}`")
    print(f"Generated mesh for {args.prompt} at `{obj_path}`")
