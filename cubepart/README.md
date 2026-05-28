# CubePart: An Open-Vocabulary Part-Controllable 3D Generator

<div align="center">
  <a href="https://cubepart.github.io/" target="_blank"><img src="https://img.shields.io/badge/Project-Page-1f6feb.svg" height="22px"></a>
  <a href="https://arxiv.org/abs/2605.28763" target="_blank"><img src="https://img.shields.io/badge/arXiv-2605.28763-b31b1b.svg?logo=arxiv" height="22px"></a>
  <a href="https://huggingface.co/Roblox/cubepart" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Models-d96902.svg" height="22px"></a>
  <a href="https://huggingface.co/spaces/Roblox/cubepart-demo" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Demo-blue.svg" height="22px"></a>
</div>

<br/>

<p align="center">
  <img src="examples/assets/teaser.jpg" alt="CubePart teaser" width="800" style="margin: 5px;"/>
</p>


CubePart is a generative framework for **open-vocabulary, part-controllable** 3D
mesh generation. Given a global text prompt and a user-defined parts schema
(an open-ended list of part names), CubePart synthesizes a set of meshes—one
per schema element—that assemble into a coherent object while respecting the
specified semantic structure. The resulting assets can be directly integrated
into game engines and driven by animation, physics, and behavior scripts
without manual post-processing.

This codebase releases the multi-part mesh decomposition
model, together with its shape VAE and inference scripts. Given any input
mesh and a list of part names, it produces the corresponding per-part meshes.

## Installation

```bash
pip install -e .
# or
pip install -r requirements.txt
```

### ROCm / AMD GPU installation

CubePart can also run in a ROCm PyTorch environment. The example below was
tested with an AMD Instinct MI300X GPU on ROCm 6.4.3 using the official PyTorch
ROCm container.

```bash
docker run --rm -it \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --ipc=host --shm-size=96G \
    -v "$PWD":"$PWD" -w "$PWD/cubepart" \
    rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0 \
    bash
```

Inside the container, keep the ROCm-enabled `torch` and `torchvision` wheels
from the image, then install the ROCm helper requirements and the package:

```bash
pip install -r requirements_rocm.txt
pip install "fpsample>=0.3" --no-build-isolation
pip install -e . --no-build-isolation
```

If you use the repository example meshes, make sure Git LFS assets are present:

```bash
git lfs pull --include="cubepart/examples/inputs/jellyfish_car.glb"
```

ROCm note: `warp-lang` may not detect AMD GPUs as CUDA devices. In that case,
geometry extraction can fall back to the CPU path and still export meshes, but
this path is not expected to provide ROCm-native geometry extraction performance.

## Download model weights

The pretrained checkpoints (multi-part DiT + shape VAE) are hosted on the
Hugging Face Hub at [`Roblox/cubepart`](https://huggingface.co/Roblox/cubepart).
Download them into a local `weights/` directory:

```bash
huggingface-cli download Roblox/cubepart --local-dir weights
```

or, equivalently, from Python:

```python
from huggingface_hub import snapshot_download

snapshot_download(repo_id="Roblox/cubepart", local_dir="weights")
```

This produces:

```
weights/
├── multi_part_dit.safetensors    # multi-part DiT (~8.6 GB)
└── vae.safetensors               # shape VAE (~1.3 GB)
```

All examples below assume this `weights/` layout. If you keep the weights
elsewhere, pass the corresponding paths via `--checkpoint` / `--vae-checkpoint`.

## Quick start

### Multi-part decomposition

The parts pipeline takes a pre-encoded shape latent plus a list of part
names and returns one mesh per part. Use `encode_shape` to obtain the
latent from an existing mesh.

```python
import torch
import trimesh

from cube_part.pipelines import PartShapeDenoiserPipeline, ShapeInput
from cube_part.utils.mesh import load_mesh, sample_surface

parts_pipe = PartShapeDenoiserPipeline(
    config_path="configs/shape_denoiser_multimesh.yaml",
    checkpoint_path="weights/multi_part_dit.safetensors",
    vae_checkpoint_path="weights/vae.safetensors",
    extract_geometry_fn_name="extract_geometry_coarse_to_fine",
)

mesh, _, _ = load_mesh("examples/inputs/jellyfish_car.glb")
surface = sample_surface(mesh, num_samples=128_000)
surface = (
    torch.from_numpy(surface).to(parts_pipe.device).unsqueeze(0).float()
)
latents, _ = parts_pipe.encode_shape(surface)

part_meshes = parts_pipe.input_to_part_shape(
    ShapeInput(prompt=[["body", "wheels"]], latents=latents),
    guidance_scale=7.5,
    num_inference_steps=50,
)

for i, (vertices, faces) in enumerate(part_meshes):
    if vertices is not None:
        trimesh.Trimesh(vertices, faces).export(f"part_{i:02d}.glb")
```

A complete, runnable example lives in [`examples/run_inference.py`](examples/run_inference.py).
```bash
export PYTHONPATH=.

python examples/run_inference.py \
    --config configs/shape_denoiser_multimesh.yaml \
    --checkpoint weights/multi_part_dit.safetensors \
    --vae-checkpoint weights/vae.safetensors \
    --mesh examples/inputs/jellyfish_car.glb \
    --parts "body, front right wheel, front left wheel, rear right wheel, rear left wheel, exhaust pipe, headlights, gun" \
    --output outputs/jellyfish_car_parts
```

> **Notes:** To get optimal results, please follow the guidelines below:
>
> 1. The input mesh should be **canonically aligned** (+Y up, +Z forward).
> 2. A watertight, single-surface mesh is preferred. Meshes with duplicated inner/outer shells (common in some AI-generated meshes) often degrade quality.

### Gradio demo

A small Gradio UI is provided in [`examples/gradio_demo.py`](examples/gradio_demo.py).
It loads the multi-mesh denoiser once, lets you drag-and-drop a `.glb`, type a
list of part names, and renders the resulting colored part scene with
`gr.Model3D`.

```bash
pip install ".[demo]"   # adds gradio
python examples/gradio_demo.py \
    --config configs/shape_denoiser_multimesh.yaml \
    --checkpoint weights/multi_part_dit.safetensors \
    --vae-checkpoint weights/vae.safetensors
```


## License

This repo packages and lightly adapts code from the [Qwen-Image](https://github.com/QwenLM/Qwen-Image) and [DINOv2](https://github.com/facebookresearch/dinov2)
projects (both Apache 2.0). See individual file headers for attribution.

The cubepart-specific code is released under the same license as [Cube](https://github.com/Roblox/cube), see
[`LICENSE`](https://github.com/Roblox/cube/tree/main?tab=License-1-ov-file).

## Citation

If you find this work helpful, please consider citing our paper:

```bibtex
@inproceedings{zhu2026cubepart,
  author    = {Zhu, Yiheng and Deng, Kangle and Fauconnier, Jean-Philippe
               and Navarro, Inaki and Li, Daiqing and Pun, Ava
               and Zhang, Yinan and Zhuang, Peiye and Sun, Xiaoxia
               and Agrawala, Maneesh and Bhat, Kiran and Zhou, Tinghui},
  title     = {CubePart: An Open-Vocabulary Part-Controllable 3D Generator},
  booktitle = {SIGGRAPH},
  year      = {2026},
}
```

## Acknowledgments

We thank the leadership, Nishchaie Khanna, Karun Channa, Anupam Singh, and
David Baszucki, for their support and guidance throughout this work. We also
thank Michael Palleschi, Maurice Chu, Keenan Crane, and Kayvon Fatahalian for
helpful discussions. We are grateful to Zhenyu Zhao, Daniel Chin, Michael
Spedden, Alvin Chan, and Saurav Dhakad for setting up the evaluation pipeline
as part of the broader project. Finally, we are thankful to the ML-Platform
team, Anying Li, Yiqing Wang, Steve Han, Sourashis Roy, Chengyi Nie, Wei Zeng,
Sal Pathare, Mandar Deshpande, and Andy Shen, for their contributions and
collaboration that helped make this project possible.

CubePart builds on a number of excellent open-source projects. We thank the
contributors of [Cube](https://github.com/Roblox/cube),
[Qwen-Image](https://github.com/QwenLM/Qwen-Image),
[DINOv2](https://github.com/facebookresearch/dinov2),
[diffusers](https://github.com/huggingface/diffusers),
[transformers](https://github.com/huggingface/transformers),
[TRELLIS](https://github.com/microsoft/TRELLIS),
[CraftsMan3D](https://github.com/wyysf-98/CraftsMan3D),
[Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2),
[PartCrafter](https://github.com/wgsxm/PartCrafter)
for their open-source contributions.
