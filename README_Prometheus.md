# Cube: Generative AI for Advanced 3D Asset Creation and Digital Content Generation

## Project Overview

Cube is an advanced generative AI system designed to revolutionize 3D asset creation and digital content generation. Developed by Roblox's Foundation AI Team, this project aims to build a foundational model for 3D intelligence that empowers developers and creators to produce sophisticated 3D content with unprecedented ease.

### Core Objective

The primary goal of Cube is to democratize 3D content creation by providing an intelligent, text-driven system for generating high-quality 3D assets. It represents a significant step towards enabling developers to create complex 3D objects, scenes, and interactive elements through simple text prompts.

### Key Features

- **Text-to-Shape Generation**: Transform natural language descriptions into detailed 3D models with high fidelity and creativity
- **Shape Tokenization**: Advanced tokenization technique that enables complex 3D asset representation and manipulation
- **Flexible Inference**: Support for various hardware configurations, including GPUs and Apple Silicon
- **Open-Source Accessibility**: Designed to be accessible to researchers, developers, and creators of all skill levels

### Technical Capabilities

- Generate 3D models from textual descriptions
- Tokenize and reconstruct 3D shapes
- Support for various 3D asset generation scenarios
- Low-friction API for easy integration into existing workflows

### Potential Applications

- Game development
- 3D asset creation
- Prototyping digital objects
- Creative design and visualization
- Research in generative AI and 3D modeling

## Getting Started, Installation, and Setup

### Prerequisites

- Python 3.7 or higher
- PyTorch 2.2.2+
- pip package manager

### Installation

You can install the project directly from GitHub:

```bash
pip install git+https://github.com/your-repo/cube3d.git
```

#### Optional Dependencies

For additional functionality, you can install optional dependencies:

```bash
# Install MeshLab support
pip install "cube3d[meshlab]"

# Install linting tools
pip install "cube3d[lint]"
```

### Quick Start

#### Basic Usage

```python
from cube3d import generate

# Example generation (placeholder - adjust based on actual API)
model = generate.create_model()
output = model.generate_3d_asset(prompt="A detailed boat")
```

### Development Setup

1. Clone the repository:
```bash
git clone https://github.com/your-repo/cube3d.git
cd cube3d
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -e .
```

### System Requirements

- Recommended: CUDA-capable GPU for optimal performance
- Minimum 16GB RAM
- Disk space: Approximately 5GB for model weights and dependencies

### Troubleshooting

- Ensure you have the latest version of PyTorch compatible with your CUDA version
- If you encounter dependency conflicts, consider using a virtual environment
- Check `requirements.txt` for exact dependency versions if needed

### Performance Notes

- Model generation time varies based on hardware
- GPU acceleration significantly improves performance

## API Reference

### Main Generation Functions

#### `generate_mesh(engine, prompt, output_dir, output_name, resolution_base=8.0, disable_postprocess=False, top_p=None)`
Generates a 3D mesh from a text prompt using the specified engine.

- **Parameters**:
  - `engine`: An inference engine (Engine or EngineFast)
  - `prompt` (str): Text description for mesh generation
  - `output_dir` (str): Directory to save the generated mesh
  - `output_name` (str): Base filename for the output mesh
  - `resolution_base` (float, optional): Resolution base for shape decoder. Defaults to 8.0.
  - `disable_postprocess` (bool, optional): Skip mesh postprocessing. Defaults to False.
  - `top_p` (float, optional): Probability threshold for token selection. Defaults to None.

- **Returns**: 
  - `str`: Path to the generated .obj mesh file

- **Example**:
```python
from cube3d.inference.engine import Engine
from cube3d.generate import generate_mesh

engine = Engine(config_path, gpt_ckpt_path, shape_ckpt_path)
mesh_path = generate_mesh(
    engine, 
    prompt="A detailed spaceship", 
    output_dir="./outputs", 
    output_name="spaceship"
)
```

### Inference Engines

#### `Engine(config_path, gpt_ckpt_path, shape_ckpt_path, device=None)`
Standard inference engine for 3D mesh generation.

- **Parameters**:
  - `config_path` (str): Path to configuration YAML
  - `gpt_ckpt_path` (str): Path to GPT checkpoint
  - `shape_ckpt_path` (str): Path to shape encoder/decoder checkpoint
  - `device` (torch.device, optional): Computation device. Defaults to auto-detect.

#### `EngineFast(config_path, gpt_ckpt_path, shape_ckpt_path, device=None)`
Optimized inference engine using CUDA graphs for faster generation.

- **Parameters**: Same as `Engine`

### Mesh Rendering

#### `renderer.render_turntable(obj_path, output_dir)`
Renders a turntable GIF animation of a 3D mesh.

- **Parameters**:
  - `obj_path` (str): Path to the .obj mesh file
  - `output_dir` (str): Directory to save the output GIF

- **Returns**: 
  - `str`: Path to the generated GIF file

### Mesh Postprocessing Utilities

These utilities are available if PyMeshLab is installed:

#### `create_pymeshset(vertices, faces)`
Creates a PyMeshLab mesh set from vertices and faces.

#### `postprocess_mesh(mesh_set, target_face_num, output_path)`
Simplifies and cleans the mesh to a target number of faces.

#### `save_mesh(mesh_set, output_path)`
Saves a PyMeshLab mesh set to a file.

### Constants

- `PYMESHLAB_AVAILABLE` (bool): Indicates whether PyMeshLab is installed for advanced mesh processing.

### Usage Notes
- Ensure required checkpoints and configuration files are available
- PyMeshLab is recommended but optional for mesh processing
- CUDA is supported for faster inference on compatible hardware

## Project Structure

The project is organized into several key directories and files to support its 3D rendering and machine learning capabilities:

### Main Package Structure
- `cube3d/`: Primary package directory containing the core implementation
    - `__init__.py`: Package initialization file
    - `generate.py`: Core generation logic
    - `colab_cube3d.ipynb`: Jupyter notebook for Colab integration
    - `configs/`: Configuration management
        - `open_model.yaml`: Model configuration file
    
### Submodules
- `cube3d/inference/`: Inference-related functionality
    - `engine.py`: Core inference engine
    - `logits_postprocesses.py`: Logits post-processing utilities
    - `utils.py`: Utility functions for inference

- `cube3d/model/`: Machine learning model implementations
    - `autoencoder/`: Autoencoder-related modules
        - `embedder.py`: Embedding logic
        - `grid.py`: Grid-related operations
        - `one_d_autoencoder.py`: One-dimensional autoencoder
        - `spherical_vq.py`: Spherical vector quantization

    - `gpt/`: GPT model implementations
        - `dual_stream_roformer.py`: Dual-stream RoFormer model

    - `transformers/`: Transformer-related modules
        - `attention.py`: Attention mechanisms
        - `cache.py`: Caching utilities
        - `dual_stream_attention.py`: Dual-stream attention implementation
        - `norm.py`: Normalization layers
        - `roformer.py`: RoFormer implementation
        - `rope.py`: Rotary Position Embedding

- `cube3d/renderer/`: Rendering utilities
    - `blender_script.py`: Blender rendering script
    - `renderer.py`: Rendering engine

### Additional Directories
- `examples/`: Sample data and example files
    - Contains `.obj` and `.gif` files for different objects
    - `prompts.json`: Prompt configurations

- `resources/`: Static resources
    - Contains various image and GIF files

### Project Configuration and Setup
- `pyproject.toml`: Project build system configuration
- `setup.py`: Package installation and setup script

### Documentation and Metadata
- `README.md`: Project documentation
- `LICENSE`: Project licensing information
- `SECURITY.md`: Security policy
- `.github/`: GitHub-specific configurations
    - Issue and pull request templates

## Technologies Used

### Programming Languages
- Python 3.10+

### Machine Learning and Deep Learning
- PyTorch (v2.2.2+): Primary deep learning framework
- Hugging Face Transformers: Transformer model architectures
- Accelerate: Distributed training and inference library
- NumPy: Numerical computing library

### 3D Graphics and Rendering
- Trimesh: 3D mesh processing and manipulation
- Warp-lang: High-performance graphics and simulation library
- Blender (via scripting): 3D rendering and visualization

### Configuration and Utilities
- OmegaConf: Configuration management
- tqdm: Progress bar and iteration tracking
- Scikit-image: Image processing utilities

### Development and Tooling
- Setuptools: Python package management
- Ruff: Python linting and code quality tool
- Hugging Face CLI: Model and dataset management

### Optional Dependencies
- PyMeshLab: Advanced mesh processing (optional)

## Additional Notes

### Model Capabilities and Limitations

The Cube 3D model is designed for generative 3D asset creation with text-to-shape capabilities. It's important to understand the current scope and constraints:

- **Generation Scope**: Primarily focused on creating individual 3D objects from text prompts
- **Resolution Flexibility**: Users can adjust output resolution using the `--resolution-base` flag, balancing between detail and computational efficiency
- **Rendering Requirements**: Turntable GIF generation requires Blender (version >= 4.3) installed in the system PATH

### Experimental Features

The project includes experimental capabilities that are actively being developed:

- Bounding box conditioning for shape generation
- Preliminary scene generation techniques
- Shape tokenization and reconstruction workflows

### Performance Considerations

- **Recommended Hardware**: 
  - GPU with 24GB VRAM for fast inference
  - Minimum 16GB VRAM for standard inference
  - Tested on Nvidia H100, A100, GeForce 3080, and Apple Silicon M2-4 Chips

### Future Development

The Cube 3D project represents an initial step towards comprehensive 3D intelligence, with ongoing research focusing on:

- Enhanced 3D object generation
- More sophisticated scene creation
- Improved text-to-3D conversion techniques

### Responsible Use

As an experimental AI technology, users are encouraged to:

- Verify generated assets for accuracy and appropriateness
- Use the model ethically and in compliance with relevant guidelines
- Understand that generated outputs may vary in quality and fidelity

## Contributing

We welcome contributions to the project! To ensure a smooth collaboration, please follow these guidelines:

### Pull Request Process
1. Fork the repository and create your branch from `main`.
2. Ensure any new code follows the project's coding standards.
3. Update relevant documentation to reflect your changes.
4. Your pull request should include:
   - A clear summary of changes
   - Context and rationale for modifications
   - Testing instructions
   - Any potential breaking changes or new dependencies

### Code Style
- Use Python 3.7+ compatible code
- Follow standard Python formatting conventions
- Use type hints where appropriate

### Development Setup
- Minimum Python version: 3.7
- Install development dependencies with `pip install .[lint]`

### Linting
- The project uses `ruff` for code linting (version 0.9.10)
- Run linter checks before submitting your pull request

### Testing
- Ensure all tests pass before submitting a pull request
- Add tests for new functionality
- Maintain or improve overall test coverage

### Reporting Issues
- Use the GitHub issue templates for bug reports, feature requests, or documentation improvements
- Provide clear, reproducible steps for any bugs discovered

### Security
If you discover a security vulnerability:
- Do NOT create a public issue
- Report vulnerabilities through the HackerOne bug bounty program: https://hackerone.com/roblox

### Code of Conduct
Be respectful, inclusive, and considerate of other contributors. Harassment and discrimination are not tolerated.

### Ownership
This project is maintained by @Roblox/aicube. All contributions are subject to review by the core team.

## License

The project is licensed under the **Cube3D Research-Only RAIL-MS License**. 

### Key License Highlights

- This is a research-only license with specific use restrictions
- The license applies to Cube3d-v0.1 and related inference code
- Usage is strictly limited to academic or research purposes

### Important Restrictions

The license includes comprehensive use restrictions, including:
- Prohibitions on discrimination
- Intellectual property protections
- Restrictions on generating harmful or unlawful content
- Limitations on military, law enforcement, and health-related applications

### Full License Details

For the complete and legally binding license terms, please refer to the [LICENSE](LICENSE) file in the repository. Users must carefully review and comply with all terms before using this software.

#### Disclaimer

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND. The licensors are not liable for damages arising from its use.