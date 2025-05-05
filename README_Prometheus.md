# Cube3D: Generative AI for 3D Asset Creation by Roblox Foundation AI Team

## Project Overview

Cube is an innovative generative AI system for 3D asset creation, developed by Roblox's Foundation AI Team. The project aims to build a foundation model for 3D intelligence that can support developers in producing comprehensive 3D content for digital experiences.

#### Key Objectives
- Create a powerful text-to-shape generation model
- Enable developers to generate 3D objects and scenes with natural language prompts
- Provide tools for creative 3D asset generation and manipulation

#### Core Features
- Text-to-Shape Generation: Convert textual descriptions into 3D models
- Shape Tokenization: Advanced tokenization technique for representing 3D shapes
- High-Quality 3D Asset Creation: Generate detailed 3D objects with complex geometries
- Flexible Inference: Support for various hardware configurations and resolution settings

#### Key Benefits
- Democratizes 3D content creation by lowering technical barriers
- Accelerates 3D asset generation for developers and creators
- Provides a flexible and powerful tool for 3D model generation
- Supports creative exploration through text-based 3D modeling

The project represents a significant step towards enabling more accessible and intelligent 3D content creation, with potential applications in game development, design, animation, and virtual environments.

## Getting Started, Installation, and Setup

### Prerequisites

- Python 3.7 or higher
- A CUDA-capable GPU (recommended, with at least 16GB VRAM)
- Blender version 4.3 or higher (optional, for rendering GIFs)

### Installation

#### Basic Installation

To install the project, clone the repository and install the package:

```bash
git clone https://github.com/Roblox/cube.git
cd cube
pip install -e .
```

#### Installation with Optional Dependencies

For additional mesh processing capabilities, install with the `meshlab` extra:

```bash
pip install -e .[meshlab]
```

#### CUDA Configuration

For Windows users or those requiring CUDA support:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

### Quick Start

#### Download Model Weights

Download the model weights from Hugging Face:

```bash
huggingface-cli download Roblox/cube3d-v0.1 --local-dir ./model_weights
```

#### Generate 3D Shapes

Generate a 3D model from a text prompt:

```bash
python -m cube3d.generate \
    --gpt-ckpt-path model_weights/shape_gpt.safetensors \
    --shape-ckpt-path model_weights/shape_tokenizer.safetensors \
    --fast-inference \
    --prompt "Broad-winged flying red dragon, elongated, folded legs."
```

### Development Tips

- For lower VRAM usage, use the `--resolution-base` flag to reduce output resolution
- The `--render-gif` flag can create a turntable animation of the generated mesh
- Ensure Blender is in your system PATH for GIF rendering

### Alternative Usage Methods

1. **Jupyter Notebook**: Use `cube3d/colab_cube3d.ipynb` for interactive development
2. **Python Library**: Import and use the `cube3d.inference.engine` module for programmatic generation

### Supported Platforms

- Linux
- macOS
- Windows

### Hardware Compatibility

Tested on:
- Nvidia H100 GPU
- Nvidia A100 GPU
- Nvidia Geforce 3080
- Apple Silicon M2-4 Chips

## Dataset

The Cube 3D project utilizes a diverse dataset for 3D shape generation, focusing on creating a generative AI system for 3D object creation. While the specific training dataset details are not explicitly provided in the repository, the project demonstrates capability through example 3D models.

#### Example Dataset
The repository includes a set of example 3D objects that showcase the model's generation capabilities:

| Object     | Description                                               |
|------------|-----------------------------------------------------------|
| Bulldozer  | A standard industrial vehicle model                       |
| Dragon     | A fantasy creature with specific attributes (broad-winged, red, folded legs) |
| Boat       | A generic boat model                                      |
| Sword      | A fantasy sword with specific design details (purple crystal, green gem accents) |

##### Data Characteristics
- File Formats: `.obj` (3D mesh), `.gif` (turntable rendering)
- Number of Example Objects: 4
- Diversity: Covers industrial, fantasy, and generic object types

#### Data Processing
The project uses a shape tokenizer to convert 3D meshes into token representations, enabling text-to-shape generation. The tokenization process allows for:
- Encoding 3D shapes into discrete token indices
- Reconstructing meshes from tokenized representations

#### External Resources
For comprehensive dataset information, refer to:
- [Hugging Face Model Page](https://huggingface.co/Roblox/cube3d-0.1)
- [ArXiv Technical Report](https://arxiv.org/abs/2503.15475)

Note: Detailed training dataset specifics are not publicly disclosed in this repository.

## Model Architecture and Training

### Model Architecture

The project utilizes a sophisticated dual-stream Roformer architecture designed for advanced 3D shape generation. The model consists of several key components:

#### Key Model Components
- **Transformer Architecture**: Dual-stream Roformer with configurable layers
- **Embedding Dimensions**: 
  - Text embedding dimension: 768 (from CLIP ViT-Large)
  - Shape model embedding dimension: 32
  - Model embedding dimension: 1536

#### Model Configuration
- **Transformer Layers**: 
  - 23 dual-stream layers
  - Additional single-stream layers optional
- **Attention Heads**: 12
- **Vocabulary Size**: 16,384 shape tokens

### Model Capabilities
The model is designed to:
- Process both text and shape representations
- Generate 3D shapes conditioned on textual descriptions
- Use rotary positional embeddings (RoPE)
- Support cross-attention between text and shape representations

### Training Considerations
- Pre-trained CLIP text encoder used as text model backbone
- Flexible embedding projection for text and shape inputs
- Custom key-value caching mechanism for efficient inference

### Training Setup
The model uses a configuration-driven approach, allowing easy modification of architectural parameters through the configuration file (`cube3d/configs/open_model.yaml`).

#### Key Training Configurations
- Rotary embedding base (theta): 10,000
- Layer normalization epsilon: 1e-6
- Bias in linear layers: Enabled
- Cross-attention levels: [0, 2, 4, 8]

### Important Model Features
- Special tokens for beginning-of-sequence (BOS), end-of-sequence (EOS), and padding
- Dual-stream attention mechanism allowing rich interaction between text and shape representations
- Flexible decoding with key-value caching

## Evaluation and Results

The Cube 3D model demonstrates advanced capabilities in generative 3D shape modeling through text-to-shape generation. The evaluation focuses on the model's ability to generate diverse and high-quality 3D assets from textual descriptions.

### Performance Characteristics

The model has been developed and tested on various hardware configurations, including:
- Nvidia H100 GPU
- Nvidia A100 GPU
- Nvidia Geforce 3080
- Apple Silicon M2-4 Chips

#### Key Performance Metrics
- Text-to-Shape Generation: The model can generate 3D models from natural language prompts
- Resolution Control: Supports variable resolution from base 4.0 to 9.0, allowing trade-offs between generation quality and computational efficiency
- Inference Speed: Optimized with fast inference mode for GPUs with sufficient VRAM

### Evaluation Methodology

The model's performance is primarily evaluated through:
- Prompt-driven Shape Generation: Ability to create 3D models that match textual descriptions
- Reconstruction Accuracy: Tokenization and de-tokenization capabilities demonstrated in `vq_vae_encode_decode.py`
- Rendering Quality: Optional GIF turntable rendering to visualize generated 3D models

### Example Evaluation Commands

Text-to-Shape Generation:
```bash
python -m cube3d.generate \
    --gpt-ckpt-path model_weights/shape_gpt.safetensors \
    --shape-ckpt-path model_weights/shape_tokenizer.safetensors \
    --fast-inference \
    --prompt "Broad-winged flying red dragon, elongated, folded legs."
```

Shape Tokenization and Reconstruction:
```bash
python -m cube3d.vq_vae_encode_decode \
    --shape-ckpt-path model_weights/shape_tokenizer.safetensors \
    --mesh-path ./outputs/output.obj
```

### Recommended Evaluation Setup

- Minimum GPU VRAM: 16GB (24GB recommended for fast inference)
- Supported Platforms: CUDA-enabled GPUs, Apple Silicon
- Resolution Range: Base 4.0 to 9.0 (lower values increase inference speed, reduce model quality)

### Limitations and Considerations

- Fast inference mode is not available on MacOS
- Rendering GIF requires Blender (version >= 4.3)
- Model performance varies with prompt complexity and specificity

## Project Structure

The project is organized into several key directories and files to support its 3D modeling and generation capabilities:

### Main Package Structure
- `cube3d/`: Primary package directory containing the core implementation
  - `configs/`: Configuration files
    - `open_model.yaml`: Model configuration settings
  
  - `inference/`: Model inference-related modules
    - `engine.py`: Core inference logic
    - `logits_postprocesses.py`: Post-processing of model logits
    - `utils.py`: Utility functions for inference

  - `mesh_utils/`: Mesh processing utilities
    - `postprocessing.py`: Mesh post-processing functions

  - `model/`: Model architecture components
    - `autoencoder/`: Autoencoder-related implementations
      - `embedder.py`: Embedding layer implementations
      - `grid.py`: Grid-related model components
      - `one_d_autoencoder.py`: One-dimensional autoencoder
      - `spherical_vq.py`: Spherical vector quantization implementation

    - `gpt/`: GPT model-specific implementations
      - `dual_stream_roformer.py`: Dual-stream RoFormer model

    - `transformers/`: Transformer architecture components
      - `attention.py`: Attention mechanism implementations
      - `cache.py`: Caching mechanisms
      - `dual_stream_attention.py`: Dual-stream attention implementation
      - `norm.py`: Normalization layers
      - `roformer.py`: RoFormer implementation
      - `rope.py`: Rotary Position Embedding implementation

  - `renderer/`: Rendering utilities
    - `blender_script.py`: Blender rendering script
    - `renderer.py`: Rendering engine

- `examples/`: Sample data and examples
  - Contains `.obj` 3D models and corresponding `.gif` animations
  - `prompts.json`: Example prompts or configuration file

- `resources/`: Additional resource files
  - Contains various image and visualization resources

### Project Configuration and Setup
- `pyproject.toml`: Project configuration and build settings
- `setup.py`: Package installation and setup script

### Documentation and Meta Files
- `README.md`: Project documentation
- `LICENSE`: Project licensing information
- `SECURITY.md`: Security policy and guidelines
- `.github/`: GitHub-specific configuration
  - Issue and pull request templates

## Technologies Used

### Programming Languages
- Python 3.10+

### Deep Learning Frameworks
- PyTorch (torch) v2.2.2+
- Transformers
- Hugging Face Accelerate

### 3D and Mesh Processing
- Trimesh
- PyMeshLab (optional)
- Warp-lang

### Machine Learning and Numerical Computing
- NumPy
- Scikit-image

### Configuration and Development
- OmegaConf
- Setuptools
- Wheel

### Utilities
- tqdm
- Hugging Face Hub CLI

### Tools and Environments
- CUDA (for GPU acceleration)
- Jupyter Notebook (for Colab support)

### Optional Tools
- Ruff (linting)

## Additional Notes

### Research and Experimental Status

This project represents an early-stage research effort in 3D generative AI, focusing on text-to-shape generation. The current implementation is experimental and may undergo significant changes as the research progresses.

### Model Limitations

- The model currently supports generating individual 3D objects rather than complete scenes
- Generation quality and complexity may vary depending on the input prompt
- Performance can be impacted by hardware constraints, particularly VRAM availability

### Future Development Roadmap

The project aims to expand capabilities in the following areas:
- Bounding box conditioning for more precise shape generation
- Scene generation capabilities
- Enhanced 3D asset creation for creative and development workflows

### Performance Considerations

- Inference speed and quality can be adjusted using the `resolution_base` parameter
- Recommended GPU memory is 24GB for fast inference, with 16GB as a minimum viable configuration
- Performance may vary across different hardware platforms (NVIDIA GPUs, Apple Silicon)

### Ethical and Responsible AI

Roblox emphasizes responsible AI development, encouraging:
- Thoughtful and creative use of generative technologies
- Exploration of 3D asset generation for diverse applications
- Collaborative research and innovation in the field of 3D intelligence

### Known Limitations

- Rendering GIFs requires Blender (version >= 4.3) installed in system PATH
- Some advanced features may have platform-specific restrictions
- Mesh quality can be affected by resolution and generation parameters

### Community and Collaboration

The project is open-sourced to:
- Engage the research community
- Encourage collaborative development
- Explore the potential of generative AI in 3D asset creation

## Contributing

We welcome contributions to the project! Here are some guidelines to help you get started:

### Contribution Process

1. Fork the repository and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. Ensure the test suite passes.
4. Make sure your code follows the project's coding style.
5. Issue a pull request with a clear and descriptive summary.

### Pull Request Guidelines

When submitting a pull request, please:
- Provide a clear, concise description of your changes
- Include the purpose of the changes
- Specify any new dependencies or breaking changes
- Include testing instructions
- Add screenshots if applicable

### Checklist for Contributors
- Verify that your code is well-documented
- Ensure all tests are passing
- Update relevant documentation
- Follow the existing code style and conventions

### Reporting Issues

If you find a bug or have a feature request, please open an issue in the GitHub repository. Use the provided issue templates to ensure we have all the necessary information.

### Code of Conduct

Please be respectful and constructive in all interactions. Harassment, discrimination, or any form of inappropriate behavior will not be tolerated.

### Questions?

If you have any questions about contributing, please reach out through the repository's issue tracker.

## License

The Cube3D project is released under the **Cube3D Research-Only RAIL-MS License**.

### Key License Highlights

- This is a research-only license
- Usage is restricted to academic or research purposes only
- Strict use restrictions are in place, including:
  - Prohibitions on discrimination
  - Restrictions on intellectual property usage
  - Limitations on potential harmful applications
  - Privacy and ethical use constraints

### Important Restrictions

The license includes comprehensive use restrictions covering areas such as:
- Discrimination and harmful content
- Intellectual property protection
- Legal compliance
- Disinformation prevention
- Privacy protection
- Health and safety considerations
- Restrictions on military or law enforcement applications

#### Full License Terms

For complete details, please refer to the [LICENSE](LICENSE) file in the repository. Users must carefully review and comply with all terms before using this project.

### Disclaimer

THE ARTIFACT IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND. THE CONTRIBUTORS SHALL NOT BE LIABLE FOR ANY DAMAGES ARISING FROM ITS USE.