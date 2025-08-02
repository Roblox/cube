# Cube: Generative AI for 3D Asset Creation by Roblox Foundation AI Team

## Project Overview

Cube is an innovative generative AI system for 3D asset creation, developed by Roblox's Foundation AI Team. The project aims to build a foundation model for 3D intelligence that can support developers in creating comprehensive digital experiences.

### Core Objective
The primary goal of Cube is to democratize 3D asset generation by providing an advanced text-to-shape generation model that enables creators to transform textual descriptions into high-quality 3D models efficiently and intuitively.

### Key Features
- **Text-to-Shape Generation**: Convert natural language prompts into detailed 3D models
- **Shape Tokenization**: Advanced tokenization technique for representing 3D geometries
- **Flexible Model Architecture**: Supports various input complexities and generation scenarios
- **Cross-Platform Compatibility**: Tested on multiple GPU architectures and Apple Silicon

### Technical Innovation
Cube represents a significant step towards foundational AI models for 3D content creation. It addresses the complex challenge of generating semantically meaningful 3D assets from textual descriptions, bridging the gap between natural language understanding and 3D geometry generation.

### Potential Applications
- 3D asset creation for game development
- Rapid prototyping of digital objects
- Creative design exploration
- Automated 3D modeling for various industries

By open-sourcing this technology, Roblox aims to engage the research community and accelerate innovation in generative 3D AI.

## Getting Started, Installation, and Setup

### Prerequisites

- Python 3.7 or higher
- CUDA-compatible GPU recommended (24GB VRAM preferred)
- Blender (version >= 4.3) for rendering GIFs (optional)

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/Roblox/cube.git
cd cube
```

#### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

#### 3. Install Dependencies

Standard installation:
```bash
pip install -e .
```

Installation with optional Meshlab support:
```bash
pip install -e .[meshlab]
```

##### CUDA Note for Windows
If using a Windows machine, install CUDA toolkit and PyTorch with CUDA support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

### Model Weights Download

Download model weights from Hugging Face:
```bash
huggingface-cli download Roblox/cube3d-v0.1 --local-dir ./model_weights
```

### Quick Start

#### Generate 3D Shape
```bash
python -m cube3d.generate \
    --gpt-ckpt-path model_weights/shape_gpt.safetensors \
    --shape-ckpt-path model_weights/shape_tokenizer.safetensors \
    --prompt "Broad-winged flying red dragon, elongated, folded legs." \
    --render-gif
```

### Development and Deployment

#### Running in Development
1. Ensure all dependencies are installed
2. Use the `cube3d` module for development and testing
3. Refer to the example scripts in the `examples/` directory

#### Supported Platforms
- Linux
- macOS
- Windows (with CUDA support)

### Recommended Hardware
- Nvidia H100, A100, or Geforce 3080 GPUs
- Minimum 16GB VRAM
- Apple Silicon M2-4 Chips supported

### Important Notes
- `--fast-inference` flag may not work on all GPUs or macOS
- Lower resolution can be specified using `--resolution-base` for faster decoding
- Blender must be installed and accessible in system PATH for GIF rendering

## Dataset

The dataset for this project consists of 3D object generation examples, demonstrating the model's capability to generate diverse 3D meshes from textual descriptions.

### Dataset Characteristics
- **Size**: 4 example objects (bulldozer, dragon, boat, sword)
- **Format**: 3D meshes (`.obj` files) with corresponding text prompts
- **Representation**: Each object includes:
  - 3D mesh file (`.obj`)
  - Animated turntable rendering (`.gif`)
  - Textual description/prompt

### Example Prompts
1. **Bulldozer**: "bulldozer"
2. **Dragon**: "Broad-winged flying red dragon, elongated, folded legs."
3. **Boat**: "a boat"
4. **Sword**: "a purple crystal blade fantasy sword with green gem accents."

### Data Structure
The examples demonstrate the model's ability to generate 3D objects from varied textual inputs, ranging from simple object names to detailed descriptive prompts.

### Limitations
These examples are demonstration samples and do not represent the full training dataset used to develop the Cube 3D model. The actual training data is not included in this repository.

## Model Architecture and Training

The project employs a sophisticated multi-model architecture designed for 3D shape generation and manipulation, combining several advanced modeling techniques:

### Model Components

#### GPT Model
- Transformer-based generative model with the following key specifications:
  - 23 layers
  - 12 attention heads
  - Embedding dimension: 1,536
  - Vocabulary size: 16,384
  - Uses Rotary Positional Embeddings (RoPE) with theta of 10,000

#### Shape Model
- Specialized encoder-decoder architecture for 3D shape representation:
  - 13 encoder layers
  - 24 decoder layers
  - 512 encoder latents
  - Embedding dimension: 32
  - Cross-attention levels: [0, 2, 4, 8]
  - Point feature dimension: 3

#### Text Model
- Pre-trained CLIP ViT-Large model (openai/clip-vit-large-patch14) for text embedding

### Model Architecture Rationale
The multi-modal architecture enables complex 3D shape generation by:
- Integrating text and shape representations
- Leveraging transformer-based generative techniques
- Supporting detailed spatial encoding
- Enabling cross-modal understanding and generation

### Training Approach
While specific training scripts are not directly visible in the repository, the configuration suggests a sophisticated training process involving:
- Multi-modal learning
- Latent space manipulation
- Cross-modal embedding techniques

### Model Configurations
Model hyperparameters are defined in `cube3d/configs/open_model.yaml`, allowing flexible configuration of architectural details and training parameters.

## Evaluation and Results

The Cube 3D model is designed for text-to-shape generation, transforming textual descriptions into 3D mesh representations. While comprehensive academic evaluation details are not directly included in the repository, the model provides several key features and performance considerations:

### Performance Characteristics

- **Inference Modes**: 
  - Standard inference mode
  - Fast inference mode (optimized for CUDA-enabled GPUs)
  - Supports deterministic and probabilistic generation via `top_p` parameter

### Generation Parameters

The model supports several key parameters that impact generation quality and performance:

- `resolution_base`: Controls the mesh resolution 
  - Range: 4.0 to 9.0
  - Lower values result in faster decoding and coarser meshes
  - Recommended default: 8.0

- `top_p`: Controls generation randomness
  - `None`: Deterministic generation
  - Float value < 1: Keeps smallest set of tokens with cumulative probability ≥ top_p

### Hardware Performance

Tested on:
- Nvidia H100 GPU
- Nvidia A100 GPU
- Nvidia Geforce 3080
- Apple Silicon M2-4 Chips

#### Recommended Hardware
- Minimum 16GB VRAM for standard inference
- Minimum 24GB VRAM for fast inference mode

### Mesh Post-processing

The model includes optional mesh refinement:
- Automatic face count reduction (default target: 10,000 faces)
- Preserves mesh quality while optimizing computational efficiency
- Can be disabled with `--disable-postprocessing` flag

### Generation Example

Typical generation command demonstrating key parameters:

```bash
python -m cube3d.generate \
    --gpt-ckpt-path model_weights/shape_gpt.safetensors \
    --shape-ckpt-path model_weights/shape_tokenizer.safetensors \
    --prompt "Broad-winged flying red dragon" \
    --resolution-base 8.0 \
    --top-p 0.9
```

### Visualization

The model supports rendering generated meshes as turntable GIFs for easy visualization, requiring Blender (version >= 4.3) to be installed.

## Project Structure

The project is organized into several key directories and files to support 3D model generation and processing:

### Main Package Structure
- `cube3d/`: Main package directory containing core functionality
  - `configs/`: Configuration files for the project
  - `inference/`: Modules for model inference and processing
    - `engine.py`: Core inference logic
    - `logits_postprocesses.py`: Postprocessing utilities for model outputs
    - `utils.py`: Supporting utility functions
  - `mesh_utils/`: Utilities for mesh processing
  - `model/`: Neural network model implementations
    - `autoencoder/`: Autoencoder-related model components
      - `embedder.py`: Embedding layer implementations
      - `grid.py`: Grid-related model components
      - `one_d_autoencoder.py`: One-dimensional autoencoder
      - `spherical_vq.py`: Spherical vector quantization implementation
    - `gpt/`: GPT model implementations
    - `transformers/`: Transformer-related model components
      - Multiple attention and normalization implementations
  - `renderer/`: Rendering-related scripts
    - `blender_script.py`: Blender rendering utilities
    - `renderer.py`: Rendering engine

### Project Supporting Files
- `setup.py`: Package installation and setup configuration
- `pyproject.toml`: Project build system and dependency configuration

### Example Resources
- `examples/`: Sample data and demonstration files
  - Contains `.obj` 3D model files
  - Includes `.gif` visualizations
  - `prompts.json`: Example prompts or configuration file

### Additional Directories
- `.github/`: GitHub-specific configuration
  - Issue and pull request templates
- `resources/`: Additional project resources
  - Contains image and visualization files

### Key Python Files
- `generate.py`: Likely responsible for model output generation
- `vq_vae_encode_decode.py`: Vector quantization and autoencoder encoding/decoding logic

## Technologies Used

### Programming Languages
- Python 3.7+

### Machine Learning and Deep Learning
- PyTorch (>=2.2.2)
- Transformers
- Hugging Face Hub
- Accelerate (>=0.26.0)

### Numerical and Scientific Computing
- NumPy
- scikit-image

### 3D Graphics and Rendering
- Trimesh
- Warp-lang

### Configuration and Utilities
- OmegaConf
- tqdm

### Optional Dependencies
- PyMeshlab (optional, for mesh processing)

### Development Tools
- Ruff (linting)
- Setuptools
- Wheel

### Compatibility
- Supports Python 3.10
- Compatible with various machine learning and 3D modeling workflows

## Additional Notes

### Experimental Status

This project is an early-stage research implementation of a 3D generative AI system. The current release represents an initial exploration into text-to-3D generation, with several important considerations:

- The model is primarily designed for experimental and research purposes
- Generated 3D assets may require manual refinement
- Performance and quality can vary depending on input prompts and hardware specifications

### Limitations and Known Constraints

#### Model Capabilities
- Text-to-shape generation is currently limited to single object generation
- Complex or highly detailed prompts may produce unpredictable results
- Recommended for creative exploration and prototype development

#### Hardware Recommendations
- Optimal performance requires high-end CUDA-enabled GPUs
- Minimum recommended VRAM: 16-24 GB
- Limited support for integrated graphics and lower-end GPUs

### Future Development Roadmap

The project aims to expand 3D generative capabilities through:
- Improved shape generation precision
- Enhanced text-to-shape understanding
- Bounding box conditioning
- Scene generation capabilities

### Compatibility and Integration

#### Supported Platforms
- Linux
- Windows
- macOS (with some limitations)
- Apple Silicon (M1/M2 chips)

#### Key Dependencies
- Python 3.8+
- PyTorch
- Blender (version 4.3+, optional for rendering)
- CUDA Toolkit (recommended for GPU acceleration)

### Responsible AI Considerations

This research prototype is released with the intent of responsible innovation. Users are encouraged to:
- Respect intellectual property rights
- Use generated assets ethically
- Provide constructive feedback to improve the technology

## Contributing

We welcome contributions from the community! By contributing, you help improve and expand this project.

### Ways to Contribute
- Report bugs and request features through GitHub Issues
- Submit pull requests with bug fixes or new functionality
- Improve documentation
- Share your use cases and provide feedback

### Pull Request Process
1. Fork the repository and create your branch from `main`
2. Ensure your code follows the project's code style and conventions
3. Include clear, concise descriptions of your changes
4. Add or update tests as appropriate
5. Verify that all tests pass before submitting

### Contribution Guidelines
- Write clean, readable, and well-documented code
- Follow existing code patterns and styles in the project
- Include appropriate type hints and docstrings
- Ensure backward compatibility when possible
- Update documentation to reflect your changes

### Reporting Issues
- Use the provided GitHub Issue templates
- Provide detailed information about the bug or feature request
- Include reproducible steps, expected vs. actual behavior
- For security vulnerabilities, follow the guidelines in SECURITY.md

### Code of Conduct
Treat all contributors with respect. Harassment, discrimination, or any form of disrespectful behavior will not be tolerated.

### Questions?
If you have questions about contributing, please open an issue with your inquiry.

## License

The project is released under the **Cube3D Research-Only RAIL-MS License**. 

### Key License Highlights

- The license is specifically for research purposes only (Permitted Purpose)
- Strict use restrictions are in place, including:
  - Prohibitions on discrimination
  - Restrictions on intellectual property use
  - Limitations on generating harmful or misleading content
  - Constraints on privacy and personal information
  - Restrictions on military, law enforcement, and high-risk applications

### Important Conditions

- Use is restricted to academic or research purposes
- Redistribution requires:
  - Inclusion of original license terms
  - Prominent notices for modified files
  - Retention of all original copyright and attribution notices

### Full License

For complete details, please refer to the [LICENSE](LICENSE) file in the repository. Users must carefully review and comply with all terms and conditions before using the artifact.

#### Disclaimer

THE ARTIFACT IS PROVIDED "AS IS" WITHOUT WARRANTIES OF ANY KIND. THE CONTRIBUTORS SHALL NOT BE LIABLE FOR ANY DAMAGES ARISING FROM ITS USE.