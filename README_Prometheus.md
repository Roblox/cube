# Cube: AI-Powered 3D Asset Generation through Advanced Generative Intelligence

## Project Overview

Cube is an innovative generative AI system for 3D asset creation, developed by Roblox's Foundation AI Team. It represents a groundbreaking approach to 3D intelligence, focusing on empowering developers and creators to generate high-quality 3D assets through advanced machine learning techniques.

### Core Purpose

The primary goal of Cube is to build a foundation model that can support comprehensive 3D asset generation, enabling developers to create complex 3D objects, scenes, and interactive elements with simple text prompts. It aims to democratize 3D content creation by reducing the technical barriers traditionally associated with 3D modeling.

### Key Features

- **Text-to-Shape Generation**: Convert natural language descriptions into detailed 3D mesh models
- **Shape Tokenization**: Advanced algorithm for representing and reconstructing 3D geometries
- **High-Quality Asset Creation**: Generate intricate 3D objects with fine-grained detail
- **Flexible Generation**: Support for various object types, from characters to props and scenes
- **Low-Barrier Entry**: Intuitive text-based interface for 3D asset creation

### Technical Innovations

The system comprises two primary components:
- A sophisticated shape tokenizer that converts 3D geometries into discrete token representations
- A powerful text-to-shape generation model that can create 3D assets from textual descriptions

By providing an accessible, AI-powered approach to 3D asset generation, Cube represents a significant step towards more democratized and efficient 3D content creation for developers, designers, and creators.

## Getting Started, Installation, and Setup

### Prerequisites

- Python 3.7 or higher
- CUDA-compatible GPU recommended (24GB VRAM for fast inference, 16GB otherwise)
- Blender version 4.3+ (optional, for rendering GIFs)

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

Install the core package with basic dependencies:

```bash
pip install -e .
```

For additional mesh processing capabilities, install with optional dependencies:

```bash
pip install -e .[meshlab]
```

##### CUDA Installation (Windows)

If you're using a Windows machine, install CUDA toolkit and PyTorch with CUDA support:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

#### 4. Download Model Weights

Download the model weights using the Hugging Face CLI:

```bash
huggingface-cli download Roblox/cube3d-v0.1 --local-dir ./model_weights
```

### Quick Start

#### Generate 3D Shapes

To generate a 3D model from a text prompt:

```bash
python -m cube3d.generate \
    --gpt-ckpt-path model_weights/shape_gpt.safetensors \
    --shape-ckpt-path model_weights/shape_tokenizer.safetensors \
    --fast-inference \
    --prompt "Broad-winged flying red dragon, elongated, folded legs."
```

#### Optional Rendering

To generate a turntable GIF of the generated mesh (requires Blender):

```bash
python -m cube3d.generate \
    --gpt-ckpt-path model_weights/shape_gpt.safetensors \
    --shape-ckpt-path model_weights/shape_tokenizer.safetensors \
    --fast-inference \
    --prompt "A detailed steampunk airship" \
    --render-gif
```

### Alternative Platforms

- Try the [Google Colab Demo](https://colab.research.google.com/drive/1ZvTj49pjDCD_crX5WPZNTAoTTzL6-E5t)
- Explore the [Hugging Face Interactive Demo](https://huggingface.co/spaces/Roblox/cube3d-interactive)

### Troubleshooting

- Ensure you have a compatible GPU and the latest CUDA toolkit
- Check that Blender is installed and accessible in your system PATH for GIF rendering
- Use `--resolution-base` flag to adjust mesh resolution and decoding speed

## Dataset

The dataset for this project consists of 3D object representations with associated text prompts. The example dataset includes multiple 3D models with descriptive text inputs:

#### Dataset Composition
- **Total Objects**: 4 unique 3D models
- **Object Types**: 
  - Bulldozer
  - Dragon
  - Boat
  - Sword

#### Data Format
Each object is represented by:
- 3D model file (`.obj` format)
- Animated representation (`.gif`)
- Corresponding text prompt

#### Example Dataset Schema
```json
[
  {
    "object": "string (object name)",
    "prompt": "string (descriptive text prompt)"
  }
]
```

#### Example Data Samples
- **Bulldozer**: Simple object description
- **Dragon**: Detailed descriptive prompt (e.g., "Broad-winged flying red dragon, elongated, folded legs")
- **Boat**: Basic object reference
- **Sword**: Detailed creative description

#### Data Characteristics
- **File Types**: 
  - 3D Models: `.obj`
  - Animations: `.gif`
  - Prompts: `JSON`
- **Size**: Small-scale dataset (4 objects)
- **Purpose**: Demonstrative dataset for 3D object generation and text-to-3D modeling

#### Notes
- The dataset serves as an example and demonstration of the model's capabilities
- Actual training dataset may differ from these example objects

## Model Architecture and Training

The project utilizes a sophisticated Dual-Stream Roformer model architecture designed for advanced 3D shape generation and manipulation. The model is a transformer-based architecture with unique design characteristics:

### Model Architecture

The core model is a `DualStreamRoformer` with the following key architectural features:
- Dual-stream transformer architecture with rotary positional embeddings (RoPE)
- 23 dual-stream decoder layers and 1 single-stream decoder layer
- Multi-head attention mechanism with 12 attention heads
- Embedding dimension of 1536
- Vocabulary size of 16,384 tokens
- Supports both text and shape embeddings

#### Key Components
- Text Projection Layer: Projects text embeddings to the model's embedding space
- Shape Projection Layer: Projects shape embeddings to the model's embedding space
- Dual-Stream Decoder Blocks: Enable complex cross-modal interactions
- Layer Normalization and Linear Head: For final token prediction

### Training Configuration

The model is configured with the following training parameters:
- Rotary Positional Embedding Base (θ): 10,000
- Layer Normalization Epsilon: 1e-6
- Bias in Linear Layers: Enabled
- Special Tokens: 
  - Shape Beginning-of-Sequence (BOS) Token
  - Shape End-of-Sequence (EOS) Token
  - Padding Token

### Embedding Details
- Text Model Embedding Dimension: 768 (using OpenAI CLIP ViT-Large-Patch14)
- Shape Model Embedding Dimension: 32
- Supports both pooled and non-pooled text embeddings

### Model Training Approach
The model is designed for cross-modal learning, enabling generation and manipulation of 3D shapes based on textual descriptions. It uses a sophisticated dual-stream attention mechanism that allows for complex interactions between text and shape representations.

## Evaluation and Results

The Cube 3D model is a generative AI system for 3D shape generation, focusing on text-to-shape capabilities. While comprehensive academic evaluation details are not explicitly provided in the repository, the project demonstrates its performance through several key aspects:

### Performance Characteristics

- **Generation Capability**: The model can generate 3D shapes from textual descriptions across various categories, including:
  - Creatures (e.g., flying dragons)
  - Objects (e.g., noise-canceling headphones)
  - Complex geometries with detailed specifications

### Evaluation Metrics and Demonstrations

The model's effectiveness is primarily illustrated through:
- Visual generation quality in example outputs
- Diversity of generated 3D shapes
- Ability to interpret and translate complex textual prompts into 3D geometries

### Experimental Setup

#### Hardware Requirements
The model has been validated on multiple GPU configurations:
- Nvidia H100 GPU
- Nvidia A100 GPU
- Nvidia Geforce 3080
- Apple Silicon M2-4 Chips

#### Recommended Hardware
- Minimum 16GB VRAM for standard inference
- 24GB VRAM recommended for fast inference mode

### Generation Flexibility

The model offers several parameters to control generation:
- `resolution_base`: Controls output mesh resolution (range 4.0-9.0)
  - Lower values: Faster decoding, coarser mesh
  - Higher values: More detailed mesh, increased computational complexity
- `top_p`: Controls generation randomness 
  - Values < 1: Probabilistic token selection
  - Default (None): Deterministic generation

### Practical Limitations

- Current version focuses on single 3D shape generation
- Performance may vary based on input prompt complexity
- Rendering requires Blender (version >= 4.3) for gif generation

### Future Research Directions

The project hints at upcoming advancements:
- Bounding box conditioning for shape generation
- Scene-level 3D generation
- Enhanced control over generative processes

## Technologies Used

### Programming Languages
- Python 3.7+

### Machine Learning and Deep Learning
- PyTorch (version >=2.2.2)
- Transformers
- Hugging Face Hub
- Accelerate (version >=0.26.0)

### 3D Processing and Rendering
- Trimesh
- Blender (version >=4.3, for rendering)
- Warp-lang

### Scientific and Numerical Computing
- NumPy
- Scikit-image

### Development and Utility Libraries
- OmegaConf (configuration management)
- tqdm (progress bars)

### Optional Dependencies
- PyMeshLab (for mesh simplification)
- Ruff (version 0.9.10, for linting)

### Platforms and Environments
- CUDA (for GPU acceleration)
- Google Colab
- Hugging Face Spaces

### Build and Package Management
- Setuptools
- Wheel

## Additional Notes

### Known Limitations

The current version of Cube 3D has several important considerations:

- The model is optimized for generating 3D shapes from text prompts, with varying levels of complexity and detail
- Performance and output quality may vary depending on the specificity and complexity of the input prompt
- Rendering GIFs requires Blender version 4.3 or higher installed in the system PATH

### Security and Responsible Use

Users are encouraged to use the model responsibly and ethically. Any potential security vulnerabilities should be reported through Roblox's official HackerOne bug bounty program.

### Future Roadmap

The project aims to expand 3D intelligence capabilities, with upcoming features including:

- Bounding box conditioning for shape generation
- Scene generation capabilities
- Enhanced 3D asset creation tools

### Performance Recommendations

- For optimal results, use a GPU with at least 24GB VRAM when using fast inference
- Lower resolution settings can improve inference speed on machines with limited GPU memory
- Experiment with the `top_p` parameter to control generation randomness

### Research and Academic Use

Researchers and academics are welcome to use and build upon this work. If you use Cube 3D in your research, please cite the project's technical report as provided in the Citation section.

### Community and Collaboration

This project is part of Roblox's ongoing efforts to advance 3D intelligence. Contributions, feedback, and collaborative research are highly encouraged.

## Contributing

We welcome contributions to this project! To help maintain code quality and collaboration, please follow these guidelines:

### Pull Request Process
1. Fork the repository and create your branch from `main`.
2. Ensure any new code passes existing tests and includes appropriate tests for new functionality.
3. Update documentation to reflect any changes you've made.
4. Fill out the PR template completely, providing:
   - A clear summary of changes
   - Context for the changes
   - Testing instructions
   - Screenshots (if applicable)

### Code Style and Linting
- This project uses `ruff` for linting (version 0.9.10)
- Before submitting a PR, run the linter to ensure code quality
- Follow Python 3.10+ coding standards

### Development Requirements
- Python 3.10 or higher
- Install development dependencies using `pip install .[lint]`

### Reporting Issues
- Use the GitHub issue templates for bug reports and feature requests
- Provide detailed information to help us understand and reproduce the issue

### Security
- For security vulnerabilities, please **do not** create a public issue
- Report security concerns through the [HackerOne bug bounty program](https://hackerone.com/roblox)

### Optional Contributions
Areas where contributions are particularly welcome:
- Documentation improvements
- Performance optimizations
- Additional test coverage
- New feature implementations

Thank you for contributing to our project!

## License

This project is released under the **CUBE3D RESEARCH-ONLY RAIL-MS LICENSE**.

### Key License Highlights

- The license is specifically for the Cube3d-v0.1 and related inference code
- Use is strictly limited to **academic or research purposes only**
- Significant use restrictions apply, including:
  - Prohibitions on discrimination
  - Restrictions on intellectual property use
  - Limitations on generating potentially harmful content
  - Restrictions on military, law enforcement, and high-risk applications

### Full License Terms

Please see the [LICENSE](LICENSE) file for complete details on usage rights, restrictions, and conditions. 

#### Important Restrictions
- Commercial use is not permitted
- Distribution is allowed only under specific conditions
- Users must comply with detailed use-based restrictions

#### Disclaimer
The software is provided "AS IS" without warranties of any kind. Users are solely responsible for determining the appropriateness of using the artifact.