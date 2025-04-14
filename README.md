# Nascent Transcription Encoder Model

This repository contains a PyTorch implementation of a nascent transcription encoder model using a BERT-like masked autoencoder architecture. The model processes genomic data from FASTA and BAM/CRAM files and learns cell type-specific features.

> Note: This is an in-progress greenfield reimplementation of research code, and functionality is still being implemented and tested.

## Overview

The nascent transcription encoder model uses a transformer-based architecture to learn representations of genomic regions based on three types of input data:

1. DNA sequence (one-hot encoded)
2. Transcription read coverage on both strands
3. Cell type information

The model is trained using a masked autoencoder approach, where random portions of the input data are masked, and the model is trained to reconstruct the masked regions. After training, a sparse autoencoder is used to extract interpretable features from the model's internal representations.

## Repository Structure

```
nasencoder/
├── data/           # Data loading modules
├── models/         # Model architecture and loss functions
├── training/       # Training utilities
├── utils/          # Analysis and visualization utilities
├── tests/          # Unit tests
└── examples/       # Example scripts
pyproject.toml      # Modern Python project configuration
.uv.toml           # UV package manager configuration
LICENSE            # MIT License
README.md          # This file
SPEC.md            # Project specification
main.py            # Main CLI script
```

## Installation

### Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/nasencoder.git
cd nasencoder

# Install the package
pip install -e .
```

### Using uv (recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver that can significantly speed up your development workflow.

```bash
# Clone the repository
git clone https://github.com/yourusername/nasencoder.git
cd nasencoder

# Install uv if you don't have it
pip install uv

# Install the package with development dependencies
uv install-dev

# Or simply install the package without dev dependencies
uv install --editable .
```

The project includes a `.uv.toml` file with custom commands to make development easier:

```bash
# Run linting
uv lint

# Format code
uv format 

# Run tests
uv test
```

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- NumPy
- PySAM
- BioPython
- Matplotlib
- pandas
- scikit-learn
- einops
- pytest

## Usage

### Data Loading

The data loading module handles loading and preprocessing data from FASTA and BAM/CRAM files:

```python
from nasencoder.data.loader import GenomicDataLoader

# Create a data loader
loader = GenomicDataLoader(
    fasta_file="path/to/genome.fa",
    bam_file="path/to/reads.bam",
    region_size=256
)

# Get a specific region
region = loader.get_region(chrom="chr1", center=1000000)

# Get multiple regions in parallel
regions = loader.get_batch_regions([
    ("chr1", 1000000),
    ("chr2", 2000000)
])
```

### Model Training

To train the model, you can use the `main.py` script:

```bash
python main.py train \
    --fasta_file path/to/genome.fa \
    --cell_types cell1 cell2 \
    --bam_files path/to/cell1.bam path/to/cell2.bam \
    --region_size 256 \
    --batch_size 32 \
    --num_epochs 100 \
    --output_dir results
```

Or use the training module directly in your code:

```python
from nasencoder.data.loader import NascentDataset
from nasencoder.models.encoder import NascentEncoder
from nasencoder.models.loss import MixedLoss
from nasencoder.training.trainer import ModelTrainer

# Create dataset
cell_type_data = {
    "cell_type1": ("path/to/genome.fa", "path/to/cell1.bam"),
    "cell_type2": ("path/to/genome.fa", "path/to/cell2.bam")
}

dataset = NascentDataset(
    cell_type_data=cell_type_data,
    region_size=256,
    batch_size=32
)

# Create model
model = NascentEncoder(
    dim=384,
    num_layers=6,
    num_heads=6,
    hidden_dim=1536,
    num_cell_types=len(dataset.cell_types),
    max_seq_len=256
)

# Create loss function
loss_fn = MixedLoss()

# Create trainer
trainer = ModelTrainer(
    model=model,
    loss_fn=loss_fn,
    dataset=dataset,
    device="cuda",
    learning_rate=1e-4
)

# Train model
metrics = trainer.train(
    num_epochs=100,
    steps_per_epoch=100,
    validate_every=10,
    save_path="results"
)
```

### Sparse Autoencoder Training

After training the base model, you can train a sparse autoencoder to extract interpretable features:

```bash
python main.py train_sae \
    --model_path results/epoch_100.pt \
    --hidden_dim 256 \
    --sparsity_k 20 \
    --num_epochs 50 \
    --output_dir results_sae
```

### Feature Analysis

You can analyze the features learned by the sparse autoencoder:

```bash
python main.py analyze \
    --model_path results/epoch_100.pt \
    --sae_path results_sae/sae_epoch_50.pt \
    --num_batches 10 \
    --threshold 3.0 \
    --output_dir analysis
```

## Testing

You can run the tests using pytest:

```bash
pytest
```

## Examples

The repository includes example scripts to demonstrate the usage of the model:

```bash
python -m nasencoder.examples.simple_training
```

## Contributing

Contributions are welcome! This is an active research preview. Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
