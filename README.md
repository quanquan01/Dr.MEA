# Dr.MEA: Leveraging Large Language Model to Enhance Multi-modal Entity Alignment


[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains the implementation of **Dr.MEA**, a novel approach for multi-modal entity alignment that leverages Large Language Models (LLMs) to enhance the alignment process. The method combines traditional embedding-based approaches with LLM reasoning capabilities to achieve superior performance on multi-modal knowledge graphs.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Architecture](#architecture)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## 🔍 Overview

Multi-modal entity alignment is a crucial task in knowledge graph integration, where the goal is to identify entities across different knowledge graphs that refer to the same real-world object. Traditional approaches rely solely on embedding-based similarity, which may not capture the rich semantic information available in multi-modal data.

**Dr.MEA** addresses this limitation by:

1. **Base Ranking**: Using traditional multi-modal contrastive learning to generate initial candidate rankings
2. **LLM Re-ranking**: Employing Large Language Models to reason about entity similarity based on descriptions, structural information, and attributes
3. **Multi-modal Fusion**: Integrating visual, textual, and structural features through attention mechanisms

## ✨ Features

- **Multi-modal Integration**: Combines visual, textual, and structural information
- **LLM-Enhanced Reasoning**: Leverages Large Language Models for sophisticated entity comparison
- **Flexible Architecture**: Supports various LLM backends (GPT, GLM, LLaMA, etc.)
- **Comprehensive Evaluation**: Provides detailed metrics including Hits@K and MRR
- **Easy Configuration**: Simple command-line interface for different experimental settings

## 🚀 Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ GPU memory (recommended)

### Setup


1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Download datasets:**
   - Follow the instructions in the [MMKB repository](https://github.com/mniepert/mmkb) to download the multi-modal knowledge graph datasets
   - Place the datasets in the `data/` directory

## 📊 Dataset

This project uses the **MMKB (Multi-Modal Knowledge Graphs)** dataset, which includes:

- **FB15K**: Freebase subset with 15K entities
- **YAGO15K**: YAGO subset with 15K entities  
- **DBpedia15K**: DBpedia subset with 15K entities

Each dataset contains:
- **Triples**: Structural relationships between entities
- **Images**: Visual representations of entities
- **Descriptions**: Textual descriptions
- **Attributes**: Entity properties and characteristics

For more details, visit: [MMKB Dataset](https://github.com/mniepert/mmkb)

## 🎯 Usage

### Step 1: Base Ranking

First, run the base ranking model to generate initial candidate rankings:

```bash
python src/run_test.py --file_dir data/FBDB15K --epochs 600 --cuda
```

**Parameters:**
- `--file_dir`: Path to the dataset directory
- `--epochs`: Number of training epochs
- `--cuda`: Use GPU acceleration

### Step 2: LLM Re-ranking

Configure your LLM API key in `src/LLM_reasoning_new.py`:

```python
api_key = "your-api-key"  # Replace with your actual API key
```

Then run the LLM re-ranking process:

```bash
python src/LLM_reasoning_new.py --data FBDB15K --ent 0 --neigh 5
```

**Parameters:**
- `--data`: Dataset name (FBDB15K, FBYG15K)
- `--ent`: Number of entities to evaluate (0 for all)
- `--neigh`: Number of neighbor relations to consider

### Complete Pipeline

```bash
# Step 1: Train base model and generate embeddings
python src/run_test.py --file_dir data/FBDB15K --epochs 300 --cuda

# Step 2: Run LLM re-ranking
python src/LLM_reasoning_new.py --data FBDB15K --ent 0 --neigh 5 --log_print
```

## 🏗️ Architecture

### Base Model (MCLEA)
The base model implements a multi-modal contrastive learning approach:

- **Multi-modal Encoder**: Processes visual, textual, and structural features
- **Graph Neural Networks**: Captures structural relationships
- **Attention Fusion**: Combines different modalities
- **Contrastive Learning**: Learns entity representations

### LLM Re-ranking Module
The LLM module performs sophisticated reasoning:

- **Entity Comparison**: Analyzes descriptions, attributes, and structural information
- **Similarity Scoring**: Provides detailed similarity scores across multiple dimensions
- **Iterative Refinement**: Uses multiple reasoning iterations for better accuracy




## 📚 Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{zhu2024drmea,
  title = {Dr.MEA: Leveraging Large Language Model to Enhance Multi-modal Entity Alignment},
  author = {Zhu, Jizhao and Jin, Quanxin and Li, Xiang and Pan, Xinlong and Lin, Lei},
  booktitle = {Proceedings of the ...},
  year = {2025},
  pages = {--},
  url = {--}
}
```

## 🙏 Acknowledgments

This work builds upon the excellent foundation provided by:

- **MCLEA**: [Multi-modal Contrastive Representation Learning for Entity Alignment](https://github.com/lzxlin/MCLEA/blob/main/README.md)
- **MMKB Dataset**: [Multi-Modal Knowledge Graphs](https://github.com/mniepert/mmkb)

We thank the authors of these works for their valuable contributions to the field.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📞 Contact

For questions and support, please open an issue in this repository or contact the authors.

---

**Note**: This project is part of ongoing research. Please check back for updates and additional features.
