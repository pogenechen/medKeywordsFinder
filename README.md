# Medical Keywords Finder

A specialized tool for finding the most relevant medical keywords from questions to assist in literature and research paper searches.

## Overview

Medical Keywords Finder is designed to help researchers and medical professionals find the most relevant keywords from their questions to improve literature search results. By leveraging state-of-the-art deep learning models and efficient vector search techniques, it identifies the most semantically relevant medical keywords that can be used to search academic papers and medical literature.

## Features

- **Semantic Keyword Matching**: Find the most relevant medical keywords for literature search
- **BioBERT Integration**: Utilizes BioBERT, a domain-specific BERT model pre-trained on biomedical text
- **Contrastive Learning**: Implements contrastive learning for better semantic representation
- **Efficient Search**: Uses FAISS for fast vector similarity search
- **Customizable Training**: Supports model fine-tuning and custom training
- **Pre-trained Models**: Includes pre-trained models for immediate use

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd query2kw
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from medkeywordsfinder import medKeywordsFinder

# Initialize the finder
finder = medKeywordsFinder()
finder.load()

# Extract keywords from a question
question = "What are the side effects of escitalopram?"
keywords = finder.search(question, topK=5)
print(keywords)
```

### Training

To train the model from scratch:

```python
from train import train

# Train the model
train(
    data_path="path/to/training/data",
    projection_dim=256,
    batch_size=32,
    negative_sample_size=10,
    epochs=10,
    lr=1e-4
)
```

To fine-tune from a checkpoint:

```python
from train import train_from_checkpoint

train_from_checkpoint(
    data_path="path/to/training/data",
    checkpoint="path/to/checkpoint.pt",
    epochs=5,
    batch_size=32,
    negative_sample_size=10,
    lr=1e-4
)
```

## Training Data Format

To train the model, you need to prepare your data in the following JSON format:

```json
{
    "questions": [
        {
            "body": "What are the side effects of escitalopram?",
            "ideal_answer": "Common side effects of escitalopram include nausea, insomnia, fatigue, dry mouth, and sexual dysfunction. More serious side effects may include serotonin syndrome, suicidal thoughts, and abnormal bleeding."
        },
        {
            "body": "How does metformin work in diabetes?",
            "ideal_answer": "Metformin works by decreasing glucose production in the liver, increasing insulin sensitivity, and reducing glucose absorption in the intestines. It primarily targets the liver to reduce gluconeogenesis."
        }
    ]
}
```

The training data should be saved as a JSON file with:
- `questions`: A list of question-answer pairs
- Each pair contains:
  - `body`: The medical question
  - `ideal_answer`: The detailed answer/explanation

## Project Structure

```
query2kw/
├── data/                  # Training data
├── models/               # Model files
├── index/                # Search indices
├── keywords map/         # Keyword mappings
├── checkpoint/           # Model checkpoints
├── medkeywordsfinder.py  # Core functionality
├── train.py             # Training scripts
├── model.py             # Model architecture
├── dataset.py           # Data handling
├── preprocessing.py     # Data preprocessing
└── requirements.txt     # Dependencies
```

## Dependencies

- PyTorch
- Transformers
- FAISS
- NumPy
- Pandas
- BioBERT

## Applications

- Medical literature search optimization
- Research paper discovery
- Academic paper search assistance
- Medical knowledge base exploration
- Clinical research support"# medKeywordsFinder" 
