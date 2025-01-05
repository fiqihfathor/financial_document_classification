# Financial Document Classification

An image-based document classification system using deep learning models.

## Table of Contents

- [About](about.md)
- [Getting Started](getting_started/prerequisites.md)
- [Usage](usage/dataset.md)
- [Authors](authors.md)
- [License](license.md)

## Project Structure

The project is structured as follows:

```bash
.
├── config
│   ├── config.yml
│   └── logging.yml
├── data
├── docs
├── logs
├── models
├── mlruns
├── scripts
├── src
│   ├── ml
│   │   ├── models
│   │   │   ├── __init__.py
│   │   │   ├── efficientnet.py
│   │   │   ├── resnet.py
│   │   │   └── visiontransformer.py
│   │   ├── preprocessing
│   │   │   ├── __init__.py
│   │   │   ├── dataset.py
│   │   │   └── dataloader.py
│   │   ├── utils
│   │   │   ├── __init__.py
│   │   │   ├── mlflow_utils.py
│   │   │   └── train_utils.py
│   │   └── __init__.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── config.py
│   └── __init__.py
├── test
├── .gitignore
├── .python-version
├── Makefile
├── pyproject.toml
├── uv.lock
└── LICENSE
```