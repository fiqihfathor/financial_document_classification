<p align="center">
  <a href="" rel="noopener">
 <img src="/docs/assets/icons/image.webp" alt="Project logo" style="border-radius: 50%; width: 200px; height: 200px;">
</p>

<h3 align="center">Document Classification</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)](https://github.com/fiqihfathor/financial_document_classification)
[![GitHub Issues](https://img.shields.io/github/issues/fiqihfathor/financial_document_classification.svg)](https://github.com/fiqihfathor/financial_document_classification/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/fiqihfathor/financial_document_classification.svg)](https://github.com/fiqihfathor/financial_document_classification/pulls)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/fiqihfathor/financial_document_classification/blob/main/LICENSE)

</div>

---

<p align="center">
    An image-based document classification system that automatically categorizes documents into predefined classes using advanced deep learning models like EfficientNet, ResNet, and Vision Transformers (ViT).
</p>


## 📝 Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Built Using](#built_using)
- [Authors](#authors)

## 🧐 About <a name = "about"></a>

This project provides an image-based document classification system that automatically classifies document images into predefined categories using deep learning models like EfficientNet, ResNet, and Vision Transformers (ViT).

## 🏁 Getting Started <a name = "getting_started"></a>
These instructions will guide you through setting up the project on your local machine for development and testing purposes.

### Prerequisites

Before you begin, make sure you have the following installed:

1. **Git**  
   Git is required to clone the repository:  
   [Download Git](https://git-scm.com/)
   **Verify Git Installation**  
   ```sh
   git --version
   ```
2. **UV**  
   An extremely fast Python package and project manager, written in Rust.
   You can read the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/).
   **Verify uv Installation**  
   ```sh
   uv version
   ```
3. **Make**  
   Make is a build utility that simplifies the process of building, testing, and packaging software.  
   You can read the [Make documentation](https://www.gnu.org/software/make/manual/).

   **Verify Make Installation**  
   Run the following command to check if Make is installed:
   ```sh
   make --version
   ```

### Clone Project
Clone the project from GitHub:
  ```sh
    git clone https://github.com/fiqihfathor/financial_document_classification.git
    cd financial_document_classification
  ```
### Installing

Install the project using the following command:

```sh
uv sync
```

## 🔧 Running the tests <a name = "tests"></a>

Run the tests using the following command:

```sh
make test
```


## 🎈 Usage <a name="usage"></a>

**Donwload Dataset**
```sh
make dataset
```

**Train Model**
```sh
make train
```
You can change the configuration in `config/config.yml`

**Test API**
```sh
make server
```
and you can access it on http://localhost:8000


## ⛏️ Built Using <a name = "built_using"></a>

- **[Python](https://www.python.org/)**: The powerhouse of programming languages, enabling versatility and efficiency.
- **[PyTorch](https://pytorch.org/)**: Cutting-edge deep learning framework for building complex models with ease.
- **[FastAPI](https://fastapi.tiangolo.com/)**: The lightning-fast web framework to power your API with speed and simplicity.
- **[UV](https://docs.astral.sh/uv/getting-started/installation/)**: An ultra-fast project manager that makes dependency management a breeze.
- **[Make](https://www.gnu.org/software/make/manual/)**: The trusted build utility to streamline your software development process.
- **[Git](https://git-scm.com/)**: The version control system that keeps your code organized and in control.
- **[MLflow](https://mlflow.org/)**: The open-source platform for managing and tracking machine learning experiments.
- **[Loguru](https://loguru.readthedocs.io/en/stable/)**: The most powerful and user-friendly logging library to simplify your code’s logging.


## ✍️ Authors <a name = "authors"></a>

- [@fiqihfathor](https://github.com/fiqihfathor) 
