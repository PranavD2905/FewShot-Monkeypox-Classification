# Few-Shot Prototypical Networks for Monkeypox Skin Lesion Classification 🐒

**A comparative study of CNNs and Prototypical Networks for disease classification from limited data, inspired by the challenges of emerging diseases like Monkeypox.**

---

## 📖 Project Overview

This project, based on the research paper "Few-Shot Prototypical Networks for Monkeypox Skin Lesion Classification Using Limited Data," provides a framework to train, evaluate, and compare deep learning models for image classification under data-scarce conditions. The core idea is to investigate the effectiveness of Prototypical Networks—a metric-learning based few-shot technique—against traditional Convolutional Neural Network (CNN) baselines when trained on a limited dataset of Monkeypox skin lesion images.

The motivation stems from the real-world challenge of diagnosing new and emerging diseases where large, annotated datasets are not yet available. This repository contains all the necessary code to reproduce the experiments, compare different model backbones, and analyze the results.

---

## ✨ Key Features

*   **🔬 Prototypical Networks:** Implementation of Prototypical Networks for few-shot classification.
*   **🧩 Multiple CNN Backbones:** Supports various architectures like `ResNet`, `DenseNet`, `EfficientNet`, and `MobileNet` for comprehensive comparison.
*   **📊 Comparative Framework:** Easily train and evaluate both baseline classifiers and Prototypical Networks to compare their performance.
*   **⚙️ Config-Driven Experiments:** Use simple `.yaml` files to define and manage experiments, making it easy to tweak hyperparameters and model configurations.
*   **📈 Results Analysis:** Includes scripts to generate confusion matrices, comparison plots, and detailed performance metrics.
*   **📂 Flexible Data Handling:** A straightforward data loader for organizing and preprocessing image datasets.

---

## 🛠️ Tech Stack

*   **Backend & Machine Learning:**
    *   [Python](https://www.python.org/)
    *   [PyTorch](https://pytorch.org/)
    *   [Torchvision](https://pytorch.org/vision/stable/index.html)
    *   [NumPy](https://numpy.org/)
    *   [Scikit-learn](https://scikit-learn.org/)
    *   [Matplotlib](https://matplotlib.org/)
    *   [Seaborn](https://seaborn.pydata.org/)
*   **Tools:**
    *   [PyYAML](https://pyyaml.org/)
    *   [tqdm](https://github.com/tqdm/tqdm)

---

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

Make sure you have the following installed:
*   Python (3.8 or higher recommended)
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/fewshot_monkeypox.git
    cd fewshot_monkeypox
    ```

2.  **Install dependencies:**
    It is recommended to create a virtual environment first.
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
    Then, install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Environment Variables

This project does not require any environment variables for its core functionality. All configurations are handled through the `.yaml` files in the root directory.

---

## Usage

To run an experiment, you can use the main training scripts with a specified configuration file. For example, to train a Prototypical Network with a DenseNet backbone:

```sh
python train.py --config config_densenet_fix.yaml
```

To train a baseline classifier:

```sh
python train_classifier.py --config config_classifier_densenet.yaml
```

After training, you can evaluate the models using the evaluation scripts:

```sh
python evaluate.py --config config_densenet_fix.yaml
```

---

## 📁 Project Structure

Here is a simplified overview of the project's directory structure:

```
fewshot_monkeypox/
├── data/
│   ├── infected/
│   └── normal/
├── models/
│   ├── backbone.py         # CNN backbone models
│   ├── classifier.py       # Baseline classifier model
│   └── proto_net.py        # Prototypical Network model
├── outs/                   # Output directory for logs and results
├── k/                      # Saved model checkpoints
├── *.yaml                  # Configuration files for experiments
├── main.py                 # Main script (can be adapted for entry)
├── train.py                # Training script for Prototypical Networks
├── train_classifier.py     # Training script for baseline classifiers
├── evaluate.py             # Evaluation script
├── data_loader.py          # Data loading and preprocessing
├── requirements.txt        # Project dependencies
└── README.md               # You are here!
```

---

## 📞 Contact

Pranav Deshpande
*   **GitHub:** [PranavD2905](https://github.com/PranavD2905)
*   **LinkedIn:** [pranav2905](https://www.linkedin.com/in/pranav2905/)
