# CNN and ResNet Evaluation: Facial Expression Recognition with Limited Data

We compare a plain CNN and a ResNet, both built from scratch in PyTorch, at recognizing 8 facial expressions. We test how much data augmentation helps each model when training data is limited, by training on 10%, 25%, 50% and 100% of the data with augmentation off and on, using SGD and AdamW (32 runs in total).

**Main result:** augmentation helped the ResNet far more than the plain CNN (up to +10.6 points versus up to +3.2). The best model, an augmented ResNet, reached 83.1% test accuracy.

## Files

- `facial_recognition_dl.ipynb`: the full project. Data download and cleaning, both models, training, and results.
- `common.txt`: the Python packages needed to run it locally.
- `project_data/results_ferplus/`: the saved results of all 32 runs (one JSON file each), so the results can be reproduced without retraining.

## Data

- **Images:** FER-2013 (Goodfellow et al., 2013), downloaded automatically from the Kaggle mirror [deadskull7/fer2013](https://www.kaggle.com/datasets/deadskull7/fer2013).
- **Labels:** FER+ (Barsoum et al., 2016), downloaded automatically from [Microsoft's FER+ GitHub](https://github.com/microsoft/FERPlus).

No manual download is needed. The notebook gets both files when it runs.

## Setup

**Google Colab:** click the badge above. Colab already has everything except `kagglehub`, which the first cell installs. For training, turn on a GPU under Runtime > Change runtime type.

**Running locally (VS Code or Jupyter):**

1. Install Python 3.12.
2. Clone the repository:
   ```
   git clone https://github.com/NathanDev2/cnn_and_resnet_evaluation.git
   cd cnn_and_resnet_evaluation
   ```
3. Install the packages:
   ```
   pip install -r common.txt
   ```
4. Open `facial_recognition_dl.ipynb` and select Python 3.12 as the kernel.

An NVIDIA GPU is recommended. The packages in `common.txt` install PyTorch with CUDA 12.8 support.

## Running the notebook

- **Reproduce the results without retraining:** keep the `results_ferplus` folder in place (locally in `project_data/`, or in Colab at `My Drive/facial-expression-dl/`). Run the notebook from the top. Cell 6d will skip every finished run, and Section 7 rebuilds all tables and charts.
- **Train from scratch:** remove or rename the `results_ferplus` folder and run the notebook from the top. All 32 runs take about 1 hour on an RTX 5070 Ti, and longer on a Colab T4.

## Tested with

Python 3.12.10, PyTorch 2.11.0 (CUDA 12.8), torchvision 0.26.0, NVIDIA RTX 5070 Ti, Windows 11. Random seed 42.

## Authors

Ismael Angulo
Nathan Darrett
Connor Todd
