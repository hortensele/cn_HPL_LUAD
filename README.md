
# Cellular Neighborhood Histomorphological Phenotype Learning (cn-HPL)

This repository contains the code used for the analysis in the paper titled "Contrastive Learning Uncovers Cellular Interactions and Morphologies in the Tumor Microenvironment of Lung Adenocarcinoma Linked to Immunotherapy Response." The code includes methods for data processing, statistical analysis, image processing, and visualization as described in the manuscript.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Usage](#usage)
5. [File Structure](#file-structure)
6. [Functions](#functions)
7. [Results](#results)
8. [Citation](#citation)
9. [License](#license)

---

## Overview

This code was used to perform [brief description of the analysis performed]. The analysis includes the following key steps:

- **Data Preprocessing**: Import and preprocess input data.
- **Image Processing**: Extract and process tiles from whole-slide images (WSIs).
- **Statistical Analysis**: Fit models and compute performance metrics like AUC.
- **Visualization**: Generate visualizations, including ROC curves.
- **Model Saving**: Save trained models and results for future use.

The functions in this repository are modular, allowing for easy adaptation to other datasets or analysis scenarios.

---

## Prerequisites

Before running the code, ensure that the following Python libraries are installed:

- **numpy**
- **pandas**
- **matplotlib**
- **seaborn**
- **scikit-learn**
- **openslide-python**
- **opencv-python**
- **tqdm**

You can install the required packages using `pip`:

```
pip install numpy pandas matplotlib seaborn scikit-learn openslide-python opencv-python tqdm
```


## Installation
1. Clone this repository to your local machine:
```
git clone https://github.com/yourusername/your-repository-name.git
cd your-repository-name
```

2. (Optional) Create a virtual environment and activate it:
```
python -m venv env
source env/bin/activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

## Usage
### Running the Analysis
1. **Load and preprocess your data**: Ensure you have a CSV (or other suitable format) containing your dataset, including columns like Centroid_x, Centroid_y, slide, and CellID.

2. **Run the analysis**: The analysis is performed using the following command:

```
python main.py --data-path data/sampled_tiles.csv --output-dir output/ --wsi-path /path/to/WSI/
```

- --data-path: Path to your data file (CSV).
- --output-dir: Directory where the processed tiles and results will be saved.
- --wsi-path: Directory containing whole-slide images.


### Key Functions
- **Data Preprocessing**: preprocess_data()
- **Tile Processing**: process_tiles()
- **Model Fitting**: fit_model()
- **Model Evaluation**: evaluate_model()
- **Visualization**: plot_roc_curve()

## File Structure
Here is a brief description of the folder structure in this repository:

```
your-repository-name/
├── data/                     # Input data files (CSV, etc.)
│   └── sampled_tiles.csv     # Sampled tile data
├── output/                   # Output folder for processed tiles and results
│   └── tiles/                # Processed image tiles
│   └── results/              # Analysis results (models, plots, etc.)
├── src/                      # Source code
│   ├── main.py               # Main script to run the analysis
│   ├── image_processing.py   # Functions for image processing
│   ├── statistical_analysis.py  # Functions for model fitting and evaluation
│   └── utils.py              # Utility functions (e.g., normalization)
└── README.md                 # This README file
```

## Functions
process_tiles(final_sampled_tiles, wsi_path, output_dir, window_size, normalize)
Generates and saves image tiles based on the sampled tiles dataframe.

fit_model(X_train, y_train)
Fits a logistic regression model to the training data.

evaluate_model(model, X_test, y_test)
Evaluates the trained model using ROC curve and computes AUC.

plot_roc_curve(fpr, tpr, auc_value)
Plots the ROC curve for model evaluation.

## Results
### Model Evaluation
The evaluation metrics for the trained models are stored in the results/ directory. You can find the performance results (e.g., ROC curves, AUC) in the following files:

results/roc_curve.png – Plot of the ROC curve.
results/model_performance.txt – A summary of the evaluation metrics.
### Tile Images
The processed image tiles will be saved in the output/tiles/ directory. Each tile will be named based on its metadata (e.g., slide_001_CellID_1234.jpeg).

## Citation
If you use this code in your research, please cite the following paper:

```csharp
[Le, H. et al. 2024. Contrastive Learning Uncovers Cellular Interactions and Morphologies in the Tumor Microenvironment of Lung Adenocarcinoma Linked to Immunotherapy Response.]
```

# Cellular Neighborhood HPL

1) Tile H&E images.
2) Run Hover-Net.
3) Generate cell-centered tiles.
4) Train Barlow Twins and project.
5) Cluster.
6) Cluster analysis (cell composition, UMAP, cell interactions).


# Molecular Analysis

1) BayesPrism
2) Gene modules
3) Gene set enrichment analysis
4) Validation dataset?



   