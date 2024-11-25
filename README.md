
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

---

## Overview

This code is divided into two main parts: **contrastive learning** and **molecular analysis**. 

The code in **contrastive learning** was used for data preprocessing and the analysis of the cn-HPL output embeddings. It includes the following key steps:

- **Generate cell neighborhood tiles**: Extract tiles from whole-slide images (WSIs) centered around cells detected by Hover-Net.
- **Analysis of cn-HPL clusters (cn-HPCs)**: Samp dle tiles, compute cellular composition and interactions from each cluster, statistical analysis of cn-HPCs and survival outcome, validation in external cohort, visualization of embeddings.

The code in the **molecular analysis** was used for the analysis of the molecular data (RNA-Seq, GeoMx WTA, Visium 10x). It includes the following key steps:

- **BayesPrism analysis**: Associate cn-HPCs with deconvoluted cells from RNA-seq using BayesPrism.
- **Co-expressed gene modules**: Generate co-expressed gene modules, perform gene set enrichment analysis and associate them to cn-HPCs.
- **WTA data analysis**: Perform gene set enrichment analysis of positively correlated genes to cn-HPCs.

---

## Prerequisites

You will require different packages for different parts of the pipeline:
- For running any tiling step or tile sampling, please follow the same requirements as in `https://github.com/ncoudray/DeepPATH`.
- For running Hover-Net, please refer to `https://github.com/vqdang/hover_net`.
- For running HPL, please refer to `https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning`.

## Installation
Clone all the following repositories to your local machine and make sure to create virtual environments according to the first 3 github repositories:
```
# Tiling and data preprocessing
git clone https://github.com/ncoudray/DeepPATH.git
```

```
# Cell segmentation and classification
git clone https://github.com/vqdang/hover_net.git
```

```
# Contrastive learning model 
git clone https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning.git
```

```
# Additional code for cell-neighborhood pipeline
git clone https://github.com/hortensele/cn_HPL_LUAD.git
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
cn_HPL_LUAD/
├── contrastive_learning/                     
│   └── 01_tile_cells/
│   │   └── 01_hovernet_cell_results_information_20x.py
│   │   └── 02_tile_sort_by_cell_types_tcga_20x.py
│   └── 02_analysis/ 
│   │   └── cell_to_cell_interaction.py
│   │   └── cn_HPC_complete_analysis_tcga.Rmd
│   │   └── cn_HPC_validation_external.Rmd
│   │   └── tile_sample_cells.py
├── molecular_analysis/                   
│   └── gene_modules_validation_cohort.Rmd                
│   └── tcga_luad_gene_modules.Rmd             
└── README.md                 
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



   