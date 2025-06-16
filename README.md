
# Cellular Neighborhood Histomorphological Phenotype Learning (cn-HPL)

This repository contains the code used for the analysis in the paper titled "Contrastive Learning Uncovers Cellular Interactions and Morphologies in the Tumor Microenvironment of Lung Adenocarcinoma Linked to Immunotherapy Response." Below is a brief description of the repository structure, pipeline, and how each script contributes to the analysis.

---

## Repository Structure

```
cn_HPL_LUAD/
├── contrastive_learning/                     
│   ├── 01_tile_cells/
│   │   ├── 01_hovernet_cell_results_information_20x.py
│   │   ├── 02_tile_by_cell_types_external_20x.py
│   │   ├── 02_tile_sort_by_cell_types_tcga_20x.py
│   ├── 02_analysis/
│   │   ├── cell_to_cell_interaction.py
│   │   ├── cn_HPC_complete_analysis_tcga.Rmd
│   │   ├── cn_HPC_validation_external.Rmd
│   │   ├── Colocalization_cell_cnHPC.ipynb
│   │   ├── GraphAnalysis_HPC_fingerprints.ipynb
│   │   ├── tile_sample_cells.py
│   │   ├── visualize_hpc_cells.py
├── molecular_analysis/                   
│   ├── gene_modules_validation_cohort.Rmd                
│   ├── tcga_luad_gene_modules.Rmd             
└── README.md             
```

---


### Key Directories and Scripts

#### `contrastive_learning/01_tile_cells/`
- **`01_hovernet_cell_results_information_20x.py`**  
  Summarizes cell features and coordinates from Hover-Net outputs for each whole slide image.

- **`02_tile_sort_by_cell_types_tcga_20x.py`**  
  Generates and samples tiles around specific cell types' coordinates for the training/validation/test sets of TCGA-LUAD.

- **`02_tile_by_cell_types_external_20x.py`**
  Generates tiles around all cells' coordinates per slide (useful for validation cohorts).

#### `contrastive_learning/02_analysis/`
- **`cell_to_cell_interaction.py`**  
  Computes cell-cell interactions for cn-HPC tiles.

- **`visualize_hpc_cells.py`**  
  Visualize tiles from cn-HPC of interest and tumor/inflammatory cells (from Hover-Net) overlaid over a whole slide image.
  
- **`tile_sample_cells.py`**  
  Randomly samples and visualizes tiles from each cn-HPC.

- **`cn_HPC_complete_analysis_tcga.Rmd`**  
  Performs UMAP projections, cell composition analysis, visualization of cell-cell interactions, and survival analysis.

- **`cn_HPC_validation_external.Rmd`**  
  Validates cn-HPCs in an external cohort using UMAPs, cell composition analysis, and immunotherapy response analysis.

- **`Colocalization_cell_cnHPC.ipynb`** 
  Colocalization analysis for each cn-HPC with immune and tumor regions.

- **`GraphAnalysis_HPC_fingerprints.ipynb`** 
  Graph analysis of each HPC with respect to their cn-HPC interactions.

#### `molecular_analysis/`
- **`tcga_luad_gene_modules.Rmd`**  
  Generates co-expressed gene modules, performs gene set enrichment, and correlates cn-HPC composition with module scores and deconvoluted cell composition.

- **`gene_modules_validation_cohort.Rmd`**  
  Analyzes GeoMx WTA data, correlating gene expression with cn-HPC composition and performing gene set enrichment in the external cohort.

---

## Full Pipeline Overview

1. **Tiling of Whole Slide Images**  
   Use the [DeepPATH repository](https://github.com/ncoudray/DeepPATH) to tile whole slide images.

2. **Run Hover-Net on Tiles**  
   Use [Hover-Net](https://github.com/vqdang/hover_net) to extract cell features and coordinates.

3. **Summarize Cell Features**  
   Run `01_hovernet_cell_results_information_20x.py` to summarize the Hover-Net results for each whole slide image.

4. **Generate and Sample Tiles**  
   Use `02_tile_sort_by_cell_types_tcga_20x.py` to generate and sample tiles around specific cell types.

5. **Create HDF5 File**  
   Use the [DeepPATH repository](https://github.com/ncoudray/DeepPATH) to create HDF5 files from the sampled tiles.

6. **Run HPL Pipeline**  
   Leverage [Histomorphological Phenotype Learning (HPL)](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning) to:
   - Train Barlow Twins.
   - Project tiles into a latent space.
   - Cluster tiles with Leiden clustering to define cn-HPCs.

7. **Morphological and Clinical Analysis of cn-HPCs**  
   - Use `cell_to_cell_interaction.py` to compute cell-cell interactions.
   - Use `tile_sample_cells.py` for tile visualization of cn-HPCs.
   - Use `visualize_hpc_cells.py` for cn-HPCs and cells visualization overlaid over whole slide images.
   - Run `cn_HPC_complete_analysis_tcga.Rmd` for embedding visualization in UMAP, cell composition analysis, and survival analysis.

8. **Validation in External Cohort**  
   - Use `cn_HPC_validation_external.Rmd` to validate cn-HPC morphological and clinical findings in an independent cohort.

9. **BayesPrism Cell Deconvolution**  
   Use [BayesPrism](https://github.com/ninashenker/LUAD) to deconvolute bulk RNA-Seq data in TCGA-LUAD.

10. **Gene Module Analysis**  
    - Run `tcga_luad_gene_modules.Rmd` to generate and analyze co-expressed gene modules, and analyze deconvoluted cells from Step 9.
    - Use `gene_modules_validation_cohort.Rmd` for external cohort gene module validation.

---

## Citation

If you use this repository in your research, please cite our paper.  
```csharp
[Le, H. et al. 2024. Contrastive Learning Uncovers Cellular Interactions and Morphologies in the Tumor Microenvironment of Lung Adenocarcinoma Linked to Immunotherapy Response.]
```
For questions or collaborations, feel free to contact us or open an issue. 







   
