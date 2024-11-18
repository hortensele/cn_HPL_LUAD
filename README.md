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

# Contrastive Learning Uncovers Cellular Interactions and Morphologies in the Tumor Microenvironment of Lung Adenocarcinoma Linked to Immunotherapy Response 

Hortense Le, Nicolas Coudray, Anna Yeaton, Sitharam Ramaswami, Afreen Karimkhan, Wei-Yi Cheng, James Cai, Tai-Hsien Ou Yang, Salman Punekar, Luis Chiriboga, Vamsidhar Velcheti, Kwok-Kin Wong, Daniel H. Sterman, Andre L. Moreira, Harvey I. Pass, Aristotelis Tsirigos

## Abstract 

Lung adenocarcinoma (LUAD) presents diverse histomorphological features within the tumor microenvironment (TME) that influence prognosis and response to immunotherapy. Leveraging contrastive learning, we developed an unbiased atlas of cell neighborhoods to systematically explore the LUAD microenvironment at the cellular scale and investigate how these cellular neighborhoods are combined to form histologic patterns. This multiscale approach enables a comprehensive understanding of both cell-specific interactions and broader histologic patterns in LUAD. Our analysis identified distinct histomorphological phenotype clusters of cellular neighborhoods (cn-HPCs) with prognostic significance. Specifically, our analysis revealed that cn-HPC 0 was associated with immune activation and correlated with favorable survival, while cn-HPC 23, was marked by necrotic tissue and immune suppression and aligned with poorer outcomes. Furthermore, using immunophenotype associations, co-expressed gene modules, and pathway enrichment, we found that cn-HPCs capture molecular signatures reflective of immune modulation, cellular growth, and inflammation, providing insights into the functional landscape of LUAD. Motivated by the association of the discovered cn-HPCs with processes related to immune system function, we hypothesized that they can also serve as biomarkers of patient response to immunotherapy. We tested this hypothesis in our immunotherapy cohort and demonstrated that specific cn-HPCs serve as predictive biomarkers for immunotherapy response, underscoring their clinical relevance for treatment stratification. Taken together, our findings emphasize the importance of fine-grained TME profiling in LUAD and highlight the potential of cn-HPCs as biomarkers for patient selection in immunotherapy.  

## Tiling H&E whole slide images (WSIs)

The WSIs are initially segmented into 224 x 224 pixels tiles without overlap at 20X magnification resulting in tiles of ~113 x 113 µm using the DeepPath pipeline.

   