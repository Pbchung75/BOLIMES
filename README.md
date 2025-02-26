# BOLIMES
BOLIMES: Boruta–LIME optiMized fEature
Selection for Gene Expression Classification

**Abstract. **
Gene expression classification is a pivotal yet challenging
task in bioinformatics, primarily due to the high dimensionality of ge-
nomic data and the risk of overfitting. To bridge this gap, we propose
BOLIMES, a novel feature selection algorithm designed to enhance gene
expression classification by systematically refining the feature subset.
Unlike conventional methods that rely solely on statistical ranking or
classifier-specific selection, we integrate the robustness of Boruta with
the interpretability of LIME, ensuring that only the most relevant and
influential genes are retained. BOLIMES first employs Boruta to filter
out non-informative genes by comparing each feature against its ran-
domized counterpart, thus preserving valuable information. It then uses
LIME to rank the remaining genes based on their local importance to
the classifier. Finally, an iterative classification evaluation determines
the optimal feature subset by selecting the number of genes that max-
imizes predictive accuracy. By combining exhaustive feature selection
with interpretability-driven refinement, our solution effectively balances
dimensionality reduction with high classification performance, offering a
powerful solution for high-dimensional gene expression analysis.
Keywords: Image Classification · Gene Expression · Boruta · LIME ·
Feature Selection


