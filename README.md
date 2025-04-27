# **Iris Classification using Clustering**

This repository contains a Jupyter Notebook (`Iris_classification (1).ipynb`) that demonstrates the application of various clustering algorithms (K-Means, Hierarchical Clustering, and DBSCAN) to the Iris dataset for classification purposes. The project aims to group the Iris flowers into their respective species based on their physical measurements.

## **Table of Contents**

* [Project Description](#project-description)

* [Dataset](#dataset)

* [Methodology](#methodology)

* [Prerequisites](#prerequisites)

* [How to Run](#how-to-run)

* [Key Steps in the Code](#key-steps-in-the-code)

* [Results](#results)

* [Visualizations](#visualizations)

* [Future Improvements](#future-improvements)

* [License](#license)

## **Project Description**

The goal of this project is to explore unsupervised learning techniques, specifically clustering, to group instances of the Iris dataset. By applying K-Means, Hierarchical Clustering, and DBSCAN, the project seeks to identify distinct clusters within the data and compare how well these clusters align with the known species labels of the Iris flowers. The notebook covers data loading, exploration, preprocessing, applying clustering algorithms, and evaluating the results using metrics like the Silhouette Score.

## **Dataset**

The project uses the **Iris dataset**, a classic dataset in machine learning.

* It contains 150 samples of Iris flowers.

* There are three species: Iris-setosa, Iris-versicolor, and Iris-virginica, with 50 samples each.

* Each sample has four features: sepal length, sepal width, petal length, and petal width (all in centimeters).

* The dataset is expected to be in a CSV file named `Iris1.csv`, with columns for 'Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', and 'Species'.

## **Methodology**

The analysis follows these main steps:

1. **Load Libraries:** Import necessary Python libraries for data manipulation, visualization, and clustering.

2. **Load Data:** Read the `Iris1.csv` file into a pandas DataFrame.

3. **Data Exploration:** Perform initial data inspection, including viewing the first few rows, checking data types (`.info()`), getting descriptive statistics (`.describe()`), checking unique values (`.nunique()`), and summing missing values (`.isna().sum().sum()`). Visualize the distribution of numerical features using a boxplot and explore correlations using a heatmap.

4. **Data Preprocessing:**

   * Set the 'Id' column as the index.

   * Separate the features (excluding 'Species') into a new DataFrame.

   * Apply `RobustScaler` to scale the features.

5. **Clustering:**

   * Apply K-Means clustering with 3 clusters.

   * Apply Agglomerative Clustering (Hierarchical) with 3 clusters.

   * Apply DBSCAN clustering (with specified `eps` and `min_samples`).

6. **Evaluation:**

   * Calculate the Silhouette Score for each clustering algorithm to evaluate the quality of the clusters.

   * Map the cluster labels from each algorithm to the original species names for comparison (this mapping is based on observed patterns in the notebook).

7. **Comparison:** Compare the results of the different clustering algorithms based on their Silhouette Scores and the alignment of their clusters with the true species labels.

## **Prerequisites**

Make sure you have Python installed. The following Python libraries are required:

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, normalize, RobustScaler
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix
