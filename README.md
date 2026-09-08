# Mapping Beer Flavor Space

This STAT 437 project investigates whether beers form meaningful groups based on alcohol content and review-derived sensory profiles. The analysis compares hard, density-based, and fuzzy clustering methods, then interprets the resulting groups using flavor composition, beer styles, representative beers, and membership uncertainty.

The complete analysis is in [`Stat 437 final project.ipynb`](./Stat%20437%20final%20project.ipynb).

## Research questions

1. Can beers be meaningfully grouped using their flavor profiles and alcohol content?
2. What broad and fine-grained flavor groupings emerge?

## Dataset

`beer_profile_and_ratings.csv` contains 3,197 unique beers from 934 breweries and 111 beer styles. Each record includes identifiers, ABV, style-level IBU ranges, eleven sensory descriptors, average review scores, and the number of consumer reviews.

The sensory fields are counts of descriptor words found in up to 25 reviews per beer. They are not direct intensity ratings. Raw count magnitude can therefore reflect the amount of descriptive text as well as flavor composition, which motivates the profile transformation used in the analysis.

Dataset source: [Beer Profile and Ratings Data Set](https://www.kaggle.com/datasets/ruthgn/beer-profile-and-ratings-data-set), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Analysis pipeline

### 1. Data-quality checks

The notebook checks missing values, duplicate rows, duplicate beer identifiers, empty sensory profiles, and extreme ABV values. It then:

- removes 12 beers whose sensory descriptors are all zero;
- excludes three specialty beers above 20% ABV to focus on the typical beer range;
- retains 3,182 beers for modeling; and
- stores the CSV row number as `source_row` so metadata and model results remain aligned.

The high-ABV removal is a scope decision, not a claim that those records are erroneous.

### 2. Exploratory analysis

The notebook summarizes feature distributions, zero frequencies, dominant sensory descriptors, and feature correlations. Malt is the most common dominant descriptor, while sour/fruit and bitter/hoppy show related behavior. These patterns suggest overlapping flavor dimensions rather than sharply separated categories.

### 3. Sensory-profile transformation

Total modeled descriptor counts vary widely across beers. On standardized raw counts, the absolute Spearman association between the first principal component and total descriptor count is 0.852, showing that count volume strongly affects the geometry.

The modeling pipeline therefore:

1. excludes `Salty` because it is zero for 59.08% of retained beers and has little variation;
2. divides each beer's remaining sensory counts by its total descriptor count;
3. applies a square-root, or Hellinger, transformation;
4. combines the transformed profile with ABV; and
5. standardizes all eleven model features.

After transformation, the PC1/count-total association falls to 0.202. Every clustering algorithm uses this same feature matrix.

### 4. Clusterability diagnostics

PCA summarizes global linear structure, while two seeded t-SNE projections provide low-dimensional visual checks. t-SNE is used only for display because it can exaggerate separation; models are fitted and evaluated in the complete eleven-dimensional space.

### 5. K-means baseline

K-means is evaluated for 2–10 clusters using silhouette, Calinski–Harabasz, Davies–Bouldin, and minimum cluster size. The strongest hard-partition evidence favors two clusters:

| Metric | Result |
|---|---:|
| Silhouette | 0.230 |
| Calinski–Harabasz | 926.673 |
| Davies–Bouldin | 1.736 |
| Smallest cluster | 1,319 beers |

This indicates two broad flavor families, although a finer resolution remains useful for description.

### 6. DBSCAN density analysis

DBSCAN is evaluated over a grid of neighborhood radii and minimum-sample values. Candidate models must cover at least 80% of the data and contain no cluster smaller than 25 beers. The selected diagnostic configuration uses `eps=1.5` and `min_samples=12`:

| Metric | Result |
|---|---:|
| Dense clusters | 2 |
| Coverage | 81.7% |
| Noise points | 581 |
| Silhouette on clustered points | 0.350 |
| Small density cluster | 138 beers |

The smaller density cluster is a concentrated sour/fruit-rich core. DBSCAN therefore supports one especially distinctive flavor island rather than a complete segmentation of all beers.

### 7. Fuzzy C-means selection

Fuzzy C-means is evaluated for 2–8 clusters and fuzziness values of 1.3, 1.5, and 1.7. The comparison uses silhouette, fuzzy partition coefficient, partition entropy, Xie–Beni index, smallest cluster size, and adjusted Rand index across five random initializations.

Two clusters provide the strongest coarse separation. A five-cluster model with `m=1.3` provides the best tested Xie–Beni score and a more useful descriptive resolution:

| Metric | Result |
|---|---:|
| Silhouette | 0.208 |
| Fuzzy partition coefficient | 0.741 |
| Median maximum membership | 0.896 |
| Maximum membership below 0.50 | 8.2% of beers |
| Mean/minimum seed-stability ARI | 1.000 / 1.000 |

## Results

The selected model produces five flavor regions:

| Cluster | Beers | Main characteristics |
|---|---:|---|
| Strong / sweet-malty | 616 | Highest mean ABV; elevated sweet, malt, and alcohol descriptors |
| Fruity / spiced | 531 | Fruit, spice, wheat-beer, and Belgian-style tendencies |
| Sour / fruit-forward | 252 | Highest sour and fruit shares; most locally distinct region |
| Bitter / hoppy | 934 | Elevated bitter and hop descriptors |
| Malty / full-bodied | 849 | Highest malt and body shares |

DBSCAN's small density cluster falls primarily within the fuzzy sour/fruit-forward group, giving independent support to that interpretation. The other groups are connected by beers with mixed membership and are better understood as overlapping regions than rigid natural classes.

## Outcome

Beer flavor in this dataset is a connected spectrum with reproducible structure. Two broad families provide the strongest separation, while five fuzzy clusters give a practical vocabulary for finer flavor differences. Fuzzy memberships preserve cross-profile beers instead of forcing every observation into an equally certain category.

For a recommendation system, the membership vectors could support similarity searches and controlled exploration between flavor regions. Beers near cluster boundaries can be presented as hybrids rather than mislabeled as pure members of one group.

## Limitations

- Sensory features reflect reviewer language and behavior as well as beer characteristics.
- Hellinger normalization emphasizes relative composition and removes absolute descriptor volume.
- `Salty` and beers above 20% ABV are excluded from the model.
- Beer styles and dominant descriptors are interpretive metadata, not ground truth.
- The dataset is not a representative census of all beers or breweries.
- t-SNE plots do not establish cluster validity.

## Repository structure

```text
.
├── Stat 437 final project.ipynb
├── beer_profile_and_ratings.csv
├── README.md
├── requirements.txt
└── scripts
    ├── build_notebook.py
    └── execute_notebook.py
```

`build_notebook.py` creates the notebook source deterministically. `execute_notebook.py` runs every code cell in a clean kernel and saves the outputs.

## Reproducing the analysis

Python 3.10 or later is recommended.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab "Stat 437 final project.ipynb"
```

Run all notebook cells from top to bottom. Random states are fixed for clustering and visualization.

To rebuild and execute the notebook programmatically:

```powershell
python scripts\build_notebook.py
python scripts\execute_notebook.py
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab "Stat 437 final project.ipynb"
```

## Potential extensions

- compare results with `Salty` included and with high-ABV specialty beers restored;
- bootstrap beers or reviews to quantify cluster uncertainty;
- use fuzzy memberships in a beer recommendation prototype;
- examine whether review count or brewery representation affects membership; and
- test alternative compositional transformations and distance measures.
