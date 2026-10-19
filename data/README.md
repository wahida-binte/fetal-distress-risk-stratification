# Data

The dataset is not committed to this repository.

1. Download `fetal_health.csv` from Kaggle: https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification
   (or with the Kaggle CLI: `kaggle datasets download -d andrewmvd/fetal-health-classification -p data/raw --unzip`)
2. Place it at `data/raw/fetal_health.csv`.

Expected shape: 2,126 rows, 22 columns (21 CTG features + `fetal_health` label in {1, 2, 3}).

Note: the file contains a small number of exact duplicate rows. Set `data.drop_duplicates: true` in the config to check how much this affects CV scores.
