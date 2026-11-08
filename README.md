# Animal Species Classification and Regression

A two-part machine learning project on a synthetically generated animal dataset. The workflow covers data generation, exploratory analysis, preprocessing, classification, regression, model comparison, and a Gradio inference interface.

| Part | Author | Focus |
| --- | --- | --- |
| Part I | Vlad-Andrei Darie | Synthetic dataset, EDA, Logistic Regression species classifier, Gradio UI |
| Part II | Theodor Ioan Pirnog | sklearn pipelines, EDA artifacts, weight regression, endangered-species classification |

Course: PCLP3, group 311CC.

---

## Problem statement

The dataset describes 1000 animal records with mixed numeric and categorical attributes. Two supervised tasks are solved on the same table:

1. **Multiclass classification** of `Nume Specie` (species name) from the remaining features (Part I).
2. **Regression** of `Greutate (Kg)` (weight) and **binary classification** of `Pericol de disparitie` (endangered: `Da` / `Nu`) (Part II).

---

## Dataset

Nine species are sampled uniformly: Wolf (`Lup`), Camel (`Camila`), Monkey (`Maimuta`), Parrot (`Papagal`), Owl (`Bufnita`), Ostrich (`Strut`), Viper (`Vipera`), Lizard (`Soparla`), Boa (`Sarpe Boa`).

| Column | Type | Generation rule |
| --- | --- | --- |
| `Nume Specie` | categorical | `np.random.choice` over the 9 names |
| `Categorie` | categorical | Deterministic map: mammal / bird / reptile |
| `Greutate (Kg)` | numeric | Species-specific `uniform` interval; `NaN` for Parrot, Owl, Viper, Boa |
| `Durata de viata (ani)` | integer | Species-specific `randint` interval |
| `Habitat` | categorical | Fixed for some species; Bernoulli-style random habitat for others |
| `Pericol de disparitie` | categorical | Independent draw with P(`Da`) = 0.2 |
| `Viteza (Km/h)` | numeric | Species-specific interval; `NaN` for Monkey, Parrot, Owl, Ostrich |
| `Inaltime (m)` | numeric | Species-specific interval; `NaN` for Wolf, Parrot, Owl, Viper, Boa |
| `Culoare` | categorical | Species-specific color probabilities |

Missing values are injected on purpose for selected species so that imputation can be demonstrated.

Reproducibility: `np.random.seed(42)` before generation. Split: `train_test_split(..., test_size=0.4, random_state=42)` yields **600 train / 400 test** rows, written as `train.csv` and `test.csv` without an index.

---

## Repository layout

```
.
├── I/
│   ├── Tema1_PCLP3_DarieVladAndrei_311CC.ipynb   # Part I notebook
│   └── tema1_pclp3_darievladandrei_311cc.py      # Part I exported script
├── II/
│   ├── 1_normalizare.py
│   ├── 2_analizare.py
│   ├── 3_antrenare.py
│   └── 4_evaluare.py
├── train.csv
└── test.csv
```

Part II scripts expect `train.csv` and `test.csv` in the current working directory.

---

## Part I — Species classification

Implemented in `I/Tema1_PCLP3_DarieVladAndrei_311CC.ipynb` (and the exported `.py`). The notebook is sequential: generate, clean, inspect, encode, train, evaluate, then serve.

### 1. Synthetic generation

Each feature is produced by a dedicated generator that iterates over the species name list. Numeric ranges are species-dependent (for example camel weight in `[0.1, 700]`, lizard weight in `[0.1, 0.3]`). Habitat and color use nested uniform draws with different thresholds per species, so the same name can still produce different categorical values.

### 2. Missing-value handling

Counts and percentages of `NaN` are printed for train and test. Numeric holes in `Greutate (Kg)`, `Viteza (Km/h)`, and `Inaltime (m)` are filled with the **column mean** of that split:

```python
train['Greutate (Kg)'] = train['Greutate (Kg)'].fillna(train['Greutate (Kg)'].mean())
```

The same procedure is applied independently on the test set. Imputed tables overwrite `train.csv` / `test.csv`.

### 3. Exploratory analysis

- **Descriptive stats:** `DataFrame.describe()` on numeric columns; `select_dtypes(include='object').describe()` on categoricals.
- **Distributions:** matplotlib histograms for the four numeric columns; bar plots from `value_counts()` for species, category, habitat, endangered status, and color.
- **Outliers:** seaborn boxplots on each numeric column (visual inspection only; no IQR filter is applied).
- **Correlations:** Pearson matrix on numeric columns, plotted as a seaborn heatmap (`cmap='coolwarm'`, `center=0`).
- **Target relations:** scatter plot of weight vs height, colored by species.

### 4. Feature matrix

Target is species name:

```python
X_train = train.drop(columns=['Nume Specie'])
y_train = train['Nume Specie']
```

Categorical columns `Categorie`, `Habitat`, `Pericol de disparitie`, `Culoare` are encoded with `pd.get_dummies(..., drop_first=True)` (dummy encoding, first level dropped). Numeric columns are standardized with `StandardScaler`: `fit_transform` on train, `transform` on test, so test statistics do not leak into the scaler.

### 5. Model

```python
LogisticRegression(max_iter=1000, random_state=42)
```

Multinomial logistic regression (sklearn’s default multi-class handling for this estimator) is fit on the encoded, scaled matrix.

### 6. Evaluation

Predictions on the encoded test set are scored with:

- `accuracy_score`
- `classification_report` (precision, recall, F1 per species)
- `confusion_matrix` via `ConfusionMatrixDisplay`
- an error bar chart: rows where `real != predicted`, grouped by true species (`value_counts` + seaborn barplot)

### 7. Gradio interface

`prezice_specie` builds a one-row DataFrame from UI widgets, applies the same dummy encoding, **aligns columns** to `X_train_codificat.columns` (missing dummy columns filled with 0), scales the numeric block with the fitted `StandardScaler`, and returns `model.predict(...)[0]`.

Widgets: dropdowns for category, habitat, endangered flag, and color; sliders for weight `[0, 700]`, lifespan `[1, 60]`, speed `[0, 70]`, height `[0, 3]`.

```python
interfata.launch()
```

---

## Part II — Pipelines, regression, and endangered classification

Scripts in `II/` reuse the CSV files. Preprocessing is expressed as sklearn `Pipeline` / `ColumnTransformer` objects rather than ad-hoc pandas transforms.

### 1. Normalization — `1_normalizare.py`

Target for this script is `Pericol de disparitie`. Features are everything else.

Two sub-pipelines are combined:

| Branch | Columns | Steps |
| --- | --- | --- |
| Numeric | weight, lifespan, speed, height | `SimpleImputer(strategy='median')` then `StandardScaler` |
| Categorical | species, category, habitat, color | `SimpleImputer(strategy='most_frequent')` then `OneHotEncoder(handle_unknown='ignore')` |

`preprocessor.fit_transform(x_train)` learns imputers, scaler means/stds, and dummy categories from train only. `preprocessor.transform(x_test)` applies those fitted parameters. Feature names are reconstructed with `OneHotEncoder.get_feature_names_out` so the dense matrix can be printed as a DataFrame.

Median imputation is used instead of the Part I mean, which is more robust to the heavy right tail of camel weights.

### 2. Analysis — `2_analizare.py`

Numeric columns are coerced with `pd.to_numeric(..., errors='coerce')`, then median-imputed with a train-fitted `SimpleImputer` (`fit` on train, `transform` on test).

| Step | Implementation | Output |
| --- | --- | --- |
| (a) Missingness | `isnull().sum()` table + seaborn heatmap of the null mask | `no_data_for_train.png`, `no_data_for_test.png` |
| (b) Descriptive stats | `describe()` for numerics; `value_counts()` per categorical column | stdout |
| (c) Distributions | `sns.histplot(..., kde=True, bins=30)` and `sns.countplot` | `eda_plots/` |
| (d) Outliers | boxplots + Tukey fences `[Q1 - 1.5 IQR, Q3 + 1.5 IQR]` | `outliers/`; outlier rows printed |
| (e) Correlations | `train[num_cols].corr()` heatmap | `corelatii_train.png` |
| (f) Target relations | `Pericol de disparitie` mapped `Nu=0`, `Da=1`; violin plots (numeric vs label) and hue countplots (categorical vs label) | `analiza_relationala/` |

Filenames are sanitized (`spaces`, `(`, `)`, `/` replaced) so column names such as `Viteza (Km/h)` become valid PNG names.

The factor **1.5** on IQR is the conventional Tukey multiplier, chosen because it approximates outlier tails of a Gaussian.

### 3. Training — `3_antrenare.py`

Regression target: `Greutate (Kg)`. Dropped from `X`: weight and endangered status.

Categorical columns are one-hot encoded with `OneHotEncoder(drop='first')` inside a `ColumnTransformer`; `remainder='passthrough'` keeps numeric columns unchanged (no extra scaling in this script).

Each estimator is wrapped:

```python
Pipeline([('preprocessor', preprocessor), ('regressor', model)])
```

Models:

- `LinearRegression`
- `RandomForestRegressor` (sklearn defaults)
- `SVR` (RBF kernel by default)

Fit on train, predict on test. Metrics: **R²** (`r2_score`) and **RMSE** (`sqrt(mean_squared_error)`).

### 4. Evaluation — `4_evaluare.py`

Same regression setup as above, plus MAE. Results are collected into a comparative `DataFrame` (RMSE, MAE, R²).

Then a second task: binary endangered classification. Labels are mapped `{'Nu': 0, 'Da': 1}`. Features still exclude weight and the raw target. Same categorical `ColumnTransformer`.

Classifiers:

- `LogisticRegression(max_iter=1024)`
- `DecisionTreeClassifier`
- `RandomForestClassifier`

Metrics: **accuracy**, **F1**, **ROC AUC**. AUC uses `predict_proba(x_test)[:, 1]` (positive-class probability). Another comparison table is printed.

---

## Design differences between the two parts

| Aspect | Part I | Part II |
| --- | --- | --- |
| Primary target | `Nume Specie` (9-way) | Weight (regression) and `Pericol de disparitie` (binary) |
| Imputation | Column mean | Median (numeric), mode (categorical) |
| Encoding | `pd.get_dummies(drop_first=True)` | `OneHotEncoder` in `ColumnTransformer` |
| Scaling | `StandardScaler` on numeric features | Used in the normalization pipeline; omitted in the training/eval scripts |
| Modeling API | Fit estimator directly | `Pipeline` so preprocess + model share one `fit`/`predict` |
| Serving | Gradio | Offline metric tables |

---

## How to run

Python 3 with `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`. Part I also needs `gradio`.

```bash
# Part I (from I/, or open the notebook)
python tema1_pclp3_darievladandrei_311cc.py

# Part II (from II/, with train.csv and test.csv present)
python 1_normalizare.py
python 2_analizare.py
python 3_antrenare.py
python 4_evaluare.py
```

---

## Authors

- **Vlad-Andrei Darie** (311CC) — Part I: dataset generation, EDA, species classifier, Gradio app.
- **Theodor Ioan Pirnog** (311CC) — Part II: preprocessing pipelines, extended EDA, regression models, comparative evaluation.
