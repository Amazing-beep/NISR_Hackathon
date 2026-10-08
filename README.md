# NISR 2026 Big Data Hackathon — Track 2 Formative
## Project Title: Predicting Household Poverty from Living-Conditions Data in Rwanda

**Course**: Machine Learning Pipeline  
**Track**: Track 2 — Financial Inclusion & Poverty Reduction  
**Dataset**: Integrated Household Living Conditions Survey (EICV7), 2023–2024, National Institute of Statistics of Rwanda (NISR)  
**Institution**: African Leadership University (ALU)

---

## 📌 Project Overview

This repository contains the end-to-end Machine Learning Pipeline implementation for Track 2 of the **NISR 2026 Big Data Hackathon**.

The objective is to build a feed-forward neural network (MLP) with learned categorical entity embeddings that predicts household poverty status (`1 = Poor`, `0 = Non-Poor`) in Rwanda using **living-condition proxy features** (e.g., dwelling construction materials, primary water source, cooking fuel, toilet facilities, sanitation, and household head demographics) **without relying on monetary consumption or expenditure data**.

---

## 🔬 Methodology & Architecture

1. **Dataset**: NISR EICV7 Household Microdata (15,054 completed household survey records).
2. **Data Leakage Prevention**: Explicitly excludes all 29 consumption, expenditure, income, quintile, and direct poverty line metrics.
3. **Statistical Split**: 70% Train (10,537), 15% Validation (2,258), 15% Test (2,259) stratified split (`seed=42`). Preprocessing parameters (scaling, imputations, vocabularies) are fitted **strictly on training data**.
4. **Model Architecture**: PyTorch `EICVEmbeddingMLP`:
   - Dedicated `nn.Embedding` layers for 15 living-condition categorical features.
   - Concatenation of categorical embeddings + 3 standardized numerical features.
   - Dense Layers: `Linear(input_dim, 128) -> BatchNorm1d -> ReLU -> Dropout(0.3) -> Linear(128, 64) -> BatchNorm1d -> ReLU -> Dropout(0.3) -> Linear(64, 1)`.
5. **Class Imbalance Handling**: Dynamic positive-class loss weighting (`pos_weight = 3.203`) calculated from training targets passed to `nn.BCEWithLogitsLoss`.
6. **Placeholder Run**: 1-Epoch formative pipeline verification pass logged to TensorBoard (`runs/eicv7_poverty_mlp`).

---

## 📂 Repository Structure

```text
├── NISR_Hackathon.ipynb     # Main Jupyter Notebook (17 structured sections with 75-150 word comments)
├── run_pipeline.py          # Standalone reproducible Python execution script
├── create_notebook.py       # Programmatic notebook generator script
├── runs/                    # TensorBoard event log files (scalars, graph, hparams)
├── .gitignore               # Git ignore rules for data and virtual environment
└── README.md                # Project documentation
```

---

## 🚀 How to Run

### Option 1: On Google Colab (Recommended)
1. Open [`NISR_Hackathon.ipynb`](NISR_Hackathon.ipynb) in Google Colab.
2. Upload your `Microdata.zip` (containing `CS_EICV7_poverty_file.dta`, `CS_S01_S5_S7_Household.dta`, and `CS_S0_S1_S2_S3_S4_S6A_S6B_S6C_Person.dta`).
3. Click **Runtime -> Run all** (`Ctrl + F9`). The notebook automatically extracts the zip archive and executes from Section 1 through Section 17 without errors.

### Option 2: Locally on Windows / Linux / macOS
1. Clone this repository:
   ```bash
   git clone https://github.com/Amazing-beep/NISR_Hackathon.git
   cd NISR_Hackathon
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1
   pip install torch tensorboard scikit-learn pandas numpy pyreadstat
   ```
3. Run the pipeline script:
   ```bash
   python run_pipeline.py
   ```
4. View TensorBoard metrics:
   ```bash
   tensorboard --logdir runs
   ```

---

## 📊 Dataset Access Notice
The microdata files used in this project are published by the National Institute of Statistics of Rwanda (NISR) and can be requested via the official Central Data Catalog: [https://microdata.statistics.gov.rw](https://microdata.statistics.gov.rw).

---

## 📜 License & Compliance
Developed for academic formative evaluation under the NISR 2026 Big Data Hackathon & ALU Machine Learning Pipeline Course.
