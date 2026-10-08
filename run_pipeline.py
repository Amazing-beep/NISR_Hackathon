import os
import sys
import glob
import zipfile
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter

# 1. Configuration & Reproducibility
SEED = 42
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 1  # Mandatory placeholder pass for formative evaluation
LOG_DIR = "runs/eicv7_poverty_mlp"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# 2. Dynamic Data Loading & Auto-Unzip Resolution (Google Colab & Local Support)
NEEDED_FILES = {
    'pov': 'CS_EICV7_poverty_file.dta',
    'hh': 'CS_S01_S5_S7_Household.dta',
    'person': 'CS_S0_S1_S2_S3_S4_S6A_S6B_S6C_Person.dta'
}

def locate_and_prepare_dataset():
    # Step 1: Auto-extract any zip files found in workspace or Colab environment
    zip_search_patterns = ['*.zip', 'Cross_Section/*.zip']
    if 'google.colab' in sys.modules:
        zip_search_patterns.extend(['/content/*.zip', '/content/drive/MyDrive/*.zip'])
    
    for pattern in zip_search_patterns:
        for z_path in glob.glob(pattern):
            try:
                print(f"Auto-extracting zip archive: {z_path}...")
                with zipfile.ZipFile(z_path, 'r') as z_ref:
                    z_ref.extractall('.')
            except Exception as e:
                print(f"Zip extraction notice ({z_path}): {e}")

    # Step 2: Search recursively for the required .dta files
    search_roots = ['.', 'Cross_Section']
    if 'google.colab' in sys.modules:
        search_roots.append('/content')

    found_paths = {}
    for key, fname in NEEDED_FILES.items():
        located = None
        for s_root in search_roots:
            matches = glob.glob(os.path.join(s_root, '**', fname), recursive=True)
            if matches:
                located = matches[0]
                break
        found_paths[key] = located

    # Step 3: If running on Colab and files are missing, offer interactive file upload
    if any(v is None for v in found_paths.values()) and 'google.colab' in sys.modules:
        print("\n[COLAB NOTICE] EICV7 microdata files not found locally in Colab.")
        print("Opening interactive file uploader... Please upload Microdata.zip or the 3 .dta files.")
        try:
            from google.colab import files
            uploaded = files.upload()
            for fn in uploaded.keys():
                if fn.endswith('.zip'):
                    with zipfile.ZipFile(fn, 'r') as z_ref:
                        z_ref.extractall('.')
            # Re-search after upload
            for key, fname in NEEDED_FILES.items():
                matches = glob.glob(os.path.join('.', '**', fname), recursive=True) + \
                          glob.glob(os.path.join('/content', '**', fname), recursive=True)
                if matches:
                    found_paths[key] = matches[0]
        except Exception as upload_err:
            print(f"Colab upload notice: {upload_err}")

    # Final Check
    missing = [k for k, v in found_paths.items() if v is None]
    if missing:
        missing_names = [NEEDED_FILES[m] for m in missing]
        raise FileNotFoundError(
            f"Missing required dataset file(s): {missing_names}.\n"
            "Please ensure CS_EICV7_poverty_file.dta, CS_S01_S5_S7_Household.dta, and CS_S0_S1_S2_S3_S4_S6A_S6B_S6C_Person.dta "
            "are placed in the working directory or uploaded as a zip file."
        )

    return found_paths

dataset_paths = locate_and_prepare_dataset()

pov_df = pd.read_stata(dataset_paths['pov'])
hh_df = pd.read_stata(dataset_paths['hh'])
person_df = pd.read_stata(dataset_paths['person'])

# Extract household head demographics
head_df = person_df[person_df['s1q2'] == 'Household head (HH)'].copy()
head_vars = head_df[['hhid', 's1q1', 's1q3y', 's1q4', 's6bq5']].rename(columns={
    's1q1': 'head_sex',
    's1q3y': 'head_age',
    's1q4': 'head_marital',
    's6bq5': 'head_sector'
}).drop_duplicates(subset=['hhid'])

# Select non-overlapping columns from household file
hh_cols = [c for c in hh_df.columns if c not in pov_df.columns or c == 'hhid']

merged = pov_df.merge(hh_df[hh_cols], on='hhid', how='inner').merge(head_vars, on='hhid', how='inner')

# Target Construction
merged['target'] = (merged['pov_jan'] == 'Poor').astype(np.float32)

# Exclude Consumption/Expenditure Variables to Prevent Data Leakage
leakage_cols = [
    'exp1', 'exp2', 'exp3', 'exp4', 'exp5', 'exp6', 'exp7', 'exp8', 'exp9', 'exp10', 'exp11',
    'cons1', 'cons1ae', 'food', 'sol_jan', 'quintile', 'Poverty_line', 'Extreme_line',
    'poverty', 'epov_jan', 's5bq1', 's5bq2a', 's5bq3a', 's5bq5a', 's5bq8', 's5cq9a', 's5cq9b',
    's5cq11', 's5cq20'
]

# Living Condition Feature Selection
cat_features = [
    'district', 'province', 'ur', 's5aq1', 's5aq2', 's5cq1', 's5cq16',
    's5cq22a', 's5cq25', 's5dq1', 's5dq2', 's5dq3', 'head_sex', 'head_marital', 'head_sector'
]

num_features = ['member', 's5aq3', 'head_age']

# Explicitly cast numerical features to float to handle Stata categories
for col in num_features:
    merged[col] = pd.to_numeric(merged[col], errors='coerce')

# Clean categorical features to pure strings
def clean_str_array(series):
    return np.array([
        str(x) if (x is not None and not pd.isna(x) and str(x) != 'nan') else 'Unknown'
        for x in series.values
    ])

# Verify no leakage features exist in X
X = merged[cat_features + num_features].copy()
y = merged['target'].values

# Train / Validation / Test Split (70 / 15 / 15)
X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)
X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp_raw, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
)

# Preprocessing Fitted ONLY on Training Data
num_imputers = {}
for col in num_features:
    med_val = X_train_raw[col].median()
    if pd.isna(med_val):
        med_val = 0.0
    num_imputers[col] = med_val

X_train_num = X_train_raw[num_features].fillna(num_imputers).values
X_val_num = X_val_raw[num_features].fillna(num_imputers).values
X_test_num = X_test_raw[num_features].fillna(num_imputers).values

scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num)
X_val_num_scaled = scaler.transform(X_val_num)
X_test_num_scaled = scaler.transform(X_test_num)

# Categorical Feature Mapping
cat_vocab = {}
emb_dims = []

X_train_cat_idx = np.zeros((len(X_train_raw), len(cat_features)), dtype=np.int64)
X_val_cat_idx = np.zeros((len(X_val_raw), len(cat_features)), dtype=np.int64)
X_test_cat_idx = np.zeros((len(X_test_raw), len(cat_features)), dtype=np.int64)

for i, col in enumerate(cat_features):
    train_vals = clean_str_array(X_train_raw[col])
    unique_cats = np.unique(train_vals)
    vocab = {val: idx + 1 for idx, val in enumerate(unique_cats)}  # 0 is reserved for Unknown/Unseen
    vocab['<unknown>'] = 0
    cat_vocab[col] = vocab

    cardinality = len(vocab)
    emb_dim = min(50, max(4, cardinality // 2))
    emb_dims.append((cardinality, emb_dim))

    # Map train, val, test
    X_train_cat_idx[:, i] = [vocab.get(val, 0) for val in train_vals]
    val_vals = clean_str_array(X_val_raw[col])
    X_val_cat_idx[:, i] = [vocab.get(val, 0) for val in val_vals]
    test_vals = clean_str_array(X_test_raw[col])
    X_test_cat_idx[:, i] = [vocab.get(val, 0) for val in test_vals]

# PyTorch Dataset Definition
class EICVDataset(Dataset):
    def __init__(self, cat_idx, num_scaled, targets):
        self.cat_idx = torch.tensor(cat_idx, dtype=torch.long)
        self.num_scaled = torch.tensor(num_scaled, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.cat_idx[idx], self.num_scaled[idx], self.targets[idx]

train_dataset = EICVDataset(X_train_cat_idx, X_train_num_scaled, y_train)
val_dataset = EICVDataset(X_val_cat_idx, X_val_num_scaled, y_val)
test_dataset = EICVDataset(X_test_cat_idx, X_test_num_scaled, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# PyTorch Embedding MLP Architecture
class EICVEmbeddingMLP(nn.Module):
    def __init__(self, emb_dims, num_features_dim):
        super(EICVEmbeddingMLP, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=cardinality, embedding_dim=emb_dim)
            for cardinality, emb_dim in emb_dims
        ])
        total_emb_dim = sum(emb_dim for _, emb_dim in emb_dims)
        total_input_dim = total_emb_dim + num_features_dim

        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, cat_idx, num_feats):
        emb_outs = [emb_layer(cat_idx[:, i]) for i, emb_layer in enumerate(self.embeddings)]
        cat_concat = torch.cat(emb_outs, dim=1)
        combined = torch.cat([cat_concat, num_feats], dim=1)
        logits = self.mlp(combined)
        return logits.squeeze(-1)

model = EICVEmbeddingMLP(emb_dims, len(num_features))

# Loss & Class Weighting Calculation
num_pos = (y_train == 1.0).sum()
num_neg = (y_train == 0.0).sum()
pos_weight_val = num_neg / num_pos
pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Training & Evaluation Functions
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for cat_b, num_b, target_b in dataloader:
        optimizer.zero_grad()
        logits = model(cat_b, num_b)
        loss = criterion(logits, target_b)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(target_b)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(target_b.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    binary_preds = (np.array(all_preds) >= 0.5).astype(int)
    epoch_acc = accuracy_score(all_targets, binary_preds)
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_probs, all_targets = [], []

    with torch.no_grad():
        for cat_b, num_b, target_b in dataloader:
            logits = model(cat_b, num_b)
            loss = criterion(logits, target_b)
            running_loss += loss.item() * len(target_b)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(target_b.cpu().numpy())

    total_loss = running_loss / len(dataloader.dataset)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    binary_preds = (all_probs >= 0.5).astype(int)

    acc = accuracy_score(all_targets, binary_preds)
    prec = precision_score(all_targets, binary_preds, zero_division=0)
    rec = recall_score(all_targets, binary_preds, zero_division=0)
    f1 = f1_score(all_targets, binary_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        auc = 0.5

    return {
        'loss': total_loss,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': auc
    }

# Execute One-Epoch Placeholder Run
train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
val_metrics = evaluate(model, val_loader, criterion)
test_metrics = evaluate(model, test_loader, criterion)

print("--- ONE-EPOCH PLACEHOLDER TRAINING COMPLETE ---")
print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val Prec: {val_metrics['precision']:.4f} | Val Rec: {val_metrics['recall']:.4f} | Val F1: {val_metrics['f1']:.4f} | Val AUC: {val_metrics['roc_auc']:.4f}")
print(f"Test Loss: {test_metrics['loss']:.4f} | Test Acc: {test_metrics['accuracy']:.4f} | Test Prec: {test_metrics['precision']:.4f} | Test Rec: {test_metrics['recall']:.4f} | Test F1: {test_metrics['f1']:.4f} | Test AUC: {test_metrics['roc_auc']:.4f}")

# TensorBoard Logging
writer = SummaryWriter(log_dir=LOG_DIR)

# Log Scalars
writer.add_scalar("Train/Loss", train_loss, 1)
writer.add_scalar("Train/Accuracy", train_acc, 1)
for metric_name, metric_val in val_metrics.items():
    writer.add_scalar(f"Val/{metric_name.capitalize()}", metric_val, 1)
for metric_name, metric_val in test_metrics.items():
    writer.add_scalar(f"Test/{metric_name.capitalize()}", metric_val, 1)

# Log Model Graph
dummy_cat, dummy_num, _ = next(iter(train_loader))
writer.add_graph(model, (dummy_cat, dummy_num))

# Log Hyperparameters & Final Metrics
hparams = {
    'learning_rate': LEARNING_RATE,
    'batch_size': BATCH_SIZE,
    'weight_decay': WEIGHT_DECAY,
    'optimizer': 'Adam',
    'architecture': 'Embedding_MLP_128_64',
    'dropout': 0.3,
    'epochs': EPOCHS
}

final_metrics = {
    'hparam/train_loss': train_loss,
    'hparam/train_accuracy': train_acc,
    'hparam/val_loss': val_metrics['loss'],
    'hparam/val_accuracy': val_metrics['accuracy'],
    'hparam/val_f1': val_metrics['f1'],
    'hparam/val_auc': val_metrics['roc_auc']
}

writer.add_hparams(hparams, final_metrics)
writer.close()

print(f"TensorBoard logs successfully written to directory: {LOG_DIR}")
