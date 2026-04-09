# =============================================================================
#  NOTEBOOK 2 · BERT Severity Classifier
#  Run this after Notebook 1 (or independently) in Google Colab.
#
#  Input  : Train_TEST.csv
#  Outputs: bert_severity_model/   — saved HuggingFace model directory
#           bert_label_map.pkl     — id2label / label2id mappings
#
#  Requirements (install once):
#      pip install transformers datasets torch scikit-learn scipy -q
# =============================================================================

# ── SECTION 0 · Install dependencies ─────────────────────────────────────────
# Uncomment the line below when running for the first time
# !pip install transformers datasets torch scikit-learn scipy -q

# ── SECTION 1 · Imports ──────────────────────────────────────────────────────
import os
import pickle
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from scipy.special import softmax

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)

from datasets import Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

print("✓ All libraries loaded")
print(f"  PyTorch  : {torch.__version__}")
print(f"  CUDA     : {'available ✓' if torch.cuda.is_available() else 'not available — using CPU'}")

# ── SECTION 2 · Configuration ─────────────────────────────────────────────────
TRAIN_CSV      = "Train_TEST.csv"       # Input: raw complaint data
MODEL_DIR      = "bert_severity_model"  # Output: saved HuggingFace model
LABEL_MAP_PKL  = "bert_label_map.pkl"  # Output: label encoding maps

BERT_CHECKPOINT = "bert-base-uncased"
MAX_LENGTH      = 128
LEARNING_RATE   = 2e-5
BATCH_SIZE      = 16
NUM_EPOCHS      = 3
WEIGHT_DECAY    = 0.01
TEST_SPLIT      = 0.2
RANDOM_STATE    = 42

print("\n✓ Configuration loaded")
print(f"  Model checkpoint : {BERT_CHECKPOINT}")
print(f"  Max token length : {MAX_LENGTH}")
print(f"  Epochs           : {NUM_EPOCHS}")
print(f"  Batch size       : {BATCH_SIZE}")

# ── SECTION 3 · Load and split data ──────────────────────────────────────────
print("\n--- Section 3: Loading data ---")

df = pd.read_csv(TRAIN_CSV)
df = df.dropna(subset=["grievance_text", "severity"])

X = df.drop(columns=["severity"])
y = df["severity"]

# Stratified 80/20 split — preserves class proportions in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SPLIT,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"Dataset shape : {df.shape}")
print(f"Train size    : {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")
print(f"Class distribution (train):")
print(y_train.value_counts().to_string())

# ── SECTION 4 · Label encoding ───────────────────────────────────────────────
print("\n--- Section 4: Encoding labels ---")

# Sort labels alphabetically for reproducibility
unique_labels = sorted(y_train.unique().tolist())
label2id      = {label: i for i, label in enumerate(unique_labels)}
id2label      = {i: label for label, i in label2id.items()}
num_labels    = len(unique_labels)

print(f"Labels ({num_labels}): {label2id}")

y_train_enc = y_train.map(label2id)
y_test_enc  = y_test.map(label2id)

# Persist label maps for inference
with open(LABEL_MAP_PKL, "wb") as fh:
    pickle.dump({"label2id": label2id, "id2label": id2label}, fh)
print(f"✓ Label maps saved → {LABEL_MAP_PKL}")

# ── SECTION 5 · Tokenisation ──────────────────────────────────────────────────
print("\n--- Section 5: Tokenising with BertTokenizer ---")

tokenizer = BertTokenizer.from_pretrained(BERT_CHECKPOINT)

def tokenize(example):
    """Tokenise a single text example with padding and truncation."""
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )

# ── SECTION 6 · Build HuggingFace datasets ───────────────────────────────────
print("\n--- Section 6: Building HuggingFace Datasets ---")

train_df = pd.DataFrame({
    "text":  X_train["grievance_text"].values,
    "label": y_train_enc.values,
})
test_df = pd.DataFrame({
    "text":  X_test["grievance_text"].values,
    "label": y_test_enc.values,
})

train_dataset = Dataset.from_pandas(train_df)
test_dataset  = Dataset.from_pandas(test_df)

# Tokenise both splits
train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset  = test_dataset.map(tokenize, batched=True)

# HuggingFace Trainer expects the label column to be named "labels"
train_dataset = train_dataset.rename_column("label", "labels")
test_dataset  = test_dataset.rename_column("label", "labels")

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format( type="torch", columns=["input_ids", "attention_mask", "labels"])

print(f"✓ Train dataset : {len(train_dataset)} examples")
print(f"✓ Test  dataset : {len(test_dataset)} examples")

# ── SECTION 7 · Model initialisation ─────────────────────────────────────────
print("\n--- Section 7: Initialising BERT for Sequence Classification ---")

model = BertForSequenceClassification.from_pretrained(
    BERT_CHECKPOINT,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)
print(f"✓ Model loaded: {BERT_CHECKPOINT}  |  Num labels: {num_labels}")

# ── SECTION 8 · Training ──────────────────────────────────────────────────────
print("\n--- Section 8: Fine-tuning BERT ---")

training_args = TrainingArguments(
    output_dir          = "./bert_results",
    learning_rate       = LEARNING_RATE,
    per_device_train_batch_size = BATCH_SIZE,
    num_train_epochs    = NUM_EPOCHS,
    weight_decay        = WEIGHT_DECAY,
    logging_dir         = "./bert_logs",
    logging_steps       = 50,
    save_strategy       = "epoch",
    evaluation_strategy = "no",          # Full eval done manually below
    use_cpu             = not torch.cuda.is_available(),
    report_to           = "none",        # Disable W&B / MLflow logging
)

trainer = Trainer(
    model         = model,
    args          = training_args,
    train_dataset = train_dataset,
)

trainer.train()
print("✓ Training complete")

# ── SECTION 9 · Prediction and Evaluation ────────────────────────────────────
print("\n--- Section 9: Evaluating on test set ---")

# Batch predictions on the full test set
predictions_output = trainer.predict(test_dataset)

# predictions_output.predictions contains raw logits (shape: N × num_labels)
logits          = predictions_output.predictions
predicted_labels = np.argmax(logits, axis=1)
true_labels     = predictions_output.label_ids

# Convert predicted integer indices back to string labels for the report
true_labels_str      = [id2label[i] for i in true_labels]
predicted_labels_str = [id2label[i] for i in predicted_labels]

# ── SECTION 10 · Metrics ──────────────────────────────────────────────────────
print("\n===== BERT Evaluation Metrics =====")

accuracy  = accuracy_score(true_labels, predicted_labels)
precision = precision_score(
    true_labels, predicted_labels, average="weighted", zero_division=0
)
f1        = f1_score(true_labels, predicted_labels, average="weighted")

# AUC — one-vs-rest, using softmax probabilities over raw logits
probabilities = softmax(logits, axis=1)
auc = roc_auc_score(
    true_labels, probabilities,
    multi_class="ovr", average="weighted",
)

print(f"  Accuracy  : {accuracy:.4f}")
print(f"  Precision : {precision:.4f}  (weighted)")
print(f"  F1-Score  : {f1:.4f}       (weighted)")
print(f"  AUC       : {auc:.4f}       (OvR, weighted)")
print("\n===== Full Classification Report =====")
print(classification_report(
    true_labels_str, predicted_labels_str, target_names=unique_labels
))

# ── SECTION 11 · Confusion Matrix ────────────────────────────────────────────
print("\n--- Section 11: Confusion Matrix ---")

cm = confusion_matrix(true_labels, predicted_labels)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=unique_labels,
    yticklabels=unique_labels,
    ax=ax,
)
ax.set_title("BERT — Confusion Matrix (Test Set)", fontsize=13, pad=12)
ax.set_xlabel("Predicted Severity")
ax.set_ylabel("True Severity")
plt.tight_layout()
plt.savefig("bert_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ Confusion matrix saved → bert_confusion_matrix.png")

# ── SECTION 12 · Model comparison summary ────────────────────────────────────
print("\n===== NLP Model Comparison — Full Pipeline =====")
print(f"{'Model':<12} {'Precision':>10} {'Recall':>8} {'F1-Score':>10} {'AUC':>8}")
print("-" * 52)
print(f"{'BiRNN':<12} {'0.27':>10} {'0.44':>8} {'0.32':>10} {'—':>8}   (from Notebook 1)")
print(f"{'BiLSTM':<12} {'0.33':>10} {'0.44':>8} {'0.37':>10} {'—':>8}   (from Notebook 1)")
print(f"{'BERT':<12} {precision:>10.2f} {'(see report)':>8} {f1:>10.2f} {auc:>8.2f}   ← selected")

# ── SECTION 13 · Save model ───────────────────────────────────────────────────
print(f"\n--- Section 13: Saving model to {MODEL_DIR}/ ---")

trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print(f"✓ Model and tokenizer saved → {MODEL_DIR}/")

# ── SECTION 14 · Inference helper ────────────────────────────────────────────
print("\n--- Section 14: Inference Helper ---")

def predict_severity_bert(text: str) -> dict:
    """
    Return the predicted severity label and class probabilities
    for a single complaint text string.

    Returns
    -------
    dict with keys: 'label' (str), 'probabilities' (dict label→float)
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs    = torch.softmax(outputs.logits, dim=1).squeeze().numpy()
    pred_idx = int(np.argmax(probs))
    return {
        "label":         id2label[pred_idx],
        "probabilities": {id2label[i]: round(float(p), 4) for i, p in enumerate(probs)},
    }

# Demo predictions
demo_texts = [
    "internet completely down cannot connect at all since last night emergency",
    "wifi a bit slow today but still working fine",
    "router sometimes disconnects but reconnects within a minute",
]
print("\nExample Predictions:")
for text in demo_texts:
    result = predict_severity_bert(text)
    print(f"  Input  : {text[:65]}...")
    print(f"  Pred   : {result['label']}")
    print(f"  Probs  : {result['probabilities']}\n")

# ── SECTION 15 · Final summary ────────────────────────────────────────────────
print("=" * 60)
print("NOTEBOOK 2 COMPLETE — Outputs")
print("=" * 60)
print(f"  ✓ Fine-tuned model    → {MODEL_DIR}/")
print(f"  ✓ Label maps          → {LABEL_MAP_PKL}")
print(f"  ✓ Confusion matrix    → bert_confusion_matrix.png")
print(f"\n  Accuracy  : {accuracy:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print(f"  AUC       : {auc:.4f}")
print("\nNext step: Run notebook3_dashboard.py with FINAL_forecast_results.xlsx")
