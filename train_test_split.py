import pandas as pd
from sklearn.model_selection import train_test_split

from project_paths import PROCESSED_DATA_DIR

# ─────────────────────────────────────────────
# SAMPLE DATA  (replace with your own dataset)
# ─────────────────────────────────────────────
data = {
    "text": [
        "I love this product, it's amazing!",
        "Terrible experience, very disappointed.",
        "It was okay, nothing special.",
        "Absolutely fantastic, will buy again!",
        "Not worth the money at all.",
        "Pretty decent for the price.",
        "Outstanding quality and fast delivery.",
        "Worst purchase I've ever made.",
        "Neutral opinion, works as expected.",
        "Highly recommend to everyone!",
        "Very bad customer service.",
        "Great product, happy with it.",
        "Average quality, could be better.",
        "Exceeded my expectations completely.",
        "Disappointed with the build quality.",
    ],
    "label": [
        "positive", "negative", "neutral",
        "positive", "negative", "neutral",
        "positive", "negative", "neutral",
        "positive", "negative", "positive",
        "neutral",  "positive", "negative",
    ]
}

df = pd.DataFrame(data)

# ─────────────────────────────────────────────
# EXPLORE DATA
# ─────────────────────────────────────────────
print("=" * 45)
print("         DATASET OVERVIEW")
print("=" * 45)
print(f"Total samples     : {len(df)}")
print(f"Columns           : {list(df.columns)}")
print(f"\nClass Distribution:")
print(df["label"].value_counts().to_string())
print(f"\nMissing Values    : {df.isnull().sum().sum()}")

# ─────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 80% train | 20% test
    random_state=42,     # reproducibility
    stratify=y           # preserve class balance
)

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
print("\n" + "=" * 45)
print("           SPLIT RESULTS")
print("=" * 45)
print(f"Training samples  : {len(X_train)}  ({len(X_train)/len(X)*100:.1f}%)")
print(f"Testing  samples  : {len(X_test)}   ({len(X_test)/len(X)*100:.1f}%)")

print(f"\nTrain Class Distribution:")
print(y_train.value_counts().to_string())

print(f"\nTest Class Distribution:")
print(y_test.value_counts().to_string())

# ─────────────────────────────────────────────
# SAVE SPLITS
# ─────────────────────────────────────────────
train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
test_df  = pd.concat([X_test,  y_test],  axis=1).reset_index(drop=True)

train_output = PROCESSED_DATA_DIR / "train.csv"
test_output = PROCESSED_DATA_DIR / "test.csv"

train_df.to_csv(train_output, index=False)
test_df.to_csv(test_output, index=False)

print("\n" + "=" * 45)
print(f"Files saved: {train_output}  |  {test_output}")
print("=" * 45)

# ─────────────────────────────────────────────
# PREVIEW
# ─────────────────────────────────────────────
print("\n--- Training Set Preview ---")
print(train_df.head())

print("\n--- Testing Set Preview ---")
print(test_df.head())
