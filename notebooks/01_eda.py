from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# ======================
# CONFIG
# ======================

DATA_PATH = Path("data/processed/clean_data.csv")
FIGURES_PATH = Path("reports/figures")
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

# ======================
# LOAD DATA
# ======================

print("🚀 Starting EDA...")

df = pd.read_csv(DATA_PATH)

print("\n===== SHAPE =====")
print(df.shape)

print("\n===== INFO =====")
df.info()

# ======================
# TARGET DISTRIBUTION
# ======================

print("\n===== TARGET DISTRIBUTION =====")
print(df["risk"].value_counts(normalize=True))

plt.figure()
sns.countplot(data=df, x="risk")
plt.title("Target Distribution")
plt.savefig(FIGURES_PATH / "target_distribution.png")
plt.close()

# ======================
# NUMERIC ANALYSIS
# ======================

num_cols = df.select_dtypes(include="number").columns

print("\n===== NUMERIC SUMMARY =====")
print(df[num_cols].describe())

print("\n===== SKEWNESS =====")
print(df[num_cols].skew())

# Histograms
df[num_cols].hist(figsize=(12, 8))
plt.tight_layout()
plt.savefig(FIGURES_PATH / "histograms.png")
plt.close()

# Boxplots
for col in num_cols:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(col)
    plt.savefig(FIGURES_PATH / f"boxplot_{col}.png")
    plt.close()

# ======================
# CORRELATION
# ======================

plt.figure(figsize=(8, 6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.savefig(FIGURES_PATH / "correlation.png")
plt.close()

# ======================
# CATEGORICAL ANALYSIS
# ======================

cat_cols = df.select_dtypes(include="object").columns

for col in cat_cols:
    print(f"\n--- {col} ---")
    print(df[col].value_counts(normalize=True))

    plt.figure()
    sns.countplot(data=df, x=col)
    plt.xticks(rotation=45)
    plt.title(col)
    plt.savefig(FIGURES_PATH / f"countplot_{col}.png")
    plt.close()

# ======================
# TARGET RELATIONSHIPS
# ======================

# Numeric vs target
for col in num_cols:
    plt.figure()
    sns.boxplot(data=df, x="risk", y=col)
    plt.title(f"{col} vs Risk")
    plt.savefig(FIGURES_PATH / f"{col}_vs_risk.png")
    plt.close()

# Categorical vs target
for col in cat_cols:
    if col == "risk":
        continue

    cross = pd.crosstab(df[col], df["risk"], normalize="index")
    print(f"\n{col} vs Risk:")
    print(cross)

print("\n✅ EDA Completed")
print(f"📊 Figures saved in: {FIGURES_PATH}")
