import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Les labels.csv
df = pd.read_csv("data/processed/labels.csv")

# Først: del ut test 10%
train_val, test = train_test_split(df, test_size=0.1, random_state=42)

# Så: del train_val i train og val (10% av train_val går til val)
train, val = train_test_split(train_val, test_size=0.1111, random_state=42)  
# 0.1111 ≈ 10% av original data

# Lagre filene
os.makedirs("data/splits", exist_ok=True)
train.to_csv("data/splits/train.csv", index=False)
val.to_csv("data/splits/val.csv", index=False)
test.to_csv("data/splits/test.csv", index=False)

print(f"✅ Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
