import os
import pandas as pd

# Sti til mappen med bilder
folder = "data/raw/crop_part1"

data = []

# Gå gjennom alle filer
for file in os.listdir(folder):
    if file.endswith(".jpg"):
        try:
            parts = file.split("_")
            age = int(parts[0])
            gender = int(parts[1])
            race = int(parts[2])
            data.append([file, age, gender, race])
        except:
            print("Feil med filnavn:", file)

# Lag DataFrame
df = pd.DataFrame(data, columns=["filename", "age", "gender", "race"])

# Sørg for at processed-mappen finnes
os.makedirs("data/processed", exist_ok=True)

# Lagre CSV-filen
df.to_csv("data/processed/labels.csv", index=False)

print(f"✅ Lagret {len(df)} rader til data/processed/labels.csv")
