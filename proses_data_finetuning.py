import pandas as pd

# File paths
file1_path = "Dataset/exercise_dataset.csv"
file2_path = "Dataset/Top 50 Excerice for your body.csv"
file3_path = "Dataset/megaGymDataset.csv"

# Load datasets
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)
df3 = pd.read_csv(file3_path)

# Proses dataset menjadi prompt-response
def process_datasets():
    rows = []

    # Dataset 1: exercise_dataset.csv
    if "Activity, Exercise or Sport (1 hour)" in df1.columns:
        for _, row in df1.iterrows():
            prompt = f"Rancang olahraga untuk aktivitas {row['Activity, Exercise or Sport (1 hour)']}."
            response = f"{row['Activity, Exercise or Sport (1 hour)']} selama 30 menit, intensitas sedang."
            rows.append({"prompt": prompt, "response": response})

    # Dataset 2: Top 50 Exercise for your body.csv
    if "Name of Exercise" in df2.columns:
        for _, row in df2.iterrows():
            prompt = (
                f"Jelaskan manfaat dan cara melakukan {row['Name of Exercise']} "
                f"untuk target otot {row['Target Muscle Group']}."
            )
            response = (
                f"{row['Name of Exercise']} memiliki manfaat {row['Benefit']} dan membakar {row['Burns Calories (per 30 min)']} kalori."
                f" Latihan ini termasuk {row['Difficulty Level']}."
            )
            rows.append({"prompt": prompt, "response": response})

    # Dataset 3: megaGymDataset.csv
    if all(col in df3.columns for col in ['Title', 'Desc', 'Type', 'BodyPart', 'Equipment', 'Level']):
        for _, row in df3.iterrows():
            prompt = f"Jelaskan latihan {row['Title']} yang menggunakan {row['Equipment']}."
            response = (
                f"{row['Title']} adalah latihan {row['Type']} untuk {row['BodyPart']}. "
                f"Deskripsi: {row['Desc']}. Level: {row['Level']}."
            )
            rows.append({"prompt": prompt, "response": response})

    return pd.DataFrame(rows)

# Save processed dataset
processed_df = process_datasets()
output_path = "Dataset/finetune_dataset.csv"
processed_df.to_csv(output_path, index=False)
print(f"Processed dataset saved to {output_path}")
