import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, GPT2Config
import torch
import os
from dotenv import load_dotenv

# Memuat variabel dari file .env
load_dotenv()
dataset_path = os.getenv('DATASET_PATH')
# Memuat dataset
data = pd.read_csv(dataset_path)

# Menampilkan beberapa baris dari dataset
print(data.head())

# Gabungkan kolom yang relevan untuk menghasilkan teks
data['text'] = data['Title'] + ": " + data['Desc'] + " (" + data['Type'] + ") - " + data['BodyPart'] + " using " + data['Equipment'] + " - Level: " + data['Level'] + " Rating: " + data['RatingDesc']

# Mengganti nilai NaN dengan string kosong dan memastikan semua nilai adalah string
data['text'] = data['text'].fillna('').astype(str)

# Membagi data menjadi input (texts)
texts = data['text'].values

# Load pre-trained GPT-2 model dan tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Fungsi untuk menghasilkan deskripsi latihan baru menggunakan GPT-2
def generate_exercise_description_gpt(seed_text, num_words):
    # Encode seed text
    input_ids = tokenizer.encode(seed_text, return_tensors='pt')
    
    # Generate text menggunakan GPT-2
    output = model.generate(
        input_ids, 
        max_length=num_words,  # Jumlah kata yang akan dihasilkan
        num_return_sequences=1,  # Hanya menghasilkan 1 teks
        no_repeat_ngram_size=2,  # Untuk menghindari pengulangan
        top_p=0.95,  # Sampling top-p
        temperature=0.7,  # Mengatur temperature untuk hasil yang lebih variatif
        do_sample=True  # Aktifkan sampling untuk variasi teks
    )
    
    # Decode hasil dari token menjadi teks
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Contoh penggunaan
seed_text = "Push-up"
print(f"Generated Exercise Description:\n{generate_exercise_description_gpt(seed_text, 50)}")

# Fungsi saran latihan menggunakan dataset dan GPT
def suggest_exercises(current_weight, target_weight, exercise_level):
    # Calculate weight difference (goal)
    weight_difference = current_weight - target_weight
    
    # Filter exercises berdasarkan level latihan pengguna
    filtered_exercises = data[data['Level'] == exercise_level]
    
    # Filter lebih lanjut berdasarkan intensitas yang diperlukan
    if weight_difference > 10:  # Jika perlu menurunkan lebih dari 10 kg
        # Fokus pada strength dan cardio
        recommended_exercises = filtered_exercises[
            (filtered_exercises['Type'] == 'Strength') | (filtered_exercises['Type'] == 'Cardio')
        ]
    else:
        # Jika penurunan berat badan minimal, fokus pada strength atau mild cardio
        recommended_exercises = filtered_exercises[
            (filtered_exercises['Type'] == 'Strength') | (filtered_exercises['Type'] == 'Mild Cardio')
        ]
    
    # Tampilkan latihan yang disarankan
    print(f"\nRecommended exercises for {exercise_level} level to achieve your target weight:\n")
    for idx, row in recommended_exercises.head(10).iterrows():
        print(f"{idx+1}. {row['Title']}")
        print(f"   Description: {generate_exercise_description_gpt(row['Title'], 50)}")
        print(f"   Body Part: {row['BodyPart']}")
        print(f"   Equipment: {row['Equipment']}")
        print("-" * 50)

# Input: current weight, target weight, and level
current_weight = float(input("Enter your current weight (kg): "))
target_weight = float(input("Enter your target weight (kg): "))
exercise_level = input("Enter your exercise level (Beginner/Intermediate/Expert): ")

# Panggil fungsi untuk memberikan saran latihan
suggest_exercises(current_weight, target_weight, exercise_level)
