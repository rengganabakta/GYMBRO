import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import os
from dotenv import load_dotenv
import re
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Memuat variabel dari file .env
load_dotenv()
dataset_path = os.getenv('DATASET_PATH')

# Memuat dataset
data = pd.read_csv(dataset_path)

# Preprocessing dataset
data['text'] = data['Title'] + ": " + data['Desc'] + " (" + data['Type'] + ") - " + data['BodyPart'] + " using " + data['Equipment'] + " - Level: " + data['Level'] + " Rating: " + data['RatingDesc']
data['text'] = data['text'].fillna('').str.replace(r'[^a-zA-Z0-9\s]', '', regex=True).str.lower()
data = data.drop_duplicates(subset=['text'])
texts = data['text'].values

# Load pre-trained T5 model dan tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Fungsi untuk menghitung BMI
def calculate_bmi(weight, height):
    return weight / (height / 100) ** 2

# Fungsi untuk mencari deskripsi dalam dataset
def get_description_from_dataset(title):
    matching_row = data[data['Title'].str.lower() == title.lower()]
    if not matching_row.empty:
        return matching_row.iloc[0]['Desc']
    return None

# Fungsi untuk menghasilkan deskripsi latihan baru menggunakan T5
def generate_exercise_description_t5(seed_text, num_words=150, keywords=None, max_attempts=3):
    attempts = 0
    while attempts < max_attempts:
        input_text = f"generate description: {seed_text}"  # Format input untuk T5
        input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        output = model.generate(
            input_ids,
            max_length=num_words,
            num_return_sequences=1,
            no_repeat_ngram_size=4,
            top_p=0.95,
            temperature=0.5,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        if keywords and all(keyword.lower() in generated_text.lower() for keyword in keywords):
            return generated_text
        logging.info(f"Generated text does not match keywords: Attempt {attempts + 1}")
        attempts += 1
    return "Failed to generate a valid description within the attempts limit."

# Fungsi untuk memvalidasi deskripsi berdasarkan kata kunci yang relevan
def validate_description(description, keywords):
    if len(description.split()) < 10:  # Panjang minimum teks
        return False
    return all(keyword.lower() in description.lower() for keyword in keywords)

# Fungsi untuk membersihkan deskripsi dari kata-kata yang tidak relevan
def clean_description(description):
    irrelevant_words = ["random_word1", "random_word2"]
    for word in irrelevant_words:
        description = re.sub(r'\b' + word + r'\b', '', description)
    return description.strip()

# Fungsi untuk menggabungkan generasi, validasi, dan pembersihan deskripsi
def generate_and_validate(seed_text, max_length=150, keywords=None):
    if keywords is None:
        keywords = ["muscle", "body"]

    # Coba cari deskripsi di dataset
    description = get_description_from_dataset(seed_text)
    if description:
        return description  # Jika ditemukan, gunakan deskripsi dari dataset

    # Jika tidak ditemukan, buat deskripsi baru
    description = generate_exercise_description_t5(seed_text, max_length, keywords)
    if not validate_description(description, keywords):
        logging.info("Description not relevant. Regenerating...")
        description = generate_exercise_description_t5(seed_text, max_length, keywords)

    description = clean_description(description)
    return description

# Fungsi saran latihan dengan validasi
def suggest_exercises_with_validation(current_weight, target_weight, height, exercise_level):
    bmi = calculate_bmi(current_weight, height)
    weight_difference = current_weight - target_weight
    filtered_exercises = data[data['Level'] == exercise_level]

    if weight_difference > 10 or bmi > 25:
        recommended_exercises = filtered_exercises[(filtered_exercises['Type'] == 'Strength') | (filtered_exercises['Type'] == 'Cardio')]
    else:
        recommended_exercises = filtered_exercises[(filtered_exercises['Type'] == 'Strength') | (filtered_exercises['Type'] == 'Mild Cardio')]

    print(f"\nNumber of recommended exercises: {len(recommended_exercises)}\n")
    print(f"Recommended exercises for {exercise_level} level to achieve your target weight:\n")

    for idx, row in recommended_exercises.head(10).iterrows():
        title = row['Title']
        keywords = ["chest", "arm", "strength"] if title.lower() == "push-up" else ["muscle", "body"]
        description = generate_and_validate(title, 150, keywords)

        print(f"{idx+1}. {title}")
        print(f"   Description: {description}")
        print(f"   Body Part: {row['BodyPart']}")
        print(f"   Equipment: {row['Equipment']}")
        print("-" * 50)

    return recommended_exercises

# Fungsi untuk menyimpan feedback pengguna ke file CSV
def save_feedback(title, description, rating):
    feedback_df = pd.DataFrame([[title, description, rating]], columns=['Title', 'GeneratedDescription', 'Rating'])
    if os.path.exists('feedback_data.csv'):
        feedback_df.to_csv('feedback_data.csv', mode='a', header=False, index=False)
    else:
        feedback_df.to_csv('feedback_data.csv', mode='w', header=True, index=False)
    print("Feedback saved.")

# Input pengguna
current_weight = float(input("Enter your current weight (kg): "))
target_weight = float(input("Enter your target weight (kg): "))
height = float(input("Enter your height (cm): "))
exercise_level = input("Enter your exercise level (Beginner/Intermediate/Expert): ")

# Panggil fungsi saran latihan dan simpan hasilnya
recommended_exercises = suggest_exercises_with_validation(current_weight, target_weight, height, exercise_level)

# Proses penilaian untuk hasil generasi
for i in range(min(3, len(recommended_exercises))):
    row = recommended_exercises.iloc[i]
    title = row['Title']
    description = generate_and_validate(title, 150)
    print(f"\nExercise: {title}")
    print(f"Generated Description: {description}")

    rating = int(input("Rate the description (1-5): "))
    save_feedback(title, description, rating)
