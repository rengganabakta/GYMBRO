# Path: gym_assistant/main.py

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from dotenv import load_dotenv
import re

# Memuat variabel lingkungan dari file .env
load_dotenv()
dataset_path = os.getenv('DATASET_PATH')  # Dataset lama
dataset_new_1 = os.getenv('DATASET_NEW_1')  # Dataset baru 1
dataset_new_2 = os.getenv('DATASET_NEW_2')  # Dataset baru 2

# Memuat dataset
try:
    data_old = pd.read_csv(dataset_path)
    data_new_1 = pd.read_csv(dataset_new_1)
    data_new_2 = pd.read_csv(dataset_new_2)
    
    # Gabungkan dataset lama dan baru
    data_old.columns = ['Activity', '130 lb', '155 lb', '180 lb', '205 lb', 'Calories per kg']
    data_new_1.columns = [
        'Name of Exercise', 'Sets', 'Reps', 'Benefit', 'Burns Calories (per 30 min)',
        'Target Muscle Group', 'Equipment Needed', 'Difficulty Level'
    ]
    
    # Gabungkan dataset baru ke struktur unified
    dataset_combined = pd.DataFrame()
    dataset_combined['Activity'] = data_new_1['Name of Exercise']
    dataset_combined['Sets'] = data_new_1['Sets']
    dataset_combined['Reps'] = data_new_1['Reps']
    dataset_combined['Benefit'] = data_new_1['Benefit']
    dataset_combined['Burns Calories'] = data_new_1['Burns Calories (per 30 min)']
    dataset_combined['Target Muscle Group'] = data_new_1['Target Muscle Group']
    dataset_combined['Equipment Needed'] = data_new_1['Equipment Needed']
    dataset_combined['Difficulty Level'] = data_new_1['Difficulty Level']
    
    # Tambahkan kolom Calories per kg dari dataset lama
    calorie_map = data_old.set_index('Activity')['Calories per kg'].to_dict()
    dataset_combined['Calories per kg'] = dataset_combined['Activity'].map(calorie_map)
    
    # Isi nilai NaN pada Calories per kg dengan rata-rata
    dataset_combined['Calories per kg'].fillna(dataset_combined['Calories per kg'].mean(), inplace=True)

except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Load pre-trained GPT model gratis (misalnya GPT-J atau GPT-Neo)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")  # Model 1.3B
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

# Fungsi untuk menghasilkan deskripsi latihan
def generate_exercise_description(seed_text, max_length=150, keywords=None):
    input_ids = tokenizer.encode(seed_text, return_tensors='pt')
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=4,
        top_p=0.85,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    if keywords and not all(keyword.lower() in generated_text.lower() for keyword in keywords):
        return generate_exercise_description(seed_text, max_length, keywords)
    return generated_text

# Fungsi validasi input pengguna
def validate_input(prompt, value_type=float, valid_options=None):
    while True:
        try:
            user_input = value_type(input(prompt))
            if valid_options and user_input not in valid_options:
                raise ValueError("Invalid option.")
            return user_input
        except Exception as e:
            print(f"Invalid input: {e}. Please try again.")

# Perhitungan BMI
def calculate_bmi(weight, height):
    height_in_m = height / 100
    bmi = weight / (height_in_m ** 2)
    return round(bmi, 2)

# Fungsi saran latihan berdasarkan BMI
def suggest_exercises(current_weight, target_weight, height, exercise_level):
    bmi = calculate_bmi(current_weight, height)
    weight_difference = current_weight - target_weight
    filtered_exercises = data[data['Level'] == exercise_level.capitalize()]
    
    # Klasifikasi saran berdasarkan BMI
    if bmi >= 25 or weight_difference > 10:
        recommended_exercises = filtered_exercises[
            (filtered_exercises['Type'].str.contains('Strength')) |
            (filtered_exercises['Type'].str.contains('Cardio'))
        ]
    else:
        recommended_exercises = filtered_exercises[
            (filtered_exercises['Type'].str.contains('Strength')) |
            (filtered_exercises['Type'].str.contains('Mild Cardio'))
        ]
    return recommended_exercises

# Fungsi utama
def main():
    print("Welcome to Gym Assistant GPT!")
    current_weight = validate_input("Enter your current weight (kg): ", float)
    height = validate_input("Enter your height (cm): ", float)
    target_weight = validate_input("Enter your target weight (kg): ", float)
    exercise_level = validate_input("Enter your exercise level (Beginner/Intermediate/Expert): ", str, 
                                    ["Beginner", "Intermediate", "Expert"])
    
    bmi = calculate_bmi(current_weight, height)
    print(f"\nYour BMI is: {bmi} (Classification: {'Overweight' if bmi >= 25 else 'Normal'})")
    
    recommended_exercises = suggest_exercises(current_weight, target_weight, height, exercise_level)
    print(f"\nRecommended exercises for your level ({exercise_level}):")
    
    for idx, row in recommended_exercises.head(5).iterrows():
        title = row['Title']
        keywords = ["muscle", "body"]
        description = generate_exercise_description(title, 150, keywords)
        print(f"\n{idx + 1}. {title}")
        print(f"   Description: {description}")
        print(f"   Body Part: {row['BodyPart']}")
        print(f"   Equipment: {row['Equipment']}")
        print("-" * 50)

if __name__ == "__main__":
    main()
