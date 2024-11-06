import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os
from dotenv import load_dotenv
import re

# Memuat variabel dari file .env
load_dotenv()
dataset_path = os.getenv('DATASET_PATH')

# Memuat dataset
data = pd.read_csv(dataset_path)

# Gabungkan kolom yang relevan untuk menghasilkan teks
data['text'] = data['Title'] + ": " + data['Desc'] + " (" + data['Type'] + ") - " + data['BodyPart'] + " using " + data['Equipment'] + " - Level: " + data['Level'] + " Rating: " + data['RatingDesc']
data['text'] = data['text'].fillna('').astype(str)
texts = data['text'].values

# Load pre-trained GPT-2 model dan tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Fungsi untuk menghasilkan deskripsi latihan baru menggunakan GPT-2
def generate_exercise_description_gpt(seed_text, num_words=150, keywords=None):
    # Encode seed text
    input_ids = tokenizer.encode(seed_text, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)  # Atur attention mask
    
    # Generate text using GPT-2
    output = model.generate(
        input_ids, 
        max_length=num_words,  # Panjang output yang lebih besar untuk hasil lebih lengkap
        num_return_sequences=1,
        no_repeat_ngram_size=4,  # Mengurangi pengulangan kata
        top_p=0.85,  # Kurangi nilai ini untuk variasi yang lebih baik
        temperature=0.7,  # Mengatur suhu untuk variasi teks
        do_sample=True,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Validasi dengan kata kunci (untuk meningkatkan relevansi)
    if keywords:
        if all(keyword.lower() in generated_text.lower() for keyword in keywords):
            return generated_text
        else:
            print("Generated description does not match keywords. Regenerating...")
            return generate_exercise_description_gpt(seed_text, num_words, keywords)
    return generated_text

# Fungsi untuk memvalidasi deskripsi berdasarkan kata kunci yang relevan
def validate_description(description, keywords):
    return all(keyword.lower() in description.lower() for keyword in keywords)

# Fungsi untuk membersihkan deskripsi dari kata-kata yang tidak relevan
def clean_description(description):
    irrelevant_words = ["random_word1", "random_word2"]  # Sesuaikan daftar ini dengan kata-kata yang ingin dihilangkan
    for word in irrelevant_words:
        description = re.sub(r'\b' + word + r'\b', '', description)
    return description.strip()

# Fungsi untuk menggabungkan generasi, validasi, dan pembersihan deskripsi
def generate_and_validate(seed_text, max_length=150, keywords=None):
    if keywords is None:
        keywords = ["muscle", "body"]  # Kata kunci umum jika tidak ada yang spesifik
    
    description = generate_exercise_description_gpt(seed_text, max_length, keywords)
    if not validate_description(description, keywords):
        print("Description not relevant. Regenerating...")
        description = generate_exercise_description_gpt(seed_text, max_length, keywords)
    
    description = clean_description(description)
    return description

# Fungsi saran latihan dengan validasi
def suggest_exercises_with_validation(current_weight, target_weight, exercise_level):
    weight_difference = current_weight - target_weight
    filtered_exercises = data[data['Level'] == exercise_level]
    
    if weight_difference > 10:
        recommended_exercises = filtered_exercises[(filtered_exercises['Type'] == 'Strength') | (filtered_exercises['Type'] == 'Cardio')]
    else:
        recommended_exercises = filtered_exercises[(filtered_exercises['Type'] == 'Strength') | (filtered_exercises['Type'] == 'Mild Cardio')]
    
    # Menampilkan jumlah hasil rekomendasi
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
    
    return recommended_exercises  # Return recommended exercises for feedback loop

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
exercise_level = input("Enter your exercise level (Beginner/Intermediate/Expert): ")

# Panggil fungsi saran latihan dan simpan hasilnya
recommended_exercises = suggest_exercises_with_validation(current_weight, target_weight, exercise_level)

# Proses penilaian untuk hasil generasi
for i in range(min(3, len(recommended_exercises))):  # Loop untuk maksimal 3 latihan pertama sebagai contoh
    row = recommended_exercises.iloc[i]
    title = row['Title']
    description = generate_and_validate(title, 150)
    print(f"\nExercise: {title}")
    print(f"Generated Description: {description}")
    
    rating = int(input("Rate the description (1-5): "))
    save_feedback(title, description, rating)
