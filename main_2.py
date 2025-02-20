import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, T5Tokenizer, T5ForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

# Load GPT-2 model dan tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

# Load T5 model dan tokenizer dengan legacy=False
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Fungsi untuk menghasilkan deskripsi dengan GPT-2
def generate_description_gpt2(seed_text, num_words=150, keywords=None):
    input_ids = gpt2_tokenizer.encode(seed_text, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    output = gpt2_model.generate(
        input_ids, 
        max_length=num_words, 
        num_return_sequences=1,
        no_repeat_ngram_size=4, 
        top_p=0.85, 
        temperature=0.7, 
        do_sample=True,
        attention_mask=attention_mask,
        pad_token_id=gpt2_tokenizer.eos_token_id
    )
    generated_text = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Fungsi untuk menghasilkan deskripsi dengan T5
def generate_description_t5(seed_text, num_words=150, keywords=None):
    input_text = f"Generate a description: {seed_text}"
    input_ids = t5_tokenizer.encode(input_text, return_tensors='pt')
    output = t5_model.generate(
        input_ids, 
        max_length=num_words, 
        num_beams=5, 
        no_repeat_ngram_size=4,
        top_p=0.85,
        temperature=0.7,
        pad_token_id=t5_tokenizer.pad_token_id
    )
    generated_text = t5_tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Fungsi untuk menghitung kemiripan teks
def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    similarity = cosine_similarity(vectors)
    return similarity[0, 1]

# Gabungan hasil dari GPT-2 dan T5 dengan deteksi kesamaan
def generate_combined_descriptions(seed_text, num_words=150, keywords=None):
    gpt2_result = generate_description_gpt2(seed_text, num_words, keywords)
    t5_result = generate_description_t5(seed_text, num_words, keywords)
    
    # Hitung kesamaan
    similarity_score = calculate_similarity(gpt2_result, t5_result)
    print(f"Similarity Score: {similarity_score:.2f}")
    
    if similarity_score > 0.9:  # Jika teks serupa
        print("Outputs are highly similar. Returning one output.")
        return gpt2_result  # Atau pilih T5 sebagai output
    else:
        combined_result = f"{gpt2_result}\n\n{t5_result}"
        return combined_result

# Fungsi utama untuk pengguna
def suggest_exercises_with_dual_models(current_weight, height, exercise_level):
    # Hitung BMI dan target weight
    bmi = current_weight / (height ** 2)
    target_bmi = 22.5  # Target BMI ideal
    target_weight = target_bmi * (height ** 2)
    
    print(f"Your current BMI is: {bmi:.2f}")
    print(f"Your target weight for a BMI of {target_bmi:.1f} is: {target_weight:.2f} kg")

    weight_difference = current_weight - target_weight
    filtered_exercises = data[data['Level'] == exercise_level]
    if weight_difference > 10:
        recommended_exercises = filtered_exercises[(filtered_exercises['Type'] == 'Strength') | (filtered_exercises['Type'] == 'Cardio')]
    else:
        recommended_exercises = filtered_exercises[(filtered_exercises['Type'] == 'Strength') | (filtered_exercises['Type'] == 'Mild Cardio')]
    print(f"\nNumber of recommended exercises: {len(recommended_exercises)}\n")
    print(f"Recommended exercises for {exercise_level} level to achieve your target BMI:\n")
    for idx, row in recommended_exercises.head(10).iterrows():
        title = row['Title']
        keywords = ["chest", "arm", "strength"] if title.lower() == "push-up" else ["muscle", "body"]
        combined_result = generate_combined_descriptions(title, 150, keywords)
        print(f"{idx+1}. {title}")
        print(f"   Description: {combined_result}")
        print(f"   Body Part: {row['BodyPart']}")
        print(f"   Equipment: {row['Equipment']}")
        print("-" * 50)
    return recommended_exercises

# Input pengguna
current_weight = float(input("Enter your current weight (kg): "))
height = float(input("Enter your height (m): "))
exercise_level = input("Enter your exercise level (Beginner/Intermediate/Expert): ")

# Panggil fungsi utama
recommended_exercises = suggest_exercises_with_dual_models(current_weight, height, exercise_level)
