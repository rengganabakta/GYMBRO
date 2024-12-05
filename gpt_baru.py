import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel
import os
from dotenv import load_dotenv

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

# Fungsi untuk memuat model
def load_model(model_name="gpt-2"):
    if model_name == "gpt-2":
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
    elif model_name == "gpt-neo":
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    else:
        raise ValueError("Invalid model name")
    return tokenizer, model

# Fungsi untuk menghasilkan deskripsi menggunakan GPT-2 dan GPT-Neo
def generate_combined_description(activity, max_length=150):
    tokenizer_gpt2, model_gpt2 = load_model("gpt-2")
    tokenizer_gpt_neo, model_gpt_neo = load_model("gpt-neo")
    
    # Generate dengan GPT-2
    input_ids_gpt2 = tokenizer_gpt2.encode(activity, return_tensors='pt')
    output_gpt2 = model_gpt2.generate(input_ids_gpt2, max_length=max_length)
    desc_gpt2 = tokenizer_gpt2.decode(output_gpt2[0], skip_special_tokens=True)
    
    # Generate dengan GPT-Neo
    input_ids_gpt_neo = tokenizer_gpt_neo.encode(activity, return_tensors='pt')
    output_gpt_neo = model_gpt_neo.generate(input_ids_gpt_neo, max_length=max_length)
    desc_gpt_neo = tokenizer_gpt_neo.decode(output_gpt_neo[0], skip_special_tokens=True)
    
    return f"GPT-2: {desc_gpt2}\n\nGPT-Neo: {desc_gpt_neo}"

# Fungsi utama
def main():
    print("Welcome to Gym Assistant GPT!")
    activity = input("Enter an activity or exercise to learn more: ").strip()
    row = dataset_combined[dataset_combined['Activity'].str.contains(activity, case=False)].iloc[0]
    
    print(f"\nActivity: {row['Activity']}")
    print(f"Sets: {row['Sets']}, Reps: {row['Reps']}")
    print(f"Benefit: {row['Benefit']}")
    print(f"Burns Calories: {row['Burns Calories']} per 30 min")
    print(f"Calories per kg: {row['Calories per kg']}")
    print(f"Target Muscle Group: {row['Target Muscle Group']}")
    print(f"Equipment Needed: {row['Equipment Needed']}")
    print(f"Difficulty Level: {row['Difficulty Level']}")
    
    # Generate combined description
    combined_description = generate_combined_description(row['Activity'])
    print(f"\nDescription:\n{combined_description}")

if __name__ == "__main__":
    main()
