import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from torch.utils.data import Dataset
import os

class TokenizedDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )

    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs["input_ids"][idx],
            "attention_mask": self.inputs["attention_mask"][idx],
            "labels": self.inputs["input_ids"][idx],
        }

def preprocess_and_merge(datasets):
    merged_data = []
    
    for file_path in datasets:
        if not os.path.exists(file_path):
            print(f"Peringatan: File tidak ditemukan {file_path}, dilewati.")
            continue
        
        df = pd.read_csv(file_path)
        if "Title" in df.columns:
            df["Exercise"] = df["Title"]
            df["Description"] = df["Desc"]
            df["MuscleGroup"] = df["BodyPart"]
            df["Level"] = df["Level"]
        elif "Name of Exercise" in df.columns:
            df["Exercise"] = df["Name of Exercise"]
            df["Description"] = df["Benefit"]
            df["MuscleGroup"] = df["Target Muscle Group"]
            df["Level"] = df["Difficulty Level"]
        elif "Activity, Exercise or Sport (1 hour)" in df.columns:
            df["Exercise"] = df["Activity, Exercise or Sport (1 hour)"]
            df["Description"] = "Tidak Tersedia"
            df["MuscleGroup"] = "Umum"
            df["Level"] = "Bervariasi"
        else:
            print(f"Peringatan: Format dataset tidak dikenali {file_path}, dilewati.")
            continue
        
        merged_data.append(df[["Exercise", "Description", "MuscleGroup", "Level"]])

    if merged_data:
        final_dataset = pd.concat(merged_data, ignore_index=True).drop_duplicates()
        final_dataset.fillna("Tidak Tersedia", inplace=True)
        print(f"Dataset berhasil diproses. Jumlah data: {len(final_dataset)}")
        return final_dataset
    else:
        print("Peringatan: Tidak ada dataset yang berhasil diproses.")
        return pd.DataFrame(columns=["Exercise", "Description", "MuscleGroup", "Level"])

def generate_response(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        inputs.input_ids, 
        attention_mask=inputs.attention_mask, 
        max_length=500, 
        num_beams=5, 
        temperature=0.7, 
        top_p=0.9, 
        do_sample=True,
        no_repeat_ngram_size=2, 
        early_stopping=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def route_prompt(prompt):
    if any(word in prompt.lower() for word in ["exercise", "workout", "training", "gym"]):
        return "EleutherAI/gpt-neo-125M"
    elif any(word in prompt.lower() for word in ["explain", "summarize", "define"]):
        return "t5-small"
    else:
        return "gpt2"

def main():
    models_to_use = {
        "EleutherAI/gpt-neo-125M": AutoModelForCausalLM,
        "gpt2": AutoModelForCausalLM,
        "t5-small": AutoModelForSeq2SeqLM
    }
    model_cache = {}
    datasets = ["Dataset/megaGymDataset.csv", "Dataset/exercise_dataset.csv", "Dataset/Top 50 Excerice for your body.csv"]
    
    # Preprocess datasets first
    print("Memproses dataset...")
    processed_data = preprocess_and_merge(datasets)
    if processed_data.empty:
        print("Dataset kosong. Program dihentikan.")
        return

    # Load models
    while True:
        prompt = input("Masukkan pertanyaan atau prompt (ketik 'exit' untuk keluar): ")
        if prompt.lower() == 'exit':
            print("Keluar dari program.")
            break

        model_name = route_prompt(prompt)
        print(f"Menggunakan model: {model_name}")

        if model_name not in model_cache:
            print("Memuat model...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model_class = models_to_use[model_name]
            model = model_class.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model_cache[model_name] = (model, tokenizer)

        model, tokenizer = model_cache[model_name]
        response = generate_response(prompt, model, tokenizer)
        print("AI:", response)

if __name__ == "__main__":
    main()
