import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
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
            "labels": self.inputs["input_ids"][idx],  # For causal LM, input_ids are used as labels
        }

def preprocess_and_merge(datasets):
    merged_data = []
    
    for file_path in datasets:
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
            raise ValueError(f"Tidak dapat memproses dataset {file_path}")
        
        merged_data.append(df[["Exercise", "Description", "MuscleGroup", "Level"]])

    final_dataset = pd.concat(merged_data, ignore_index=True).drop_duplicates()
    final_dataset.fillna("Tidak Tersedia", inplace=True)
    return final_dataset

def train_model(model, tokenizer, datasets):
    texts = preprocess_and_merge(datasets)['Exercise'].tolist()
    train_dataset = TokenizedDataset(texts, tokenizer)

    training_args = TrainingArguments(
        output_dir="./model_checkpoint",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        save_steps=500,
        logging_dir="./logs",
        logging_steps=100,
        overwrite_output_dir=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()

def generate_response(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        inputs.input_ids, 
        attention_mask=inputs.attention_mask, 
        max_length=500, 
        num_beams=5, 
        temperature=0.7, 
        top_p=0.9, 
        no_repeat_ngram_size=2, 
        early_stopping=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    global tokenizer, model
    MODEL_NAME = "EleutherAI/gpt-neo-125M"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    TRAIN_MODEL = False  # Saklar untuk melatih ulang atau memuat model
    #Set TRAIN_MODEL = True untuk melatih ulang dan TRAIN_MODEL = False untuk memuat model yang ada.

    if TRAIN_MODEL:
        os.makedirs("./final_model", exist_ok=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        datasets = ["Dataset/megaGymDataset.csv", "Dataset/exercise_dataset.csv", "Dataset/Top 50 Excerice for your body.csv"]
        print("Memulai pelatihan model...")
        train_model(model, tokenizer, datasets)
        model.save_pretrained("./final_model")
        tokenizer.save_pretrained("./final_model")
        print("Pelatihan model selesai.")
    else:
        if not os.path.exists("./final_model"):
            raise FileNotFoundError("Direktori './final_model' tidak ditemukan. Latih ulang model terlebih dahulu.")
        print("Memuat model yang sudah dilatih...")
        model = AutoModelForCausalLM.from_pretrained("./final_model")
        tokenizer = AutoTokenizer.from_pretrained("./final_model")
        print("Model berhasil dimuat.")

    while True:
        prompt = input("Masukkan pertanyaan atau prompt (ketik 'exit' untuk keluar): ")
        if prompt.lower() == 'exit':
            print("Keluar dari program.")
            break
        response = generate_response(prompt, model, tokenizer)
        print("AI: ", response)

if __name__ == "__main__":
    main()
