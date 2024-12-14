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

def train_model(model, tokenizer, datasets, model_name):
    texts = preprocess_and_merge(datasets)['Exercise'].tolist()
    train_dataset = TokenizedDataset(texts, tokenizer)

    training_args = TrainingArguments(
        output_dir=f"./model_checkpoint_{model_name.replace('/', '_')}",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        save_steps=500,
        logging_dir=f"./logs_{model_name.replace('/', '_')}",
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
        do_sample=True,  # Mengaktifkan sampling untuk output lebih bervariasi
        no_repeat_ngram_size=2, 
        early_stopping=True
    )
    print("DEBUG: Generated Output (Raw):", tokenizer.decode(outputs[0], skip_special_tokens=False))  # Debugging
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    models_to_use = {
        "EleutherAI/gpt-neo-125M": AutoModelForCausalLM,
        "gpt2": AutoModelForCausalLM,
        "t5-small": AutoModelForSeq2SeqLM
    }

    TRAIN_MODEL = False  # Saklar untuk melatih ulang atau memuat model
    #Set TRAIN_MODEL = True untuk melatih ulang dan TRAIN_MODEL = False untuk memuat model yang ada.

    datasets = ["Dataset/megaGymDataset.csv", "Dataset/exercise_dataset.csv", "Dataset/Top 50 Exercise for your body.csv"]

    for model_name, model_class in models_to_use.items():
        print(f"Using model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = model_class.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if TRAIN_MODEL:
            print(f"Training {model_name}...")
            train_model(model, tokenizer, datasets, model_name)
            model.save_pretrained(f"./final_model_{model_name.replace('/', '_')}" )
            tokenizer.save_pretrained(f"./final_model_{model_name.replace('/', '_')}" )
            print(f"Model {model_name} trained and saved.")
        else:
            model_path = f"./final_model_{model_name.replace('/', '_')}"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model directory {model_path} not found. Train the model first.")
            print(f"Loading model from {model_path}...")
            model = model_class.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)

        while True:
            prompt = input(f"{model_name} - Masukkan pertanyaan atau prompt (ketik 'exit' untuk keluar): ")
            if prompt.lower() == 'exit':
                print(f"Keluar dari {model_name}.")
                break
            response = generate_response(prompt, model, tokenizer)
            print(f"{model_name} AI: ", response)

if __name__ == "__main__":
    main()
