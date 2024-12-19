import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, MarianMTModel, MarianTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
import os

# Load Model for Translation
translation_model_name = "Helsinki-NLP/opus-mt-id-en"
translator_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
translator_model = MarianMTModel.from_pretrained(translation_model_name)

def translate_prompt(prompt, target_language="en"):
    try:
        inputs = translator_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        translated_tokens = translator_model.generate(**inputs)
        translated_prompt = translator_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return translated_prompt
    except Exception as e:
        print(f"Terjadi kesalahan dalam penerjemahan: {e}")
        return prompt

# Load Model for Fine-Tuning
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_name == "t5-small":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# Define Tokenized Dataset Class
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

# Preprocess Dataset
def preprocess_and_merge(datasets):
    merged_data = []
    
    for file_path in datasets:
        if not os.path.exists(file_path):
            print(f"Peringatan: File tidak ditemukan {file_path}, dilewati.")
            with open("missing_files.log", "a") as log_file:
                log_file.write(f"File tidak ditemukan: {file_path}\n")
            continue
        
        df = pd.read_csv(file_path)
        if "Title" in df.columns and "Desc" in df.columns:
            df["Exercise"] = df["Title"]
            df["Description"] = df["Desc"]
        elif "Name of Exercise" in df.columns and "Benefit" in df.columns:
            df["Exercise"] = df["Name of Exercise"]
            df["Description"] = df["Benefit"]
        elif "Activity, Exercise or Sport (1 hour)" in df.columns:
            df["Exercise"] = df["Activity, Exercise or Sport (1 hour)"]
            df["Description"] = "Latihan kardio umum"
        else:
            print(f"Peringatan: Format dataset tidak dikenali {file_path}, dilewati.")
            continue
        
        merged_data.append(df[["Exercise", "Description"]])

    if merged_data:
        final_dataset = pd.concat(merged_data, ignore_index=True).drop_duplicates()
        final_dataset.fillna("Tidak Tersedia", inplace=True)
        print(f"Dataset berhasil diproses. Jumlah data: {len(final_dataset)}")
        return final_dataset
    else:
        print("Peringatan: Tidak ada dataset yang berhasil diproses.")
        return pd.DataFrame(columns=["Exercise", "Description"])

# Train Model
def fine_tune_model(processed_data, model_name, model, tokenizer):
    texts = (processed_data["Exercise"] + ": " + processed_data["Description"]).tolist()
    print(f"Melatih model {model_name} menggunakan dataset dengan {len(texts)} sampel.")
    if len(texts) == 0:
        print("Tidak ada data untuk pelatihan. Pastikan dataset telah diproses dengan benar.")
        return

    train_dataset = TokenizedDataset(texts, tokenizer)

    training_args = TrainingArguments(
        output_dir=f"./model_checkpoint_{model_name.replace('/', '_')}",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=10,
        learning_rate=2e-5,
        evaluation_strategy="no",
        save_strategy="epoch",
        save_total_limit=2,
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
    model.save_pretrained(f"./final_model_{model_name.replace('/', '_')}")
    tokenizer.save_pretrained(f"./final_model_{model_name.replace('/', '_')}")
    print(f"Model {model_name} berhasil dilatih dan disimpan.")

# Generate Response
def generate_response(prompt, model, tokenizer, max_length=150):
    translated_prompt = translate_prompt(prompt, "en")
    formatted_prompt = f"You are a fitness assistant. Suggest specific exercises with detailed instructions for: {translated_prompt}."
    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    outputs = model.generate(
        inputs.input_ids, 
        attention_mask=inputs.attention_mask, 
        max_length=max_length,  
        num_beams=5, 
        temperature=0.7, 
        top_p=0.9, 
        do_sample=True,
        no_repeat_ngram_size=2, 
        early_stopping=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Main Program
def main():
    TRAIN_MODEL = True

    datasets = ["Dataset/megaGymDataset.csv", "Dataset/exercise_dataset.csv", "Dataset/Top 50 Excerice for your body.csv"]
    processed_data = preprocess_and_merge(datasets)
    model_name = "EleutherAI/gpt-neo-125M"
    model, tokenizer = load_model_and_tokenizer(model_name)

    if TRAIN_MODEL:
        fine_tune_model(processed_data, model_name, model, tokenizer)

    while True:
        prompt = input("Masukkan pertanyaan atau prompt (ketik 'exit' untuk keluar): ")
        if prompt.lower() == 'exit':
            break
        response = generate_response(prompt, model, tokenizer, max_length=150)
        print("AI:", response)

if __name__ == "__main__":
    main()
