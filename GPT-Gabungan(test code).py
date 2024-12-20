import pandas as pd
from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, MarianMTModel, MarianTokenizer, Trainer, TrainingArguments)
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os

# Load Translation Model
translation_model_name = "Helsinki-NLP/opus-mt-id-en"
translator_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
translator_model = MarianMTModel.from_pretrained(translation_model_name)

def translate_prompt(prompt, target_language="en"):
    inputs = translator_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    translated_tokens = translator_model.generate(**inputs)
    translated_prompt = translator_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_prompt

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_name == "t5-small":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

class TokenizedDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        )

    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs["input_ids"][idx],
            "attention_mask": self.inputs["attention_mask"][idx],
            "labels": self.inputs["input_ids"][idx],
        }

def preprocess_and_merge_v2(datasets):
    merged_data = []
    for file_path in datasets:
        df = pd.read_csv(file_path)
        if 'Title' in df.columns and 'Desc' in df.columns:
            df['Exercise'] = df['Title']
            df['Description'] = df['Desc']
        elif 'Name of Exercise' in df.columns and 'Benefit' in df.columns:
            df['Exercise'] = df['Name of Exercise']
            df['Description'] = df['Benefit']
        elif 'Activity, Exercise or Sport (1 hour)' in df.columns:
            df['Exercise'] = df['Activity, Exercise or Sport (1 hour)']
            df['Description'] = "General Cardio Exercise"
        else:
            print(f"Peringatan: Format dataset tidak dikenali {file_path}, dilewati.")
            continue
        final_df = df[['Exercise', 'Description']].drop_duplicates()
        merged_data.append(final_df)
    if merged_data:
        final_dataset = pd.concat(merged_data, ignore_index=True).drop_duplicates()
        final_dataset.fillna("Tidak Tersedia", inplace=True)
        final_dataset.to_csv("processed_dataset.csv", index=False)
        print(f"Dataset berhasil diproses. Jumlah data: {len(final_dataset)}")
        return final_dataset
    else:
        print("Peringatan: Tidak ada dataset yang berhasil diproses.")
        return pd.DataFrame(columns=['Exercise', 'Description'])

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    pred_labels = predictions.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, pred_labels),
        "f1": f1_score(labels, pred_labels, average='weighted'),
        "precision": precision_score(labels, pred_labels, average='weighted'),
        "recall": recall_score(labels, pred_labels, average='weighted')
    }

def fine_tune_model_v2(processed_data, model_name, model, tokenizer):
    if processed_data.empty:
        print(f"Dataset kosong. Pelatihan untuk {model_name} dihentikan.")
        return

    texts = (processed_data["Exercise"] + ": " + processed_data["Description"]).tolist()
    print(f"Melatih model {model_name} menggunakan dataset dengan {len(texts)} sampel.")

    train_dataset = TokenizedDataset(texts, tokenizer)
    eval_dataset = TokenizedDataset(texts, tokenizer)

    training_args = TrainingArguments(
        output_dir=f"./model_checkpoint_{model_name.replace('/', '_')}",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,  
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=5e-5,
        logging_dir=f"./logs_{model_name.replace('/', '_')}",
        logging_steps=100,
        overwrite_output_dir=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()
    model.save_pretrained(f"./final_model_{model_name.replace('/', '_')}")
    tokenizer.save_pretrained(f"./final_model_{model_name.replace('/', '_')}")
    print(f"Model {model_name} berhasil dilatih dan disimpan.")

if __name__ == "__main__":
    TRAIN_MODEL = True
    datasets = ["Dataset/megaGymDataset.csv", "Dataset/exercise_dataset.csv", "Dataset/Top 50 Excerice for your body.csv"]
    print("Memproses dataset...")
    processed_data = preprocess_and_merge_v2(datasets)
    if processed_data.empty:
        print("Dataset kosong. Program dihentikan.")
    else:
        model_names = {"EleutherAI/gpt-neo-125M": "GPT-Neo", "gpt2": "GPT-2", "t5-small": "GPT-T5"}
        if TRAIN_MODEL:
            for model_name, label in model_names.items():
                model, tokenizer = load_model_and_tokenizer(model_name)
                fine_tune_model_v2(processed_data, model_name, model, tokenizer)
