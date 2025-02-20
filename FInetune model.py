from transformers import GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import os

# Load Dataset
dataset_path = "Dataset/megaGymDataset.csv"
data = pd.read_csv(dataset_path)

# Preprocess Data for GPT-2 and T5
data['formatted'] = data['Title'] + ": " + data['Desc']
data = data.dropna(subset=['formatted']).reset_index(drop=True)

# GPT-2 Fine-Tuning
def fine_tune_gpt2(data, model_name="gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Avoid padding token errors

    # Format data ke Dataset
    dataset = Dataset.from_pandas(data[['formatted']])
    def tokenize_function(examples):
        tokenized = tokenizer(examples['formatted'], truncation=True, padding=True, max_length=512)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir="./gpt2-finetuned",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_steps=100,
        save_total_limit=2,
        learning_rate=5e-5,
        logging_dir="./gpt2_logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    model.save_pretrained("./gpt2-finetuned")
    tokenizer.save_pretrained("./gpt2-finetuned")
    print("GPT-2 fine-tuning completed.")

# T5 Fine-Tuning
def fine_tune_t5(data, model_name="t5-small"):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Format data ke Dataset
    dataset = Dataset.from_pandas(data[['formatted']])
    def tokenize_function(examples):
        tokenized = tokenizer(examples['formatted'], truncation=True, padding=True, max_length=512)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir="./t5-finetuned",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_steps=100,
        save_total_limit=2,
        learning_rate=5e-5,
        logging_dir="./t5_logs",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    trainer.train()
    model.save_pretrained("./t5-finetuned")
    tokenizer.save_pretrained("./t5-finetuned")
    print("T5 fine-tuning completed.")

# Main
if __name__ == "__main__":
    fine_tune_gpt2(data)
    fine_tune_t5(data)
