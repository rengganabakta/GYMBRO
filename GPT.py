from transformers import GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
import torch
from datasets import Dataset
import pandas as pd
import os

# Load Dataset for Fine-Tuning
def load_dataset_for_finetuning():
    dataset_path = "Dataset/megaGymDataset.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset not found. Please make sure megaGymDataset.csv exists.")

    data = pd.read_csv(dataset_path)
    print("Dataset loaded. Number of samples:", len(data))  # Log dataset size
    data['formatted'] = data['Title'] + ": " + data['Desc']
    data = data.dropna(subset=['formatted']).reset_index(drop=True)
    print("Dataset after preprocessing. Number of samples:", len(data))  # Log size after preprocessing
    return Dataset.from_pandas(data[['formatted']])

# Fine-Tune GPT-2
def fine_tune_gpt2():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set

    dataset = load_dataset_for_finetuning()
    def tokenize_function(examples):
        tokenized = tokenizer(examples['formatted'], truncation=True, padding=True, max_length=512)
        tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    print("Tokenized dataset example:", tokenized_dataset[0])  # Log a tokenized sample

    training_args = TrainingArguments(
        output_dir="./gpt2-finetuned",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        save_steps=100,
        save_total_limit=2,
        learning_rate=2e-5,
        logging_dir="./gpt2_logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("Starting fine-tuning for GPT-2...")
    trainer.train()
    model.save_pretrained("./gpt2-finetuned")
    tokenizer.save_pretrained("./gpt2-finetuned")
    print("GPT-2 fine-tuning completed.")

# Fine-Tune T5
def fine_tune_t5():
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    dataset = load_dataset_for_finetuning()
    def tokenize_function(examples):
        tokenized = tokenizer(examples['formatted'], truncation=True, padding=True, max_length=512)
        tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    print("Tokenized dataset example:", tokenized_dataset[0])  # Log a tokenized sample

    training_args = TrainingArguments(
        output_dir="./t5-finetuned",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        save_steps=100,
        save_total_limit=2,
        learning_rate=2e-5,
        logging_dir="./t5_logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("Starting fine-tuning for T5...")
    trainer.train()
    model.save_pretrained("./t5-finetuned")
    tokenizer.save_pretrained("./t5-finetuned")
    print("T5 fine-tuning completed.")

# Load Fine-Tuned Models
def load_finetuned_model_gpt2():
    model_path = "./gpt2-finetuned"
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model, tokenizer

def load_finetuned_model_t5():
    model_path = "./t5-finetuned"
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    return model, tokenizer

# Generate Text with GPT-2
def generate_with_gpt2(prompt, model, tokenizer, max_length=250):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True, padding=True)
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Generate Text with T5
def generate_with_t5(prompt, model, tokenizer, max_length=250):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True, padding=True)
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Calculate BMI
def calculate_bmi(weight, height):
    return weight / (height / 100) ** 2

# Suggest Exercises Based on Input
def suggest_exercises(weight, target_weight, height, level, model_gpt2, tokenizer_gpt2, model_t5, tokenizer_t5):
    bmi = calculate_bmi(weight, height)
    prompt_gpt2 = (
        f"Provide a list of 5 beginner exercises to help reduce weight from {weight}kg to {target_weight}kg. "
        f"Include details on how each exercise helps in weight loss and mention their difficulty."
    )
    prompt_t5 = (
        f"Describe 5 beginner-level exercises suitable for a person weighing {weight}kg, targeting {target_weight}kg, "
        f"with a height of {height}cm. Provide detailed descriptions, including benefits and required equipment."
    )

    print("\n=== GPT-2 Suggested Exercises ===")
    gpt2_output = generate_with_gpt2(prompt_gpt2, model_gpt2, tokenizer_gpt2)
    print(gpt2_output)

    print("\n=== T5 Exercise Description ===")
    t5_output = generate_with_t5(prompt_t5, model_t5, tokenizer_t5)
    print(t5_output)

if __name__ == "__main__":
    # Example Input
    action = input("Choose an action (fine-tune/test): ").strip().lower()

    if action == "fine-tune":
        model_type = input("Which model to fine-tune? (gpt2/t5): ").strip().lower()
        if model_type == "gpt2":
            fine_tune_gpt2()
        elif model_type == "t5":
            fine_tune_t5()
        else:
            print("Invalid model type.")
    elif action == "test":
        current_weight = float(input("Enter your current weight (kg): "))
        target_weight = float(input("Enter your target weight (kg): "))
        height = float(input("Enter your height (cm): "))
        level = input("Enter your exercise level (Beginner/Intermediate/Expert): ")

        # Load Models
        gpt2_model, gpt2_tokenizer = load_finetuned_model_gpt2()
        t5_model, t5_tokenizer = load_finetuned_model_t5()

        # Suggest Exercises
        suggest_exercises(current_weight, target_weight, height, level, gpt2_model, gpt2_tokenizer, t5_model, t5_tokenizer)
    else:
        print("Invalid action.")
