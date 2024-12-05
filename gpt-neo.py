# Path: gym_assistant/main.py

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
datasets = [
    os.getenv('DATASET_PATH'),  # Dataset 1
    os.getenv('DATASET_NEW_1'),  # Dataset 2
    os.getenv('DATASET_NEW_2')   # Dataset 3
]

# Initialize tokenizer and model
MODEL_NAME = "EleutherAI/gpt-neo-125M"  # Pre-trained GPT-Neo model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Set a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use the end-of-sequence token as the pad token

# Utility function to preprocess datasets
def preprocess_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded dataset from {file_path}. Columns: {list(df.columns)}")

        # Detect and adjust based on known dataset structures
        if "Activity, Exercise or Sport (1 hour)" in df.columns:
            texts = df["Activity, Exercise or Sport (1 hour)"].astype(str).tolist()
        elif "Name of Exercise" in df.columns:
            texts = df["Name of Exercise"].astype(str).tolist()
        elif "Title" in df.columns:  # Assuming 'Title' is the primary column for text
            texts = df["Title"].astype(str).tolist()
        else:
            raise ValueError(f"Unexpected columns: {list(df.columns)} in dataset {file_path}")

        if len(texts) == 0:
            raise ValueError("Dataset is empty after processing.")
        return texts
    except Exception as e:
        print(f"Error in preprocess_dataset: {e}")
        raise

# Train model sequentially on multiple datasets
def train_sequentially(model, tokenizer, datasets):
    for i, dataset_path in enumerate(datasets, start=1):
        if not dataset_path:
            print(f"Dataset {i} path not defined in .env. Skipping.")
            continue

        try:
            texts = preprocess_dataset(dataset_path)
            print(f"Dataset {i} loaded successfully. Number of samples: {len(texts)}.")
            
            # Tokenize the dataset
            tokenized_texts = tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            if len(tokenized_texts["input_ids"]) == 0:
                raise ValueError("Tokenized dataset is empty. Check the input format.")

            # Define training arguments
            training_args = TrainingArguments(
                output_dir=f"./model_checkpoint_{i}",  # Save checkpoints for each stage
                per_device_train_batch_size=4,
                num_train_epochs=1,
                save_steps=500,
                logging_dir=f"./logs_{i}",
                logging_steps=100,
                overwrite_output_dir=True,
            )
            
            # Define Trainer and train model
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_texts["input_ids"],
            )
            trainer.train()
            print(f"Training on dataset {i} completed.")
        except Exception as e:
            print(f"Error processing dataset {i}: {e}")

# Run sequential training
train_sequentially(model, tokenizer, datasets)
