from dotenv import load_dotenv
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os

def calculate_ideal_weight(height, weight):
    height_m = height / 100  # Convert height to meters
    ideal_bmi = 22.5  # Typical value for a healthy BMI
    ideal_weight = ideal_bmi * (height_m ** 2)
    return ideal_weight

def recommend_exercise(model, tokenizer, weight, ideal_weight):
    weight_diff = weight - ideal_weight
    goal = "lose weight" if weight_diff > 0 else "maintain weight"

    prompt = f"Recommend an exercise to {goal}."
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).input_ids
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
    recommendation = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return recommendation

# Load dataset
class ExerciseDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

# Load GPT-2 model and tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add padding token
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

def train_gpt2(model, tokenizer, dataset, epochs=3, batch_size=8, lr=5e-5):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=lr)

    # Synchronize tokenizer with model
    model.resize_token_embeddings(len(tokenizer))

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Training GPT-2 Epoch {epoch + 1}"):
            try:
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).input_ids
                labels = inputs.clone()

                outputs = model(inputs, labels=labels)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() 
            except IndexError as e:
                print(f"IndexError encountered: {e}. Skipping batch...")
                continue  # Skip problematic batch
        print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(dataloader)}")

# Prepare dataset for GPT-2 training
load_dotenv()
dataset_path = os.getenv('DATASET_PATH')
data = pd.read_csv(dataset_path)
data['text'] = (
    data['Title'] + ": " + data['Desc'] + " (" + data['Type'] + ") - " +
    data['BodyPart'] + " using " + data['Equipment'] + " - Level: " +
    data['Level'] + " Rating: " + data['RatingDesc']
).fillna('').astype(str)
dataset = ExerciseDataset(data['text'].values)

# Train GPT-2
train_gpt2(gpt2_model, gpt2_tokenizer, dataset, epochs=3)

def fine_tune_t5(model, tokenizer, dataset, epochs=3, batch_size=8, lr=5e-5):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Training T5 Epoch {epoch + 1}"):
            try:
                inputs = tokenizer(batch['input_text'], return_tensors="pt", padding=True, truncation=True).input_ids
                labels = tokenizer(batch['target_text'], return_tensors="pt", padding=True, truncation=True).input_ids

                outputs = model(input_ids=inputs, labels=labels)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            except IndexError as e:
                print(f"IndexError encountered: {e}. Skipping batch...")
                continue  # Skip problematic batch
        print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(dataloader)}")

# Prepare dataset for T5 training
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

t5_training_data = [{
    "input_text": row['text'],
    "target_text": row['Desc']  # Use specific description as target
} for _, row in data.iterrows()]

t5_dataset = ExerciseDataset(t5_training_data)

# Train T5
fine_tune_t5(t5_model, t5_tokenizer, t5_dataset, epochs=3)

# User input for height and weight
height = float(input("Enter your height in cm: "))
weight = float(input("Enter your current weight in kg: "))
ideal_weight = calculate_ideal_weight(height, weight)

print(f"Your ideal weight based on your height ({height} cm) is approximately {ideal_weight:.2f} kg.")

# Generate exercise recommendation
exercise_recommendation = recommend_exercise(gpt2_model, gpt2_tokenizer, weight, ideal_weight)
print(f"Recommended exercise: {exercise_recommendation}")

print("Training completed for GPT-2 and T5.")
