from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Path ke dataset yang sudah diproses
dataset_path = "Dataset/finetune_dataset.csv"

# 1. Load Dataset
dataset = load_dataset("csv", data_files=dataset_path)

# 2. Load Pretrained Model and Tokenizer
MODEL_NAME = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# 3. Menambahkan Padding Token jika Tidak Ada
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))  # Perbarui model dengan token baru

# 4. Tokenize Dataset
def tokenize_function(examples):
    return tokenizer(
        examples["prompt"],
        text_target=examples["response"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 5. Define Training Arguments
training_args = TrainingArguments(
    output_dir="Dataset/finetuned_model",  # Path penyimpanan model
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100,
    fp16=torch.cuda.is_available(),  # Menggunakan FP16 jika GPU tersedia
)

# 6. Trainer Initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

# 7. Fine-Tune the Model
trainer.train()

# 8. Save the Model
model.save_pretrained("Dataset/finetuned_model")
tokenizer.save_pretrained("Dataset/finetuned_model")
print("Fine-tuned model saved to Dataset/finetuned_model")
