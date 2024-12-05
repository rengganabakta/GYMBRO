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

def preprocess_dataset(file_path):
    """
    Processes a dataset file and returns a list of concatenated texts for training.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded dataset from {file_path}. Columns: {list(df.columns)}")

        if all(col in df.columns for col in ['Title', 'Desc', 'Type', 'BodyPart', 'Equipment', 'Level', 'RatingDesc']):
            # Dataset 3 processing
            df['text'] = (
                df['Title'] + ": " + df['Desc'] + " (" + df['Type'] + ") - " +
                df['BodyPart'] + " menggunakan " + df['Equipment'] + " - Level: " +
                df['Level'] + " Rating: " + df['RatingDesc']
            ).fillna('').astype(str)
            texts = df['text'].tolist()

        elif "Activity, Exercise or Sport (1 hour)" in df.columns:
            # Dataset 1 processing
            texts = df["Activity, Exercise or Sport (1 hour)"].astype(str).tolist()

        elif "Name of Exercise" in df.columns:
            # Dataset 2 processing
            df['text'] = (
                df['Name of Exercise'] + ": " + df['Benefit'] + " - Target: " +
                df['Target Muscle Group'] + " - Level: " + df['Difficulty Level']
            ).fillna('').astype(str)
            texts = df['text'].tolist()

        else:
            raise ValueError(f"Unexpected columns: {list(df.columns)} in dataset {file_path}")

        if len(texts) == 0:
            raise ValueError("Dataset is empty after processing.")
        return texts

    except Exception as e:
        print(f"Error in preprocess_dataset: {e}")
        raise

def train_model(model, tokenizer, datasets):
    for i, dataset_path in enumerate(datasets, start=1):
        if not dataset_path:
            print(f"Dataset {i} path not defined. Skipping.")
            continue

        try:
            texts = preprocess_dataset(dataset_path)
            print(f"Dataset {i} loaded successfully. Number of samples: {len(texts)}.")

            train_dataset = TokenizedDataset(texts, tokenizer)

            training_args = TrainingArguments(
                output_dir=f"./model_checkpoint_{i}",
                per_device_train_batch_size=4,
                num_train_epochs=1,
                save_steps=500,
                logging_dir=f"./logs_{i}",
                logging_steps=100,
                overwrite_output_dir=True,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
            )
            trainer.train()
            print(f"Training on dataset {i} completed.")
        except Exception as e:
            print(f"Error processing dataset {i}: {e}")

def hitung_bmi(tinggi, berat):
    tinggi_m = tinggi / 100
    bmi = berat / (tinggi_m ** 2)

    if bmi < 18.5:
        kategori = "kekurangan berat badan"
    elif 18.5 <= bmi < 24.9:
        kategori = "berat badan normal"
    elif 25 <= bmi < 29.9:
        kategori = "kelebihan berat badan"
    else:
        kategori = "obesitas"

    return bmi, kategori

def buat_prompt(tinggi, berat, bmi, kategori):
    prompt = (
        f"Seorang klien dengan tinggi badan {tinggi} cm dan berat badan {berat} kg memiliki BMI {bmi:.2f}, "
        f"yang termasuk dalam kategori {kategori}. Rancang jadwal olahraga rinci selama 6 hari dalam seminggu "
        f"untuk membantu klien mencapai berat badan yang sehat. Setiap hari harus memiliki informasi berikut: "
        f"- Jenis olahraga\n- Durasi latihan (dalam menit)\n- Intensitas latihan (rendah/sedang/tinggi).\n"
    )
    return prompt

def rekomendasi_olahraga(model, tokenizer, prompt, panjang_maks=500):
    """
    Generate a 6-day workout plan based on the user's prompt with improved generation parameters.
    """
    try:
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

        # Generate output
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=panjang_maks,
            num_beams=5,
            temperature=0.7,  # Allowing for creative outputs
            top_p=0.9,        # Top-p sampling for diversity
            no_repeat_ngram_size=2,
            early_stopping=True,
        )

        # Decode the generated text
        rekomendasi = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\nDEBUG: Generated Output (Raw):", rekomendasi)
        return rekomendasi

    except Exception as e:
        print(f"Error generating recommendation: {e}")
        return "Error in generating workout plan."

# Example prompt with examples for better guidance
prompt = (
    "Buatkan jadwal olahraga 6 hari untuk kategori berat badan normal. Contoh:\n"
    "- Hari 1: Lari ringan (30 menit, intensitas sedang)\n"
    "- Hari 2: Yoga (45 menit, intensitas rendah)\n"
    "- Hari 3: Latihan kekuatan (30 menit, intensitas sedang)\n"
    "Buat jadwal seperti ini untuk 6 hari berturut-turut:"
)

if __name__ == "__main__":
    MODEL_NAME = "EleutherAI/gpt-neo-125M"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Pastikan ada pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    datasets = ["Dataset/megaGymDataset.csv", "Dataset/exercise_dataset.csv", "Dataset/Top 50 Excerice for your body.csv"]

    print("Starting training...")
    train_model(model, tokenizer, datasets)

    tinggi = float(input("Masukkan tinggi badan Anda dalam cm: "))
    berat = float(input("Masukkan berat badan Anda dalam kg: "))

    bmi, kategori = hitung_bmi(tinggi, berat)
    print(f"BMI Anda adalah {bmi:.2f}, yang termasuk dalam kategori {kategori}.")

    # Buat prompt baru
    prompt = buat_prompt(tinggi, berat, bmi, kategori)
    print("\nPrompt yang digunakan:")
    print(prompt)

    # Generate recommendation
    rekomendasi = rekomendasi_olahraga(model, tokenizer, prompt)
    print("\nRencana Olahraga yang Direkomendasikan:")
    print(rekomendasi)