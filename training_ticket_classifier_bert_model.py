from datasets import load_dataset
from transformers import AutoTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
import pandas as pd
import torch
from datasets import Dataset

def tokenize_function(text):
    return tokenizer(text["corpo"], padding="max_length", truncation=True, max_length=256)

# Test GPU availability
print(f"CUDA Available: {torch.cuda.is_available()}\n")
print(f"GPU: {torch.cuda.get_device_name(0)}\n")

# Load BERTugues
model_name = "ricardoz/BERTugues-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=10)

print("\nBERT model loaded successfully!\n")

# Load the  dataset
df = pd.read_csv('datasets/dataset-tickets.csv')

# Mapping string fila to integers
label_mapping = {
    "Suporte Técnico": 0,
    "Devoluções e Trocas": 1 ,
    "Faturamento e Pagamentos": 2 ,
    "Vendas e pré-vendas": 3 ,
    "Interrupções de serviço e manutenção": 4 ,
    "Suporte ao produto": 5 ,
    "Suporte de TI": 6 ,
    "Atendimento ao Cliente": 7 ,
    "Recursos Humanos": 8 ,
    "Consulta Geral": 9
}
df['label'] = df['fila'].map(label_mapping)
#Check null and Unique vvalues
print(f"\nNull Values: {df['label'].isnull().sum()}")
print(f"Unique Labels: {sorted(df['label'].unique())}\n")

dataset = Dataset.from_pandas(df)

# Split in train and test 
split_dataset = dataset.train_test_split(test_size=0.3)

# Apply the same preprocessing function
tokenized_dataset = split_dataset.map(tokenize_function, batched=True)

#See how the tokenized test is
#print(tokenized_dataset['train'][0])
#print(tokenized_dataset['train'][0]["input_ids"][:10])  # Print first 10 tokens
#print(tokenized_dataset['train'][0]["attention_mask"][:10])

# Define training arguments
training_args = TrainingArguments(
    output_dir="./bert_finetuned_classification",          # Directory to save model checkpoints
    eval_strategy="steps",            # Evaluate every 'n' steps
    eval_steps=500,                         # Evaluation frequency
    per_device_train_batch_size=8,          # Training batch size
    per_device_eval_batch_size=8,           # Evaluation batch size
    num_train_epochs=5,                     # Number of training epochs
    learning_rate=2e-5,                     # Standard learning rate for fine-tuning BERT
    weight_decay=0.01,                      # Regularization to prevent overfitting
    logging_steps=100,                      # Log metrics every 'n' steps
    save_steps=500,                         # Save checkpoints every 'n' steps
    load_best_model_at_end=True,            # Load the best model based on evaluation metrics
    fp16=True                               # Enable mixed precision for faster training
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],  # Training dataset
    eval_dataset=tokenized_dataset["test"],    # Validation dataset
)

print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning complete!")

results = trainer.evaluate()
print("\nEvaluation Results:", results)

# Save the fine-tuned model and tokenizer
model.save_pretrained("./bert_finetuned_classification")
tokenizer.save_pretrained("./bert_finetuned_classification")

print("Model and tokenizer saved successfully!")

# sample_text = """Você poderia fornecer mais informações sobre serviços de análise de dados para otimização de investimentos? Sua assistência oportuna é importante para nós.
# """
# inputs = tokenizer(sample_text, return_tensors="pt").to("cuda")
# outputs = model(**inputs)

# # Get prediction
# prediction = outputs.logits.argmax(dim=1).item()
# print("Prediction:", prediction)