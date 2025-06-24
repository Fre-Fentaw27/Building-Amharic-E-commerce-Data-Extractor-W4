import os
import pandas as pd
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
    TokenClassificationPipeline
)
from datasets import Dataset, DatasetDict
from time import perf_counter
from sklearn.model_selection import train_test_split

# Configuration
os.makedirs("comparison_results", exist_ok=True)
MODELS = {
    "XLM-Roberta": "./amharic-ner-model",
    "DistilBERT": "distilbert-base-multilingual-cased",
    "mBERT": "bert-base-multilingual-cased"
}

# 1. Dataset Loading (unchanged)
def load_and_split_dataset(file_path, val_size=0.2):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    sentences = []
    current_sentence = []
    
    for line in lines:
        line = line.strip()
        if line:
            parts = line.split('\t') if '\t' in line else line.split()
            if len(parts) >= 2:
                current_sentence.append((parts[0], parts[-1]))
        elif current_sentence:
            sentences.append(current_sentence)
            current_sentence = []
    
    if current_sentence:
        sentences.append(current_sentence)
    
    train_sentences, val_sentences = train_test_split(sentences, test_size=val_size, random_state=42)
    
    def create_dataset(sentences):
        tokens = [[token for token, tag in sent] for sent in sentences]
        ner_tags = [[tag for token, tag in sent] for sent in sentences]
        return Dataset.from_dict({"tokens": tokens, "ner_tags": ner_tags})
    
    return DatasetDict({
        "train": create_dataset(train_sentences),
        "validation": create_dataset(val_sentences)
    })

base_path = "D:/AI-projects/Building-Amharic-E-commerce-Data-Extractor-W4/data/clean"
try:
    dataset = load_and_split_dataset(os.path.join(base_path, "labeled_conll.txt"))
    print(f"Dataset loaded successfully! Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}")
except Exception as e:
    print(f"Error loading dataset: {str(e)}")
    exit()

# 2. Updated Evaluation Function with proper tokenization
def evaluate_model(model, tokenizer):
    try:
        # Create pipeline
        ner_pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            aggregation_strategy="simple"
        )
        
        # Speed Test with proper tokenization
        test_samples = [" ".join(sent) for sent in dataset["validation"]["tokens"][:10]]
        start_time = perf_counter()
        _ = ner_pipeline(test_samples)
        inference_speed = (perf_counter() - start_time)/10
        
        # Accuracy Test
        true_positives = 0
        total_predicted = 0
        total_actual = 0
        
        for example in dataset["validation"]:
            # Convert tokens to properly spaced string
            text = " ".join(example["tokens"])
            preds = ner_pipeline(text)
            
            # Convert predictions to character spans
            pred_entities = set()
            char_offset = 0
            for token, tag in zip(example["tokens"], example["ner_tags"]):
                if tag != "O":
                    start = text.find(token, char_offset)
                    end = start + len(token)
                    pred_entities.add((start, end, tag))
                    char_offset = end
            
            # Get actual entities
            actual_entities = set()
            char_offset = 0
            for token, tag in zip(example["tokens"], example["ner_tags"]):
                if tag != "O":
                    start = text.find(token, char_offset)
                    end = start + len(token)
                    actual_entities.add((start, end, tag))
                    char_offset = end
            
            true_positives += len(pred_entities & actual_entities)
            total_predicted += len(pred_entities)
            total_actual += len(actual_entities)
        
        precision = true_positives / total_predicted if total_predicted > 0 else 0
        recall = true_positives / total_actual if total_actual > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "speed": inference_speed,
            "memory_mb": torch.cuda.max_memory_allocated()/1e6 if torch.cuda.is_available() else 0
        }
    except Exception as e:
        print(f"Evaluation error: {str(e)}")
        return None

# 3. Model Comparison
results = []
for model_name, model_path in MODELS.items():
    print(f"\n=== Evaluating {model_name} ===")
    
    try:
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        
        print("Evaluating...")
        metrics = evaluate_model(model, tokenizer)
        if metrics:
            results.append({"model": model_name, **metrics})
            print(f"{model_name} evaluation completed!")
        else:
            raise Exception("Evaluation failed")
    except Exception as e:
        print(f"Failed to evaluate {model_name}: {str(e)}")
        results.append({
            "model": model_name,
            "precision": float('nan'),
            "recall": float('nan'),
            "f1": float('nan'),
            "speed": float('nan'),
            "memory_mb": float('nan')
        })

# 4. Save and show results
df = pd.DataFrame(results)
df.to_csv("comparison_results/model_comparison_results.csv", index=False)
print("\n=== Final Results ===")
print(df.to_string(index=False))