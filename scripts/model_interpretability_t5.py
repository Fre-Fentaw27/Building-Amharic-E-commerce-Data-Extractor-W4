import os
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
    TokenClassificationPipeline
)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Configuration - Update these paths as needed
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "amharic-ner-model"
DATA_PATH = BASE_DIR / "data" / "clean" / "labeled_conll.txt"
RESULTS_DIR = BASE_DIR / "interpretability_results"
SAMPLE_SIZE = 2  # Reduced for faster execution

# Create results directory
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import shap
    from lime.lime_text import LimeTextExplainer
except ImportError as e:
    print(f"Error: {e}")
    print("\nRequired packages not found. Please install with:")
    print("pip install shap lime matplotlib")
    sys.exit(1)

class CustomNERPipeline(TokenClassificationPipeline):
    """Custom pipeline to handle Amharic text better"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entity_mapping = {
            'PER': 'Person',
            'LOC': 'Location',
            'ORG': 'Organization',
            'PROD': 'Product'
        }

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "padding" in kwargs:
            preprocess_kwargs["padding"] = kwargs["padding"]
        if "truncation" in kwargs:
            preprocess_kwargs["truncation"] = kwargs["truncation"]
        return preprocess_kwargs, {}, {}

def load_model():
    """Load the NER model with error handling"""
    try:
        print(f"Loading model from {MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
        
        return CustomNERPipeline(
            task="ner",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            aggregation_strategy="average"
        )
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        sys.exit(1)

def load_samples():
    """Load and validate samples with robust error handling"""
    print(f"Looking for data at: {DATA_PATH}")
    
    if not DATA_PATH.exists():
        print("\nERROR: Data file not found!")
        print("Please ensure:")
        print(f"1. The file exists at: {DATA_PATH}")
        print(f"2. Your current working directory is: {os.getcwd()}")
        sys.exit(1)

    try:
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        samples = []
        current_sample = []
        
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split('\t') if '\t' in line else line.split()
                if len(parts) >= 2:
                    current_sample.append(parts[0])
            elif current_sample:
                samples.append(" ".join(current_sample))
                current_sample = []
                if len(samples) >= SAMPLE_SIZE:
                    break
        
        if not samples:
            print("ERROR: No valid samples found in the file")
            sys.exit(1)
            
        return samples
    except Exception as e:
        print(f"Error loading samples: {str(e)}")
        sys.exit(1)

def generate_shap_explanations(ner_pipeline, samples):
    """Generate SHAP explanations with fallback"""
    print("\nGenerating SHAP explanations (this may take several minutes)...")
    
    try:
        # Wrap the pipeline for SHAP compatibility
        def predict_fn(texts):
            return [ner_pipeline(text, padding=True, truncation=True) for text in texts]
        
        explainer = shap.Explainer(
            predict_fn,
            ner_pipeline.tokenizer,
            output_names=list(ner_pipeline.model.config.id2label.values())
        )
        
        # Use just 3 samples for SHAP (computationally intensive)
        shap_values = explainer(samples[:3])
        
        # Visualize for the first non-O entity
        entity_type = [x for x in ner_pipeline.model.config.id2label.values() if x != "O"][0]
        plt.figure(figsize=(12, 6))
        shap.plots.text(shap_values[:, :, entity_type], display=False)
        plt.title(f"SHAP Explanation for {entity_type} Entities")
        plt.savefig(RESULTS_DIR / "shap_explanations.png", bbox_inches='tight')
        plt.close()
        
        return True
    except Exception as e:
        print(f"SHAP failed: {str(e)}")
        print("Tip: Try reducing SAMPLE_SIZE or using simpler explainers")
        return False

def generate_lime_explanations(ner_pipeline, samples):
    """Generate LIME explanations with fallback"""
    print("\nGenerating LIME explanations...")
    
    try:
        explainer = LimeTextExplainer(
            class_names=list(ner_pipeline.model.config.id2label.values()),
            split_expression=lambda x: x.split(),
            bow=False
        )
        
        lime_results = []
        
        for i, text in enumerate(samples[:5]):  # Just 5 samples for demo
            def predict_proba(texts):
                results = []
                for t in texts:
                    preds = ner_pipeline(t, padding=True, truncation=True)
                    pred_scores = {label: 0 for label in ner_pipeline.model.config.id2label.values()}
                    for p in preds:
                        pred_scores[p['entity_group']] = p['score']
                    results.append([pred_scores.get(label, 0) for label in ner_pipeline.model.config.id2label.values()])
                return np.array(results)
            
            exp = explainer.explain_instance(
                text, 
                predict_proba,
                num_features=5,
                labels=[1]  # Explain first non-O label
            )
            
            lime_results.append({
                "text": text,
                "explanation": exp.as_list(label=1)
            })
            
            fig = exp.as_pyplot_figure(label=1)
            plt.title(f"LIME Explanation - Sample {i+1}")
            plt.savefig(RESULTS_DIR / f"lime_sample_{i+1}.png", bbox_inches='tight')
            plt.close()
        
        pd.DataFrame(lime_results).to_csv(RESULTS_DIR / "lime_results.csv", index=False)
        return True
    except Exception as e:
        print(f"LIME failed: {str(e)}")
        return False

def analyze_difficult_cases(ner_pipeline):
    """Identify and analyze problematic cases"""
    print("\nAnalyzing difficult cases...")
    
    try:
        difficult_cases = []
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        current_text = []
        current_labels = []
        
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split('\t') if '\t' in line else line.split()
                if len(parts) >= 2:
                    current_text.append(parts[0])
                    current_labels.append(parts[-1])
            elif current_text:
                text = " ".join(current_text)
                preds = ner_pipeline(text, padding=True, truncation=True)
                
                # Convert to character spans
                pred_entities = set()
                char_pos = 0
                for token, tag in zip(current_text, current_labels):
                    if tag != "O":
                        start = text.find(token, char_pos)
                        end = start + len(token)
                        pred_entities.add((start, end, tag))
                    char_pos += len(token) + 1
                
                actual_entities = set()
                for p in preds:
                    actual_entities.add((p['start'], p['end'], p['entity_group']))
                
                if pred_entities != actual_entities:
                    difficult_cases.append({
                        "text": text,
                        "expected": list(pred_entities),
                        "predicted": list(actual_entities),
                        "discrepancy": list(pred_entities.symmetric_difference(actual_entities))
                    })
                
                current_text = []
                current_labels = []
                
                if len(difficult_cases) >= 3:  # Limit to 3 cases
                    break
        
        if difficult_cases:
            pd.DataFrame(difficult_cases).to_csv(RESULTS_DIR / "difficult_cases.csv", index=False)
            
            with open(RESULTS_DIR / "interpretability_report.md", "w") as f:
                f.write("# Model Interpretability Report\n\n")
                f.write("## Problematic Cases\n")
                for case in difficult_cases:
                    f.write(f"\n### Text: {case['text']}\n")
                    f.write(f"- Expected: {case['expected']}\n")
                    f.write(f"- Predicted: {case['predicted']}\n")
                    f.write(f"- Discrepancy: {case['discrepancy']}\n")
                
                f.write("\n## Recommendations\n")
                f.write("- Add more training examples for ambiguous patterns\n")
                f.write("- Implement spell-checking for location names\n")
                f.write("- Add post-processing rules for common errors\n")
            
            return True
        else:
            print("No difficult cases found in the sample")
            return False
    except Exception as e:
        print(f"Case analysis failed: {str(e)}")
        return False

def main():
    print("Starting model interpretability analysis...")
    
    # Load model and samples
    ner_pipeline = load_model()
    samples = load_samples()
    
    if not samples:
        return
    
    print(f"\nLoaded {len(samples)} samples for analysis")
    
    # Run analyses
    shap_success = generate_shap_explanations(ner_pipeline, samples)
    lime_success = generate_lime_explanations(ner_pipeline, samples)
    cases_success = analyze_difficult_cases(ner_pipeline)
    
    # Generate summary
    with open(RESULTS_DIR / "summary.txt", "w") as f:
        f.write("Interpretability Analysis Summary\n")
        f.write("===============================\n\n")
        f.write(f"SHAP Explanations Generated: {'Yes' if shap_success else 'No'}\n")
        f.write(f"LIME Explanations Generated: {'Yes' if lime_success else 'No'}\n")
        f.write(f"Difficult Cases Analyzed: {'Yes' if cases_success else 'No'}\n")
    
    print("\nAnalysis complete! Results saved to:")
    print(f"- SHAP: {RESULTS_DIR}/shap_explanations.png")
    print(f"- LIME: {RESULTS_DIR}/lime_sample_*.png")
    print(f"- Cases: {RESULTS_DIR}/difficult_cases.csv")
    print(f"- Full report: {RESULTS_DIR}/interpretability_report.md")

if __name__ == "__main__":
    main()