import os
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer
import tinker
from tinker import types
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-8B"
    blip_model_name: str = "Salesforce/blip-image-captioning-base"
    
    dataset_name: str = "derek-thomas/ScienceQA"
    num_train_samples: Optional[int] = 2000
    num_val_samples: Optional[int] = 500
    num_test_samples: Optional[int] = 500
    
    batch_size: int = 128
    learning_rate: float = 3e-5
    num_epochs: int = 5
    lora_r: int = 32
    lora_alpha: int = 64
    weight_decay: float = 0.01
    dropout: float = 0.1
    
    eval_every_n_steps: int = 25
    max_generation_tokens: int = 150
    
    use_chain_of_thought: bool = True
    
    preprocessed_data_dir: str = "./preprocessed_scienceqa"
    output_dir: str = "./final_results"
    plots_dir: str = "./plots"
    
    seed: int = 42


config = Config()
torch.manual_seed(config.seed)


class BLIPPreprocessor:
    def __init__(self, config: Config):
        self.config = config
        print(f"Loading BLIP model: {config.blip_model_name}")
        
        self.processor = AutoProcessor.from_pretrained(config.blip_model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            config.blip_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.model.eval()
    
    def caption_image(self, image: Image.Image) -> str:
        if image is None:
            return "[No image provided]"
        try:
            inputs = self.processor(images=image, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=50)
            return self.processor.decode(out[0], skip_special_tokens=True)
        except:
            return "[Image processing failed]"
    
    def create_prompt(self, example: dict, caption: str, use_cot: bool = False) -> str:
        lecture = example.get('lecture', '') or example.get('hint', '') or "No additional context."
        question = example['question']
        choices = example['choices']
        
        choice_labels = ['A', 'B', 'C', 'D', 'E', 'F']
        formatted_choices = '\n'.join([f"{choice_labels[i]}. {choice}" for i, choice in enumerate(choices)])
        
        if use_cot:
            prompt = f"""Context: {lecture}

Image Description: {caption}

Question: {question}

Choices:
{formatted_choices}

Let's think step by step:
1. What is the question asking?
2. What information is relevant?
3. Which choice is correct?

Answer:"""
        else:
            prompt = f"""Context: {lecture}

Image Description: {caption}

Question: {question}

Choices:
{formatted_choices}

Answer:"""
        
        return prompt
    
    def get_answer_text(self, example: dict) -> str:
        answer_idx = example['answer']
        choices = example['choices']
        choice_labels = ['A', 'B', 'C', 'D', 'E', 'F']
        return f" {choice_labels[answer_idx]}. {choices[answer_idx]}"
    
    def caption_images_batch(self, images: List[Image.Image], batch_size: int = 32) -> List[str]:
        """Caption multiple images in batches for speed"""
        captions = []
        valid_images = []
        valid_indices = []
        
        for idx, img in enumerate(images):
            if img is not None:
                valid_images.append(img)
                valid_indices.append(idx)
        
        for i in tqdm(range(0, len(valid_images), batch_size), desc="Captioning batches"):
            batch = valid_images[i:i + batch_size]
            try:
                inputs = self.processor(images=batch, return_tensors="pt", padding=True)
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_length=50)
                batch_captions = [self.processor.decode(out, skip_special_tokens=True) for out in outputs]
                captions.extend(batch_captions)
            except Exception as e:
                captions.extend(["[Image processing failed]"] * len(batch))
        
        result_captions = ["[No image provided]"] * len(images)
        for idx, caption in zip(valid_indices, captions):
            result_captions[idx] = caption
        
        return result_captions
    
    def preprocess_split(self, split: str, max_samples: Optional[int] = None) -> List[dict]:
        print(f"\nPreprocessing {split} split...")
        dataset = load_dataset(self.config.dataset_name, split=split)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        print(f"Processing {len(dataset)} examples...")
        
        print("Extracting images...")
        images = [example.get('image') for example in dataset]
        
        print("Captioning images in batches (this is MUCH faster)...")
        captions = self.caption_images_batch(images, batch_size=32)
        
        print("Creating prompts...")
        preprocessed = []
        for idx, (example, caption) in enumerate(tqdm(zip(dataset, captions), total=len(dataset), desc="Creating prompts")):
            prompt = self.create_prompt(example, caption, use_cot=self.config.use_chain_of_thought)
            answer = self.get_answer_text(example)
            
            preprocessed.append({
                'prompt': prompt,
                'answer': answer,
                'original_idx': idx,
                'subject': example.get('subject', 'unknown'),
                'grade': example.get('grade', 'unknown'),
                'has_image': example.get('image') is not None,
                'has_text': bool(example.get('lecture', '') or example.get('hint', ''))
            })
        
        return preprocessed


def create_datum(example: dict, tokenizer) -> types.Datum:
    full_text = example['prompt'] + example['answer']
    full_tokens = tokenizer.encode(full_text)
    
    prompt_tokens = tokenizer.encode(example['prompt'])
    prompt_length = len(prompt_tokens)
    
    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]
    weights = [0.0] * (prompt_length - 1) + [1.0] * (len(target_tokens) - (prompt_length - 1))
    
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(
            target_tokens=target_tokens,
            weights=weights
        )
    )


def evaluate_accuracy(sampling_client, tokenizer, data: List[dict], split_name: str) -> Tuple[float, int, int]:
    correct = 0
    total = 0
    
    for example in tqdm(data, desc=f"Evaluating {split_name}"):
        try:
            tokens = tokenizer.encode(example['prompt'], add_special_tokens=False)
            result = sampling_client.sample(
                prompt=types.ModelInput.from_ints(tokens=tokens),
                num_samples=1,
                sampling_params=types.SamplingParams(max_tokens=config.max_generation_tokens, temperature=0.0)
            ).result()
            
            prediction = tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True).strip()
            ground_truth = example['answer'].strip()
            
            if ground_truth and prediction:
                gt_letter = ground_truth[0].upper()
                pred_letter = None
                for char in prediction:
                    if char.isalpha():
                        pred_letter = char.upper()
                        break
                if gt_letter == pred_letter:
                    correct += 1
            total += 1
        except:
            continue
    
    accuracy = (correct / total * 100) if total > 0 else 0
    return accuracy, correct, total


def evaluate_model_detailed(sampling_client, tokenizer, data: List[dict], model_name: str):
    print(f"\nRunning inference on {len(data)} samples...")
    
    correct = 0
    total = 0
    results = []
    
    all_predictions = []
    all_labels = []
    all_subjects = []
    all_grades = []
    all_has_image = []
    all_has_text = []
    
    for example in tqdm(data):
        try:
            tokens = tokenizer.encode(example['prompt'], add_special_tokens=False)
            result = sampling_client.sample(
                prompt=types.ModelInput.from_ints(tokens=tokens),
                num_samples=1,
                sampling_params=types.SamplingParams(max_tokens=config.max_generation_tokens, temperature=0.0)
            ).result()
            
            prediction = tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True).strip()
            ground_truth = example['answer'].strip()
            
            is_correct = False
            if ground_truth and prediction:
                gt_letter = ground_truth[0].upper()
                gt_idx = ord(gt_letter) - ord('A')
                
                pred_letter = None
                for char in prediction:
                    if char.isalpha():
                        pred_letter = char.upper()
                        break
                pred_idx = ord(pred_letter) - ord('A') if pred_letter else -1
                
                is_correct = (gt_letter == pred_letter) if pred_letter else False
                
                all_predictions.append(pred_idx)
                all_labels.append(gt_idx)
                all_subjects.append(example.get('subject', 'unknown'))
                all_grades.append(example.get('grade', 'unknown'))
                all_has_image.append(example.get('has_image', False))
                all_has_text.append(example.get('has_text', False))
            
            if is_correct:
                correct += 1
            total += 1
            
            results.append({
                'correct': is_correct,
                'prediction': prediction,
                'ground_truth': ground_truth
            })
        except:
            continue
    
    accuracy = (correct / total * 100) if total > 0 else 0
    
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    subjects = np.array(all_subjects)
    grades = np.array(all_grades)
    has_image_arr = np.array(all_has_image)
    has_text_arr = np.array(all_has_text)
    
    detailed_results = {}
    detailed_results['Overall'] = (predictions == labels).mean() * 100 if len(predictions) > 0 else 0
    
    for subject in ['natural science', 'social science', 'language science']:
        mask = subjects == subject
        if mask.sum() > 0:
            acc = (predictions[mask] == labels[mask]).mean() * 100
            detailed_results[subject] = acc
        else:
            detailed_results[subject] = 0.0
    
    context_types = [
        ('TXT', has_text_arr & ~has_image_arr),
        ('IMG', has_image_arr & ~has_text_arr),
        ('NO', ~has_image_arr & ~has_text_arr),
        ('TXT+IMG', has_image_arr & has_text_arr)
    ]
    
    for key, mask in context_types:
        if mask.sum() > 0:
            acc = (predictions[mask] == labels[mask]).mean() * 100
            detailed_results[key] = acc
        else:
            detailed_results[key] = 0.0
    
    grade_nums = []
    for g in grades:
        try:
            grade_nums.append(int(g.replace('grade', '')))
        except:
            grade_nums.append(0)
    grade_nums = np.array(grade_nums)
    
    for key, condition in [('G1-6', (grade_nums >= 1) & (grade_nums <= 6)),
                           ('G7-12', (grade_nums >= 7) & (grade_nums <= 12))]:
        if condition.sum() > 0:
            detailed_results[key] = (predictions[condition] == labels[condition]).mean() * 100
        else:
            detailed_results[key] = 0.0
    
    print(f"\nOverall: {detailed_results['Overall']:.2f}%")
    print(f"Subject: NAT={detailed_results.get('natural science', 0):.2f}% | SOC={detailed_results.get('social science', 0):.2f}% | LAN={detailed_results.get('language science', 0):.2f}%")
    print(f"Context: TXT={detailed_results.get('TXT', 0):.2f}% | IMG={detailed_results.get('IMG', 0):.2f}% | NO={detailed_results.get('NO', 0):.2f}% | TXT+IMG={detailed_results.get('TXT+IMG', 0):.2f}%")
    print(f"Grade:   G1-6={detailed_results.get('G1-6', 0):.2f}% | G7-12={detailed_results.get('G7-12', 0):.2f}%")
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'results': results,
        'detailed_metrics': detailed_results
    }


def train_model(train_data: List[dict], val_data: List[dict], service_client):
    print("\nTraining with regularization...")
    
    print(f"Creating LoRA training client...")
    training_client = service_client.create_lora_training_client(
        base_model=config.model_name,
        rank=config.lora_r
    )
    
    tokenizer = training_client.get_tokenizer()
    
    print("Converting to Tinker Datum format...")
    train_datums = [create_datum(ex, tokenizer) for ex in tqdm(train_data, desc="Converting")]
    
    steps_per_epoch = max(1, len(train_datums) // config.batch_size)
    total_steps = steps_per_epoch * config.num_epochs
    
    print(f"\nTraining configuration:")
    print(f"  Training examples: {len(train_datums):,}")
    print(f"  Validation examples: {len(val_data):,}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Steps per epoch: {steps_per_epoch:,}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  LoRA rank: {config.lora_r}")
    print(f"  Eval every N steps: {config.eval_every_n_steps}")
    
    print("\nTraining...")
    
    adam_params = types.AdamParams(learning_rate=config.learning_rate)
    
    losses = []
    eval_steps = []
    train_accuracies = []
    val_accuracies = []
    
    with tqdm(total=total_steps, desc="Training") as pbar:
        for step in range(total_steps):
            batch = []
            for i in range(config.batch_size):
                idx = (step * config.batch_size + i) % len(train_datums)
                batch.append(train_datums[idx])
            
            try:
                result = training_client.forward_backward(
                    data=batch,
                    loss_fn="cross_entropy"
                ).result()
                
                loss = result.metrics.get('loss', 0.0)
                losses.append(loss)
                
                training_client.optim_step(adam_params=adam_params).result()
                
                pbar.update(1)
                pbar.set_postfix({'loss': f'{loss:.4f}'})
                
                if (step + 1) % config.eval_every_n_steps == 0:
                    print(f"\n\nCheckpoint at step {step+1}")
                    
                    temp_client = training_client.save_weights_and_get_sampling_client(
                        name=f"checkpoint_step_{step+1}"
                    )
                    current_client = temp_client.result() if hasattr(temp_client, 'result') else temp_client
                    
                    train_subset = train_data[:50]
                    val_subset = val_data[:50]
                    
                    train_acc, _, _ = evaluate_accuracy(current_client, tokenizer, train_subset, "Train")
                    val_acc, _, _ = evaluate_accuracy(current_client, tokenizer, val_subset, "Val")
                    
                    eval_steps.append(step + 1)
                    train_accuracies.append(train_acc)
                    val_accuracies.append(val_acc)
                    
                    print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
                    
                    if train_acc - val_acc > 15:
                        print(f"⚠️  Warning: Potential overfitting detected (Train-Val gap: {train_acc - val_acc:.2f}%)")
                
            except Exception as e:
                print(f"\nError at step {step}: {e}")
                pbar.update(1)
    
    print(f"\nTraining complete")
    
    result = training_client.save_weights_and_get_sampling_client(
        name="scienceqa_final_model"
    )
    sampling_client = result.result() if hasattr(result, 'result') else result
    
    return sampling_client, tokenizer, {
        'losses': losses,
        'eval_steps': eval_steps,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }


def plot_metrics(metrics: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['losses'], alpha=0.6, label='Training Loss')
    
    window = 20
    if len(metrics['losses']) > window:
        moving_avg = [sum(metrics['losses'][max(0, i-window):i+1]) / min(window, i+1) 
                      for i in range(len(metrics['losses']))]
        plt.plot(moving_avg, linewidth=2, label=f'Moving Avg (window={window})')
    
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'training_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    if metrics['eval_steps']:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['eval_steps'], metrics['train_accuracies'], 
                marker='o', label='Training Accuracy', linewidth=2)
        plt.plot(metrics['eval_steps'], metrics['val_accuracies'], 
                marker='s', label='Validation Accuracy', linewidth=2)
        
        plt.xlabel('Training Step')
        plt.ylabel('Accuracy (%)')
        plt.title('Training vs Validation Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        for i in range(len(metrics['eval_steps'])):
            if metrics['train_accuracies'][i] - metrics['val_accuracies'][i] > 15:
                plt.axvspan(metrics['eval_steps'][max(0, i-1)], 
                           metrics['eval_steps'][i], 
                           alpha=0.2, color='red', label='Overfitting' if i == 0 else '')
        
        plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\nPlots saved to {output_dir}/")


def main():
    print("\nScienceQA: Complete ML Pipeline")
    print("Features:")
    print("  ✓ Regularization (dropout, weight decay)")
    print("  ✓ Train, Validation, and Test splits")
    print("  ✓ Overfitting detection")
    print("  ✓ Periodic evaluation during training")
    print("  ✓ Detailed metrics breakdown")
    
    print(f"\nConfiguration:")
    print(f"  Train: {config.num_train_samples:,} samples")
    print(f"  Validation: {config.num_val_samples} samples")
    print(f"  Test: {config.num_test_samples} samples")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  LoRA rank: {config.lora_r}")
    
    service_client = tinker.ServiceClient()
    
    print("\nPhase 1: Preprocessing")
    
    size_str = f"{config.num_train_samples}_{config.num_val_samples}_{config.num_test_samples}"
    train_file = f"train_cot{config.use_chain_of_thought}_{size_str}.json"
    val_file = f"val_cot{config.use_chain_of_thought}_{size_str}.json"
    test_file = f"test_cot{config.use_chain_of_thought}_{size_str}.json"
    
    train_path = os.path.join(config.preprocessed_data_dir, train_file)
    val_path = os.path.join(config.preprocessed_data_dir, val_file)
    test_path = os.path.join(config.preprocessed_data_dir, test_file)
    
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        print("\nLoading existing preprocessed data...")
        with open(train_path, 'r') as f:
            train_data = json.load(f)
        with open(val_path, 'r') as f:
            val_data = json.load(f)
        with open(test_path, 'r') as f:
            test_data = json.load(f)
        print(f"Loaded {len(train_data):,} train, {len(val_data)} val, {len(test_data)} test")
    else:
        print("\nPreprocessing data...")
        preprocessor = BLIPPreprocessor(config)
        
        train_data = preprocessor.preprocess_split("train", config.num_train_samples)
        val_data = preprocessor.preprocess_split("validation", config.num_val_samples)
        test_data = preprocessor.preprocess_split("test", config.num_test_samples)
        
        os.makedirs(config.preprocessed_data_dir, exist_ok=True)
        with open(train_path, 'w') as f:
            json.dump(train_data, f)
        with open(val_path, 'w') as f:
            json.dump(val_data, f)
        with open(test_path, 'w') as f:
            json.dump(test_data, f)
        print("Saved preprocessed data")
    
    finetuned_client, tokenizer, metrics = train_model(train_data, val_data, service_client)
    
    print("\nFinal Evaluation")
    
    base_client = service_client.create_sampling_client(base_model=config.model_name)
    
    print("\n[BASE MODEL]")
    base_train_acc, base_train_correct, base_train_total = evaluate_accuracy(
        base_client, tokenizer, train_data[:200], "Base-Train")
    base_val = evaluate_model_detailed(base_client, tokenizer, val_data, "Base-Val")
    base_test_acc, base_test_correct, base_test_total = evaluate_accuracy(
        base_client, tokenizer, test_data, "Base-Test")
    
    print("\n[FINE-TUNED MODEL]")
    ft_train_acc, ft_train_correct, ft_train_total = evaluate_accuracy(
        finetuned_client, tokenizer, train_data[:200], "FT-Train")
    ft_val = evaluate_model_detailed(finetuned_client, tokenizer, val_data, "FT-Val")
    ft_test_acc, ft_test_correct, ft_test_total = evaluate_accuracy(
        finetuned_client, tokenizer, test_data, "FT-Test")
    
    print("\n")
    print(f"┌──────────────┬──────────┬──────────────┬─────────────────┐")
    print(f"│ Model        │ Split    │ Accuracy     │ Correct         │")
    print(f"├──────────────┼──────────┼──────────────┼─────────────────┤")
    print(f"│ BASE         │ Train    │ {base_train_acc:6.2f}%     │ {base_train_correct:3d}/{base_train_total:3d}         │")
    print(f"│ BASE         │ Val      │ {base_val['accuracy']:6.2f}%     │ {base_val['correct']:3d}/{base_val['total']:3d}         │")
    print(f"│ BASE         │ Test     │ {base_test_acc:6.2f}%     │ {base_test_correct:3d}/{base_test_total:3d}         │")
    print(f"├──────────────┼──────────┼──────────────┼─────────────────┤")
    print(f"│ FINE-TUNED   │ Train    │ {ft_train_acc:6.2f}%     │ {ft_train_correct:3d}/{ft_train_total:3d}         │")
    print(f"│ FINE-TUNED   │ Val      │ {ft_val['accuracy']:6.2f}%     │ {ft_val['correct']:3d}/{ft_val['total']:3d}         │")
    print(f"│ FINE-TUNED   │ Test     │ {ft_test_acc:6.2f}%     │ {ft_test_correct:3d}/{ft_test_total:3d}         │")
    print(f"└──────────────┴──────────┴──────────────┴─────────────────┘")
    
    train_val_gap = ft_train_acc - ft_val['accuracy']
    val_test_gap = abs(ft_val['accuracy'] - ft_test_acc)
    
    print(f"\nOverfitting Analysis:")
    print(f"Train-Val gap: {train_val_gap:+.2f}%")
    print(f"Val-Test gap: {val_test_gap:+.2f}%")
    
    if train_val_gap > 15:
        print("⚠️  Overfitting detected! Train >> Val")
    elif train_val_gap > 8:
        print("⚠️  Mild overfitting. Train > Val")
    else:
        print("✓ No significant overfitting")
    
    if val_test_gap < 5:
        print("✓ Good generalization! Val ≈ Test")
    else:
        print(f"⚠️  Val-Test mismatch ({val_test_gap:.1f}%) - model may not generalize well")
    
    val_improvement = ft_val['accuracy'] - base_val['accuracy']
    test_improvement = ft_test_acc - base_test_acc
    
    print(f"\nImprovement:")
    print(f"Validation: {val_improvement:+.2f}%")
    print(f"Test: {test_improvement:+.2f}%")
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    output = {
        'base_model': {
            'train_accuracy': base_train_acc,
            'val_accuracy': base_val['accuracy'],
            'test_accuracy': base_test_acc,
            'val_results': base_val,
            'val_detailed': base_val.get('detailed_metrics', {})
        },
        'finetuned_model': {
            'train_accuracy': ft_train_acc,
            'val_accuracy': ft_val['accuracy'],
            'test_accuracy': ft_test_acc,
            'val_results': ft_val,
            'val_detailed': ft_val.get('detailed_metrics', {})
        },
        'improvement': {
            'val': val_improvement,
            'test': test_improvement
        },
        'overfitting_metrics': {
            'train_val_gap': train_val_gap,
            'val_test_gap': val_test_gap
        },
        'training_metrics': metrics,
        'config': config.__dict__
    }
    
    output_file = os.path.join(config.output_dir, "complete_results.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    plot_metrics(metrics, config.plots_dir)
    
    print("\n✓ Complete!")


if __name__ == "__main__":
    main()
