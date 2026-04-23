#!/usr/bin/env python3
"""
Train dual LoRA adapters for AgentClinic v2.1 Dynamic Routing
=============================================================
Produces two PEFT adapters that plug directly into agentic_clinic_future_scope.py:
  1. adapters/reasoning_lora  — medical differential diagnosis (structured JSON)
  2. adapters/dialogue_lora   — empathetic clinical conversation

Usage:
  # Step 1: Generate training data from your own AgentClinic scenarios
  python train_lora_adapters.py --generate-data

  # Step 2: Train reasoning adapter
  python train_lora_adapters.py --train reasoning

  # Step 3: Train dialogue adapter
  python train_lora_adapters.py --train dialogue

  # OR: Download pre-existing community adapters from HuggingFace
  python train_lora_adapters.py --download-community

Requirements:
  pip install peft trl datasets bitsandbytes accelerate
"""

import argparse
import json
import os
import random
import sys
import torch
from pathlib import Path

# =============================================================================
# SECTION 1: TRAINING DATA GENERATION FROM AgentClinic SCENARIOS
# =============================================================================

def generate_training_data(scenario_file: str = "agentclinic_nejm.jsonl",
                           output_dir: str = "training_data"):
    """
    Converts AgentClinic-MedQA scenarios into training pairs for both adapters.
    
    Reasoning adapter data format:
      Input:  Patient history + symptoms
      Output: JSON array of top-3 differential diagnoses
    
    Dialogue adapter data format:
      Input:  Patient statement
      Output: Empathetic doctor follow-up question
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[Data Gen] Loading scenarios from {scenario_file}...")
    with open(scenario_file, 'r') as f:
        scenarios = [json.loads(line) for line in f]
    
    reasoning_data = []
    dialogue_data = []
    
    for i, scenario in enumerate(scenarios):
        osce = scenario.get("OSCE_Examination", {})
        patient_info = osce.get("Patient_Actor", "")
        diagnosis = osce.get("Correct_Diagnosis", "")
        objective = osce.get("Objective_for_Doctor", "")
        physical_exam = osce.get("Physical_Examination_Findings", {})
        test_results = osce.get("Test_Results", {})
        
        if not diagnosis:
            continue
        
        patient_str = json.dumps(patient_info) if isinstance(patient_info, dict) else str(patient_info)
        
        # --- Reasoning adapter: Generate differential diagnosis training pairs ---
        # Create plausible differentials around the correct diagnosis
        differentials = _generate_plausible_differentials(diagnosis)
        
        reasoning_data.append({
            "instruction": (
                "You are an internal medical logic engine. Based on the patient "
                "history, list the top 3 differential diagnoses. Output ONLY a "
                'valid JSON array of strings. Example: ["Asthma", "COPD", "Pneumonia"]. '
                "No other text."
            ),
            "input": f"Patient History:\n{patient_str}\n\nObjective: {objective}\n\nUpdate Differential Diagnosis:",
            "output": json.dumps(differentials)
        })
        
        # Add a variant with physical exam findings (mid-case reasoning)
        if physical_exam:
            phys_str = json.dumps(physical_exam) if isinstance(physical_exam, dict) else str(physical_exam)
            reasoning_data.append({
                "instruction": (
                    "You are an internal medical logic engine. Based on all available "
                    "clinical information, list the top 3 differential diagnoses. "
                    'Output ONLY a valid JSON array. Example: ["Disease A", "Disease B", "Disease C"].'
                ),
                "input": (
                    f"Patient History:\n{patient_str}\n\n"
                    f"Physical Examination:\n{phys_str}\n\n"
                    f"Update Differential Diagnosis:"
                ),
                "output": json.dumps([diagnosis] + differentials[1:])  # Correct diagnosis at top
            })
        
        # --- Dialogue adapter: Generate empathetic conversation training pairs ---
        dialogue_templates = [
            {
                "input": f"Patient: I've been feeling really unwell lately. {_extract_chief_complaint(patient_info)}",
                "output": "I understand that must be concerning. Can you tell me more about when these symptoms first started and whether anything makes them better or worse?"
            },
            {
                "input": f"Patient: I'm worried about what's wrong with me.",
                "output": "I completely understand your concern. Let's work through this together step by step. First, can you describe your main symptom in your own words?"
            },
            {
                "input": f"Patient: The doctor told me to get some tests done and I'm nervous.",
                "output": "That's completely normal to feel nervous. These tests will help us understand exactly what's going on so we can find the best treatment for you. Let me explain what each test is for."
            },
            {
                "input": f"Patient: I've had these symptoms for a while but I kept putting off coming to see the doctor.",
                "output": "I'm glad you've come in now. It's important we take a thorough look at everything. Can you walk me through your symptoms from the very beginning?"
            },
        ]
        
        for tmpl in dialogue_templates:
            dialogue_data.append({
                "instruction": (
                    "You are a compassionate and professional doctor speaking directly "
                    "to a patient. Respond with empathy and ask relevant follow-up "
                    "questions. Keep your response to 1-3 sentences. Speak as if "
                    "aloud to the patient."
                ),
                "input": tmpl["input"],
                "output": tmpl["output"]
            })
        
        # Generate case-specific dialogue
        dialogue_data.append({
            "instruction": (
                "You are a compassionate doctor. The patient has described their "
                "symptoms. Respond with empathy and ask a targeted follow-up question "
                "to gather more clinical information. Keep response to 1-3 sentences."
            ),
            "input": f"Patient history context: {objective}\nPatient: I'm here because I haven't been feeling well.",
            "output": _generate_doctor_followup(objective)
        })
    
    # Shuffle and save
    random.shuffle(reasoning_data)
    random.shuffle(dialogue_data)
    
    reasoning_path = os.path.join(output_dir, "reasoning_train.jsonl")
    dialogue_path = os.path.join(output_dir, "dialogue_train.jsonl")
    
    with open(reasoning_path, 'w') as f:
        for item in reasoning_data:
            f.write(json.dumps(item) + "\n")
    
    with open(dialogue_path, 'w') as f:
        for item in dialogue_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"[Data Gen] Generated {len(reasoning_data)} reasoning samples → {reasoning_path}")
    print(f"[Data Gen] Generated {len(dialogue_data)} dialogue samples → {dialogue_path}")
    print(f"[Data Gen] Done! You can now train with: python {sys.argv[0]} --train reasoning")
    return reasoning_path, dialogue_path


def _generate_plausible_differentials(correct_diagnosis: str) -> list:
    """Generate a list of 3 differentials with the correct diagnosis included."""
    # Common differential diagnosis groups
    differential_groups = {
        "cardiac": ["Acute Myocardial Infarction", "Unstable Angina", "Pericarditis",
                     "Aortic Dissection", "Pulmonary Embolism", "Costochondritis"],
        "respiratory": ["Pneumonia", "Asthma", "COPD Exacerbation", "Pulmonary Embolism",
                        "Pleural Effusion", "Lung Cancer", "Bronchitis"],
        "gi": ["Appendicitis", "Cholecystitis", "Peptic Ulcer Disease", "Pancreatitis",
               "Bowel Obstruction", "Diverticulitis", "Gastroenteritis"],
        "neuro": ["Stroke", "Transient Ischemic Attack", "Migraine", "Meningitis",
                  "Multiple Sclerosis", "Brain Tumor", "Epilepsy"],
        "endo": ["Diabetes Mellitus Type 2", "Hypothyroidism", "Hyperthyroidism",
                 "Cushing Syndrome", "Addison Disease", "Diabetes Insipidus"],
        "infectious": ["Sepsis", "Pneumonia", "Urinary Tract Infection", "Cellulitis",
                       "Endocarditis", "Meningitis", "Tuberculosis"],
        "general": ["Anemia", "Hypertension", "Deep Vein Thrombosis", "Fibromyalgia",
                    "Systemic Lupus Erythematosus", "Sarcoidosis"],
    }
    
    # Find the best-matching group and pick 2 random alternatives
    all_diseases = []
    for group in differential_groups.values():
        all_diseases.extend(group)
    
    # Remove the correct diagnosis from alternatives
    alternatives = [d for d in all_diseases if d.lower() != correct_diagnosis.lower()]
    random.shuffle(alternatives)
    
    return [correct_diagnosis, alternatives[0], alternatives[1]]


def _extract_chief_complaint(patient_info) -> str:
    """Extract a brief complaint phrase from patient info."""
    if isinstance(patient_info, dict):
        for key in ["chief_complaint", "presenting_complaint", "symptoms", "HPI"]:
            if key in patient_info:
                val = str(patient_info[key])
                return val[:200] if len(val) > 200 else val
        return str(patient_info)[:200]
    return str(patient_info)[:200]


def _generate_doctor_followup(objective: str) -> str:
    """Generate a context-appropriate doctor follow-up based on the objective."""
    obj_lower = str(objective).lower()
    if any(w in obj_lower for w in ["chest", "cardiac", "heart"]):
        return "I'd like to understand more about your chest discomfort. Can you describe it — is it sharp, dull, or pressure-like? And does it radiate anywhere?"
    elif any(w in obj_lower for w in ["abdominal", "stomach", "belly", "gi"]):
        return "Let me ask about your abdominal symptoms. Where exactly does it hurt, and have you noticed any changes in appetite, nausea, or bowel habits?"
    elif any(w in obj_lower for w in ["headache", "neuro", "dizzy", "weakness"]):
        return "I want to understand your neurological symptoms better. Have you had any vision changes, numbness, tingling, or difficulty with balance?"
    elif any(w in obj_lower for w in ["cough", "breath", "respiratory", "lung"]):
        return "Tell me more about your breathing symptoms. Is the cough productive? Do you notice it worsening at any particular time of day?"
    else:
        return "Thank you for coming in. Can you walk me through your main concern and when it first started?"


# =============================================================================
# SECTION 2: LoRA TRAINING
# =============================================================================

def train_adapter(adapter_type: str,
                  base_model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
                  data_dir: str = "training_data",
                  output_dir: str = "adapters",
                  lora_r: int = 16,
                  lora_alpha: int = 32,
                  num_epochs: int = 3,
                  batch_size: int = 4,
                  learning_rate: float = 2e-4,
                  max_seq_length: int = 512,
                  quantize: bool = True):
    """
    Train a LoRA adapter using QLoRA (4-bit NF4 base + FP16 adapters).
    
    Args:
        adapter_type: "reasoning" or "dialogue"
        base_model_id: HuggingFace model ID for the base model
        data_dir: Directory containing training JSONL files
        output_dir: Where to save the trained adapter
        lora_r: LoRA rank (16 = good balance of capacity vs size)
        lora_alpha: LoRA scaling factor
        num_epochs: Training epochs
        batch_size: Per-device batch size (reduce if OOM)
        learning_rate: AdamW learning rate
        max_seq_length: Maximum sequence length
        quantize: Whether to use 4-bit NF4 quantization for training
    """
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset
    
    # --- Resolve paths ---
    adapter_name = f"{adapter_type}_lora"
    data_file = os.path.join(data_dir, f"{adapter_type}_train.jsonl")
    save_path = os.path.join(output_dir, adapter_name)
    
    if not os.path.exists(data_file):
        print(f"[Train] ERROR: Training data not found at {data_file}")
        print(f"[Train] Run first: python {sys.argv[0]} --generate-data")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"  Training: {adapter_type.upper()} LoRA Adapter")
    print(f"  Base model: {base_model_id}")
    print(f"  Data file: {data_file}")
    print(f"  Output: {save_path}")
    print(f"  LoRA config: r={lora_r}, alpha={lora_alpha}")
    print(f"  Quantized: {quantize}")
    print(f"{'='*70}\n")
    
    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # --- Load base model with QLoRA config ---
    load_kwargs = {"trust_remote_code": True, "device_map": "auto"}
    
    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["quantization_config"] = bnb_config
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
    
    print("[Train] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(base_model_id, **load_kwargs)
    
    if quantize:
        model = prepare_model_for_kbit_training(model)
    
    # --- Configure LoRA ---
    # Target all attention projection matrices for maximum adapter expressiveness
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")
    
    # --- Load dataset ---
    dataset = load_dataset("json", data_files=data_file, split="train")
    print(f"[Train] Loaded {len(dataset)} training examples")
    
    # --- Format for chat template ---
    def format_example(example):
        """Format training example into chat template."""
        messages = [
            {"role": "system", "content": example["instruction"]},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}
    
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    
    # --- Training arguments ---
    training_args = SFTConfig(
        output_dir=f"{save_path}_checkpoints",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        packing=False,
        report_to="none",  # Disable wandb/tensorboard
    )
    
    # --- Train ---
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    print("[Train] Starting training...")
    trainer.train()
    
    # --- Save adapter ---
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"\n[Train] ✅ {adapter_type.upper()} adapter saved to: {save_path}")
    print(f"[Train] Adapter size: {_dir_size_mb(save_path):.1f} MB")
    print(f"[Train] You can now use it in agentic_clinic_future_scope.py!")
    

def _dir_size_mb(path: str) -> float:
    """Calculate directory size in MB."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total / (1024 * 1024)


# =============================================================================
# SECTION 3: DOWNLOAD COMMUNITY ADAPTERS (FALLBACK)
# =============================================================================

def download_community_adapters(output_dir: str = "adapters"):
    """
    Download existing medical LoRA adapters from HuggingFace as a starting point.
    
    NOTE: These are generic medical adapters, NOT specifically trained for the
    dual reasoning/dialogue split. They work as a baseline but training your
    own adapters (Section 2) will give better results for AgentClinic.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[Download] ERROR: pip install huggingface_hub")
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Best available community adapters for Mistral-7B medical tasks
    # These are the closest matches from HuggingFace
    adapters = {
        "reasoning_lora": {
            "repo_id": "ritvik77/Medical_Doctor_AI_LoRA-Mistral-7B-Instruct_FullModel",
            "description": "Medical Doctor AI LoRA — trained on clinical reasoning QA",
            "note": "This is a full model merge; you may need to extract the adapter weights."
        },
        "dialogue_lora": {
            "repo_id": "orYx-models/Mistral-7b-Lora-Medical-ChatSupport",
            "description": "Medical ChatSupport LoRA — trained on empathetic medical dialogue",
            "note": "PEFT adapter format, should load directly."
        },
    }
    
    for adapter_name, info in adapters.items():
        save_path = os.path.join(output_dir, adapter_name)
        print(f"\n[Download] Downloading {adapter_name}...")
        print(f"  Repo: {info['repo_id']}")
        print(f"  Description: {info['description']}")
        print(f"  Note: {info['note']}")
        
        try:
            snapshot_download(
                repo_id=info["repo_id"],
                local_dir=save_path,
                ignore_patterns=["*.bin", "*.safetensors"] if "FullModel" in info["repo_id"] else None,
            )
            print(f"  ✅ Saved to {save_path}")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            print(f"  Try manually: huggingface-cli download {info['repo_id']}")
    
    print("\n[Download] ⚠️  IMPORTANT: Community adapters are a starting point.")
    print("[Download] For best results with AgentClinic, train your own:")
    print(f"[Download]   python {sys.argv[0]} --generate-data")
    print(f"[Download]   python {sys.argv[0]} --train reasoning")
    print(f"[Download]   python {sys.argv[0]} --train dialogue")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or download LoRA adapters for AgentClinic v2.1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate training data from AgentClinic scenarios
  python train_lora_adapters.py --generate-data

  # Train reasoning adapter (differential diagnosis)
  python train_lora_adapters.py --train reasoning

  # Train dialogue adapter (empathetic conversation)
  python train_lora_adapters.py --train dialogue

  # Train both adapters sequentially
  python train_lora_adapters.py --train reasoning
  python train_lora_adapters.py --train dialogue

  # Download pre-existing community adapters from HuggingFace
  python train_lora_adapters.py --download-community

  # Use a different base model (e.g., JSL-MedMistral)
  python train_lora_adapters.py --train reasoning --base-model johnsnowlabs/JSL-MedMistral-7B-v2.2
        """
    )
    
    parser.add_argument("--generate-data", action="store_true",
                        help="Generate training JSONL from agentclinic_medqa.jsonl")
    parser.add_argument("--train", choices=["reasoning", "dialogue"],
                        help="Train a specific adapter type")
    parser.add_argument("--download-community", action="store_true",
                        help="Download community medical LoRA adapters from HuggingFace")
    parser.add_argument("--base-model", default="mistralai/Mistral-7B-Instruct-v0.2",
                        help="Base model HuggingFace ID (default: Mistral-7B-Instruct-v0.2)")
    parser.add_argument("--scenario-file", default="agentclinic_nejm.jsonl",
                        help="JSONL scenario file for data generation (default: agentclinic_nejm.jsonl)")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--no-quantize", action="store_true",
                        help="Disable NF4 quantization during training (needs more VRAM)")
    
    args = parser.parse_args()
    
    if not any([args.generate_data, args.train, args.download_community]):
        parser.print_help()
        sys.exit(0)
    
    if args.generate_data:
        generate_training_data(scenario_file=args.scenario_file)
    
    if args.train:
        train_adapter(
            adapter_type=args.train,
            base_model_id=args.base_model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            quantize=not args.no_quantize,
        )
    
    if args.download_community:
        download_community_adapters()
