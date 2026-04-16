#!/usr/bin/env python3
"""
AgentClinic v2.1 — LangGraph Clinical Decision-Making State Machine
====================================================================
M.Tech Thesis: "Evaluating Large Language Models for Clinical Decision-Making"
IIT Kharagpur | April 2026

Architecture:
    Directed Cyclic Graph (LangGraph) enforcing:
    doctor_reflection → doctor_speaker → {patient | measurement | END}
                  ↑______________________________|

Key Features (v2.1 over v2.0):
    - 4-bit NF4 quantization with double quantization (BitsAndBytes)
    - Heterogeneous multi-engine architecture (Doctor ≠ Patient ≠ Moderator)
    - Full trajectory logging for process-oriented metrics
    - Constrained JSON generation for reflection node (optional)
    - Cognitive bias injection (Recency, Confirmation) ported from Phase 1
    - JSONL results persistence with per-scenario metrics
    - ClinicalTrajectoryEvaluator integration (Stability, Rationality, Efficiency)
"""

import argparse
import anthropic
import openai
import re
import random
import time
import json
import replicate
import os
import datetime
import torch
import subprocess
from typing import TypedDict, List, Optional, Annotated
from operator import add as _list_concat  # for LangGraph reducer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    logging as hf_logging,
)

# --- LANGGRAPH IMPORTS ---
try:
    from langgraph.graph import StateGraph, END
except ImportError:
    print("CRITICAL ERROR: LangGraph is not installed. Please run: pip install langgraph langchain-core")
    exit(1)

# --- REPLICATE MODEL URLS ---
llama2_url = "meta/llama-2-70b-chat"
llama3_url = "meta/meta-llama-3-70b-instruct"
mixtral_url = "mistralai/mixtral-8x7b-instruct-v0.1"

# --- GLOBAL CONSTANTS ---
MAX_RECENT_CASES = 5
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Suppress verbose HuggingFace logging
hf_logging.set_verbosity_error()

# Lazily initialize Replicate client (only if token is set)
replicate_client = None


def _get_replicate_client():
    """Lazy initialization of Replicate client to avoid import-time failures."""
    global replicate_client
    if replicate_client is None:
        token = os.environ.get('REPLICATE_API_TOKEN')
        if token:
            replicate_client = replicate.Client(api_token=token)
    return replicate_client


# =============================================================================
# SECTION 1: UTILITY FUNCTIONS
# =============================================================================

def _stringify_info(info):
    """Safely serialize scenario data to string for prompt injection."""
    if isinstance(info, str):
        return info
    try:
        return json.dumps(info, ensure_ascii=True)
    except TypeError:
        return str(info)


def _clean_dialogue_response(text: str) -> str:
    """
    Strip internal reasoning artifacts from LLM output.
    Removes <think> tags, XML markers, and excessive whitespace.
    Critical for open-source models that leak chain-of-thought.
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    # Remove <think>...</think> blocks (e.g., DeepSeek, Qwen reasoning)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    # Remove other XML/HTML-style tags such as <Answer>, <|assistant|>
    text = re.sub(r"</?[^>]+>", "", text)
    # Collapse whitespace and trim
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_json_list(text: str) -> List[str]:
    """
    Robust JSON array extractor with multi-level fallback.
    Handles: valid JSON → bracket extraction → comma-split → default.
    Designed to tolerate open-source model JSON formatting hallucinations.
    """
    if not text:
        return ["Awaiting further info"]

    # Level 1: Try strict JSON parse
    text_stripped = text.strip()
    if text_stripped.startswith("["):
        try:
            result = json.loads(text_stripped)
            if isinstance(result, list):
                return [str(item) for item in result]
        except json.JSONDecodeError:
            pass

    # Level 2: Extract first bracketed region
    match = re.search(r'\[(.*?)\]', text, re.DOTALL)
    if match:
        inner = match.group(1)
        try:
            return json.loads(f"[{inner}]")
        except (json.JSONDecodeError, ValueError):
            # Level 3: Comma-split fallback
            items = inner.split(',')
            cleaned = [item.strip().strip('"').strip("'") for item in items if item.strip()]
            if cleaned:
                return cleaned

    # Level 4: Try to extract quoted strings from freeform text
    quoted = re.findall(r'"([^"]+)"', text)
    if quoted:
        return quoted[:5]  # Cap at 5 diagnoses

    return ["Awaiting further info"]


def summarize_case_for_recency(scenario):
    """Create a one-line summary of a completed case for recency bias injection."""
    return "Presentation: {} | Diagnosis: {}".format(
        _stringify_info(scenario.patient_information()),
        _stringify_info(scenario.diagnosis_information())
    )


def _extract_test_name(doctor_message: str) -> str:
    """
    Extract the test name from a 'REQUEST TEST: <test>' formatted message.
    Handles variations like 'REQUEST TEST: CBC', 'Request test: Blood Pressure check'.
    """
    match = re.search(r'REQUEST\s+TEST\s*:\s*(.+?)(?:\.|$|\n)', doctor_message, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "Unknown Test"


# =============================================================================
# SECTION 2: MODEL INFERENCE LAYER
# =============================================================================

def query_model(model_str, prompt, system_prompt, model_class,
                tries=30, timeout=20.0, image_requested=False, scene=None,
                max_prompt_len=2**14, clip_prompt=False, is_final=False):
    """
    Unified inference dispatcher for all supported model backends.
    Supports: OpenAI GPT-*, Claude 3.5, Replicate (Llama/Mixtral), HuggingFace local.

    Args:
        model_str: Model identifier string (e.g., 'gpt4', 'HF_mistralai/...')
        prompt: User-facing prompt text
        system_prompt: System instruction text
        model_class: LocalServerLLMWrapper instance (for HF models) or None
        tries: Max retry attempts for API errors
        timeout: Sleep duration between retries
        is_final: If True, increases max_tokens for final diagnosis generation
    """
    # Validate model string
    valid_models = [
        "gpt4", "gpt3.5", "gpt4o", "llama-2-70b-chat",
        "HF_mistralai/Mixtral-8x7B-v0.1", "mixtral-8x7b",
        "gpt-4o-mini", "llama-3-70b-instruct", "gpt4v",
        "claude3.5sonnet", "o1-preview"
    ]
    if model_str not in valid_models and "_HF" not in model_str and not model_str.startswith("HF_"):
        raise ValueError(f"Unsupported model: {model_str}. Valid: {valid_models}")

    for attempt in range(tries):
        if clip_prompt:
            prompt = prompt[:max_prompt_len]
        try:
            # --- HuggingFace Local Models (primary path for thesis experiments) ---
            if model_str.startswith('HF_') or '_HF' in model_str:
                max_tokens = 200 if is_final else 150
                answer = model_class.generate_response(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_new_tokens=max_tokens
                )
                answer = re.sub(r"\s+", " ", answer)
                return answer

            # --- OpenAI Models ---
            elif model_str in ("gpt4", "gpt3.5", "gpt-4o-mini", "gpt4o", "gpt4v"):
                model_map = {
                    "gpt4": "gpt-4-turbo-preview",
                    "gpt3.5": "gpt-3.5-turbo",
                    "gpt-4o-mini": "gpt-4o-mini",
                    "gpt4o": "gpt-4o",
                    "gpt4v": "gpt-4-vision-preview",
                }
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                response = openai.ChatCompletion.create(
                    model=model_map[model_str],
                    messages=messages,
                    temperature=0.05,
                    max_tokens=200,
                )
                answer = response["choices"][0]["message"]["content"]
                answer = re.sub(r"\s+", " ", answer)

            # --- Anthropic Claude ---
            elif model_str == "claude3.5sonnet":
                client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                message = client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    system=system_prompt,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                )
                answer = json.loads(message.to_json())["content"][0]["text"]

            # --- Replicate API Models ---
            elif model_str == 'llama-3-70b-instruct':
                output = replicate.run(
                    llama3_url, input={
                        "prompt": prompt,
                        "system_prompt": system_prompt,
                        "max_new_tokens": 200,
                    })
                answer = ''.join(output)
                answer = re.sub(r"\s+", " ", answer)

            elif model_str == 'llama-2-70b-chat':
                output = replicate.run(
                    llama2_url, input={
                        "prompt": prompt,
                        "system_prompt": system_prompt,
                        "max_new_tokens": 200,
                    })
                answer = ''.join(output)
                answer = re.sub(r"\s+", " ", answer)

            elif model_str == "o1-preview":
                messages = [{"role": "user", "content": system_prompt + prompt}]
                response = openai.ChatCompletion.create(
                    model="o1-preview-2024-09-12",
                    messages=messages,
                )
                answer = response["choices"][0]["message"]["content"]
                answer = re.sub(r"\s+", " ", answer)

            else:
                raise ValueError(f"Model {model_str} matched validation but has no handler.")

            return answer

        except Exception as e:
            print(f"[Attempt {attempt+1}/{tries}] Model {model_str} error: {e}. Retrying in {timeout}s...")
            time.sleep(timeout)
            continue

    raise RuntimeError(f"Max retries ({tries}) exceeded for model {model_str}")


def send_notification(topic, message):
    """Sends a push notification to phone via ntfy.sh"""
    if not topic:
        return
    try:
        # Use curl directly to avoid new dependencies
        cmd = ["curl", "-s", "-d", message, f"ntfy.sh/{topic}"]
        subprocess.run(cmd, timeout=5)
    except Exception as e:
        print(f"[Warning] Failed to send notification: {e}")


# =============================================================================
# SECTION 3: LOCAL LLM WRAPPER WITH QUANTIZATION
# =============================================================================

class LocalServerLLMWrapper:
    """
    Wrapper for HuggingFace models with 4-bit NF4 quantization.

    Technical Details:
        - Uses BitsAndBytes NF4 quantization (Dettmers et al., 2023)
        - Double quantization reduces memory by ~0.37 bits/param further
        - Flash Attention 2 for O(n) memory attention on H100
        - Compute in bfloat16 for numerical stability

    Memory footprint: 7B model → ~4 GB VRAM (vs ~14 GB in fp16)
    """

    def __init__(self, model_id: str, quantize: bool = True):
        """
        Args:
            model_id: HuggingFace model identifier
            quantize: If True, load with 4-bit NF4 quantization
        """
        quantize=False
        print(f"[LocalServerLLMWrapper] Loading model: {model_id} (quantize={quantize})")
        load_start = time.time()

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # --- Quantization Configuration ---
        load_kwargs = {"device_map": "auto", "trust_remote_code": True}

        if quantize:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",             # NormalFloat4 — optimal for normally distributed weights
                bnb_4bit_compute_dtype=torch.bfloat16,  # bf16 compute for stability on H100
                bnb_4bit_use_double_quant=True,         # Nested quantization: saves ~0.37 bits/param
            )
            load_kwargs["quantization_config"] = bnb_config
        else:
            load_kwargs["torch_dtype"] = "auto"

        # Attempt Flash Attention 2 (silently fall back if not available)
        try:
            import flash_attn
            load_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            pass  # Fall back to default attention

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        load_duration = time.time() - load_start
        param_count = sum(p.numel() for p in self.model.parameters()) / 1e9
        print(f"[LocalServerLLMWrapper] Loaded {model_id} ({param_count:.1f}B params) in {load_duration:.1f}s")

    def generate_response(self, prompt: str, system_prompt: str,
                          max_new_tokens: int = 75, temperature: float = 0.7) -> str:
        """
        Generate a response using the chat template format.

        Args:
            prompt: User message content
            system_prompt: System instruction
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.1 for structured, 0.7 for dialogue)

        Returns:
            Generated text string, stripped of prompt echo
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        output = self.pipe(
            formatted_prompt,
            max_new_tokens=max_new_tokens,
            return_full_text=False,   # Only return generated text, not the prompt
            do_sample=True,
            temperature=max(temperature, 0.01),  # Prevent division by zero
            repetition_penalty=1.15,  # Prevents looping/repetition hallucinations
        )
        return output[0]['generated_text'].strip()


# =============================================================================
# SECTION 4: CLINICAL TRAJECTORY EVALUATOR
# =============================================================================

class ClinicalTrajectoryEvaluator:
    """
    Process-oriented evaluation metrics that go beyond binary accuracy.

    Metrics:
        1. Diagnostic Stability: Jaccard similarity of consecutive differential diagnoses
        2. Test Rationality: Penalizes ordering expensive tests before basic workup
        3. Information Efficiency: Useful clinical data points per turn
    """

    def __init__(self):
        self.basic_tests = {
            "cbc", "bmp", "vitals", "blood pressure", "temperature",
            "x-ray", "ecg", "ekg", "urinalysis", "blood glucose",
            "pulse oximetry", "chest x-ray",
        }
        self.advanced_tests = {
            "mri", "ct", "ct scan", "biopsy", "endoscopy",
            "lumbar puncture", "pet scan", "bronchoscopy",
            "angiography", "bone marrow biopsy",
        }

    def compute_diagnostic_stability(self, differential_trajectory: List[List[str]]) -> float:
        """
        Metric 1: Diagnostic Stability Score (0.0 to 1.0)
        Measures the average Jaccard Similarity between consecutive differential diagnoses.
        High (~1.0) = stable, logical refinement. Low (~0.0) = erratic hypothesis changes.
        """
        if len(differential_trajectory) < 2:
            return 1.0

        stability_scores = []
        for i in range(1, len(differential_trajectory)):
            prev_set = set(d.lower().strip() for d in differential_trajectory[i - 1] if d.strip())
            curr_set = set(d.lower().strip() for d in differential_trajectory[i] if d.strip())

            if not prev_set and not curr_set:
                stability_scores.append(1.0)
                continue
            if not prev_set or not curr_set:
                stability_scores.append(0.0)
                continue

            intersection = len(prev_set.intersection(curr_set))
            union = len(prev_set.union(curr_set))
            stability_scores.append(intersection / union if union > 0 else 0.0)

        return sum(stability_scores) / len(stability_scores) if stability_scores else 1.0

    def compute_test_rationality(self, tests_ordered: List[str]) -> float:
        """
        Metric 2: Test Rationality Score (0.0 or 1.0)
        Returns 0.0 if an advanced/expensive test was ordered before any basic test.
        """
        if not tests_ordered:
            return 1.0

        had_basic_prior = False
        for test in tests_ordered:
            test_lower = test.lower()
            is_advanced = any(adv in test_lower for adv in self.advanced_tests)
            is_basic = any(bsc in test_lower for bsc in self.basic_tests)

            if is_basic:
                had_basic_prior = True
            if is_advanced and not had_basic_prior:
                return 0.0  # Violated protocol

        return 1.0

    def compute_information_efficiency(self, turn_count: int, num_symptoms_gathered: int) -> float:
        """
        Metric 3: Information Efficiency = data points gathered / turns used.
        """
        if turn_count == 0:
            return 0.0
        return round(num_symptoms_gathered / turn_count, 2)

    def generate_full_report(self, trajectory_data: dict) -> dict:
        """Generate all metrics from a completed scenario's trajectory data."""
        stability = self.compute_diagnostic_stability(
            trajectory_data.get("differential_trajectory", [])
        )
        rationality = self.compute_test_rationality(
            trajectory_data.get("tests_ordered", [])
        )
        efficiency = self.compute_information_efficiency(
            trajectory_data.get("turn_count", 1),
            len(trajectory_data.get("tests_ordered", [])) + trajectory_data.get("turn_count", 1),
        )

        return {
            "Diagnostic_Stability": round(stability, 4),
            "Test_Rationality": rationality,
            "Information_Efficiency": efficiency,
        }


# =============================================================================
# SECTION 5: DATASET CLASSES
# =============================================================================

class ScenarioMedQA:
    """Structured OSCE clinical scenario from MedQA dataset."""

    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]

    def patient_information(self):
        return self.patient_info

    def examiner_information(self):
        return self.examiner_info

    def exam_information(self):
        exams = dict(self.physical_exams)  # Defensive copy
        exams["tests"] = self.tests
        return exams

    def diagnosis_information(self):
        return self.diagnosis


class ScenarioLoaderMedQA:
    """Loads all scenarios from agentclinic_medqa.jsonl."""

    def __init__(self) -> None:
        with open("agentclinic_medqa.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioMedQA(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)

    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios) - 1)]

    def get_scenario(self, id):
        return self.sample_scenario() if id is None else self.scenarios[id]


# =============================================================================
# SECTION 6: AGENT CLASSES WITH BIAS SUPPORT
# =============================================================================

class PatientAgent:
    """
    Simulates a patient with persona-driven responses.
    Supports cognitive bias injection (recency, confirmation) for
    ecological validity experiments.
    """

    def __init__(self, scenario, backend_str="gpt4", bias_present=None) -> None:
        self.agent_hist = ""
        self.backend = backend_str
        self.bias_present = (None if bias_present == "None" or bias_present is None else bias_present)
        self.scenario = scenario
        self.reset()

    def generate_bias(self) -> str:
        """Generate cognitive bias prompt injection for the patient persona."""
        bias_prompts = {
            "recency": "\nRecently, you had a friend with similar symptoms to yours that was diagnosed with cancer. This affects how you interact with the doctor.\n",
            "frequency": "\nYou recently learned that most frequently, doctors diagnose this as cancer based on past cases. This affects how you interact with the doctor.\n",
            "false_consensus": "\nMost of your friends who heard about your symptoms believe that you have cancer. This affects how you interact with the doctor.\n",
            "self_diagnosis": "\nYou recently performed a thorough investigation of your symptoms online and have come to the conclusion that you should be diagnosed with cancer. While the doctor may believe otherwise you remain convinced you have cancer. This affects how you interact with the doctor.\n",
            "confirmation": "\nYou are already convinced that you have cancer based on your own research. You selectively listen to information that confirms this belief and dismiss anything that contradicts it.\n",
        }
        if self.bias_present and self.bias_present in bias_prompts:
            return bias_prompts[self.bias_present]
        elif self.bias_present and self.bias_present not in bias_prompts:
            print(f"WARNING: Patient bias type '{self.bias_present}' not supported, ignoring.")
        return ""

    def inference_patient(self, question, model_class) -> str:
        prompt = (
            f"\nHere is a history of your dialogue: {self.agent_hist}\n"
            f" Here was the doctor response: {question}"
            f"Now please continue your dialogue\nPatient: "
        )
        answer = query_model(self.backend, prompt, self.system_prompt(), model_class)
        answer = _clean_dialogue_response(answer)
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer

    def system_prompt(self) -> str:
        bias_prompt = self.generate_bias()
        base = (
            "You are a patient in a clinic who only responds in the form of dialogue. "
            "You are being inspected by a doctor who will ask you questions and will perform "
            "exams on you in order to understand your disease. Your answer will only be 1-3 "
            "sentences in length. Speak as if aloud to the doctor. Never include analysis, "
            "bullet points, <think> tags, stage directions, or anything besides your spoken reply."
        )
        symptoms = (
            f"\n\nBelow is all of your information. {self.symptoms}. "
            "\n\n Remember, you must not reveal your disease explicitly but may only "
            "convey the symptoms you have in the form of dialogue if you are asked."
        )
        return base + bias_prompt + symptoms

    def reset(self) -> None:
        self.agent_hist = ""
        self.symptoms = self.scenario.patient_information()

    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"


class DoctorAgent:
    """
    Domain-expert clinician agent with metacognitive reflection.

    Features:
        - Hidden differential diagnosis tracking (Reflection Node)
        - Cognitive bias injection (recency with case memory, confirmation)
        - Forced diagnosis fallback on final turn
        - Turn-count enforcement
    """

    def __init__(self, scenario, backend_str="gpt4", max_infs=20,
                 bias_present=None, recent_cases=None) -> None:
        self.infs = 0
        self.MAX_INFS = max_infs
        self.agent_hist = ""
        self.backend = backend_str
        self.bias_present = (None if bias_present == "None" or bias_present is None else bias_present)
        self.scenario = scenario
        self.recent_cases = list(recent_cases) if recent_cases else []
        self.reset()

    def generate_bias(self) -> str:
        """Generate cognitive bias prompt injection for the doctor agent."""
        if self.bias_present == "recency":
            return self._recency_bias_prompt()
        elif self.bias_present == "confirmation":
            return (
                "\nYou are initially confident that the patient has cancer. "
                "This affects how you interact with the patient. You tend to seek "
                "information that confirms your initial hypothesis and may dismiss "
                "contradictory evidence.\n"
            )
        elif self.bias_present and self.bias_present not in ("recency", "confirmation"):
            print(f"WARNING: Doctor bias type '{self.bias_present}' not in supported set {{recency, confirmation}}.")
        return ""

    def _recency_bias_prompt(self) -> str:
        """
        Generates recency bias using actual recent case summaries.
        More ecologically valid than a generic "you saw cancer recently" prompt.
        """
        if not self.recent_cases:
            return (
                "\nRecently, you learned that a doctor with similar symptoms to the patient "
                "diagnosed their patient with cancer. This affects how you interact with the patient.\n"
            )
        recent_text = "\nYour most recent patients were:\n"
        for idx, case in enumerate(self.recent_cases[-MAX_RECENT_CASES:], 1):
            recent_text += f"{idx}. {case}\n"
        return recent_text + "These encounters strongly influence your current judgement and increase your tendency toward recency bias.\n"

    def reflect_metacognition(self, latest_input, model_class) -> List[str]:
        """
        THESIS INNOVATION 1: Hidden Reflection Node.
        Forces the Doctor LLM to output its latent differential diagnosis as a
        structured JSON array BEFORE generating spoken dialogue. This makes
        internal reasoning observable and mathematically graphable turn-by-turn.
        """
        sys_prompt = (
            "You are an internal medical logic engine. Based on the patient history, "
            "list the top 3 differential diagnoses. Output ONLY a valid JSON array of "
            'strings. Example: ["Asthma", "COPD", "Pneumonia"]. No other text.'
        )
        prompt = (
            f"History:\n{self.agent_hist}\n"
            f"Latest Input:\n{latest_input}\n\n"
            f"Update Differential Diagnosis:"
        )

        # Use very low temperature for structured JSON output
        if hasattr(model_class, 'generate_response'):
            raw_response = model_class.generate_response(
                prompt=prompt,
                system_prompt=sys_prompt,
                max_new_tokens=80,
                temperature=0.1,
            )
        else:
            raw_response = query_model(
                self.backend, prompt, sys_prompt, model_class, is_final=False
            )
        return extract_json_list(raw_response)

    def inference_doctor(self, question, model_class, is_final=False) -> str:
        """Generate doctor's spoken response, with forced diagnosis on final turn."""
        if self.infs >= self.MAX_INFS:
            return "Maximum inferences reached"

        if is_final:
            prompt = (
                f"\nHistory: {self.agent_hist}\nPatient: {question}\n\n"
                "⚠️ FINAL TURN. You must provide a definitive diagnosis now. "
                "Do NOT suggest more tests. Do NOT say 'Pending'.\n"
                "Output ONLY: DIAGNOSIS READY: [Exact Disease Name]\nDoctor: "
            )
        else:
            prompt = (
                f"\nHistory: {self.agent_hist}\nPatient: {question}\n"
                "Continue your dialogue\nDoctor: "
            )

        answer = query_model(
            self.backend, prompt, self.system_prompt(), model_class, is_final=is_final
        )
        answer = _clean_dialogue_response(answer)

        # Forced diagnosis fallback on final turn
        if is_final and "DIAGNOSIS READY" not in answer.upper():
            answer = self._force_final_diagnosis(question, model_class)

        self.agent_hist += question + "\n\n" + answer + "\n\n"
        self.infs += 1
        return answer

    def _force_final_diagnosis(self, question, model_class) -> str:
        """
        Aggressive fallback: directly ask the model to emit DIAGNOSIS READY.
        Uses few-shot examples to demonstrate the expected format.
        """
        force_prompt = (
            f"History: {self.agent_hist}\n"
            f"Patient: {question}\n\n"
            "You must now provide your final diagnosis. "
            "Examples of correct output format:\n"
            "DIAGNOSIS READY: Asthma\n"
            "DIAGNOSIS READY: Multiple Sclerosis\n"
            "DIAGNOSIS READY: Hirschsprung disease\n\n"
            "Now provide your diagnosis for this case:\n"
            "DIAGNOSIS READY: "
        )
        force_sys = (
            "You are a senior clinician. Output ONLY the diagnosis name "
            "after 'DIAGNOSIS READY:'. No other words."
        )

        if hasattr(model_class, 'generate_response'):
            forced = model_class.generate_response(
                prompt=force_prompt,
                system_prompt=force_sys,
                max_new_tokens=50,
                temperature=0.1,
            )
        else:
            forced = query_model(
                self.backend, force_prompt, force_sys, model_class, is_final=True
            )

        forced = _clean_dialogue_response(forced)

        # Extract only the diagnosis name
        parts = re.split(r'DIAGNOSIS READY:', forced, flags=re.IGNORECASE)
        diagnosis = parts[-1].strip().split('.')[0].split('\n')[0].strip()

        if not diagnosis or diagnosis.lower() in ("", "unable", "unknown"):
            diagnosis = "Unable to determine (insufficient information)"

        return f"DIAGNOSIS READY: {diagnosis}"

    def system_prompt(self) -> str:
        bias_prompt = self.generate_bias()

        if self.infs >= self.MAX_INFS - 1:
            base = (
                f"You are a doctor named Dr. Agent. This is your FINAL question — "
                f"you MUST provide a diagnosis NOW. You have asked {self.infs} out of "
                f"{self.MAX_INFS} questions. You MUST respond with: DIAGNOSIS READY: "
                "[diagnosis name here]. Do not ask more questions."
            )
        else:
            base = (
                f"You are a doctor named Dr. Agent who only responds in the form of dialogue. "
                f"You are inspecting a patient who you will ask questions in order to understand "
                f"their disease. You are only allowed to ask {self.MAX_INFS} questions total "
                f"before you must make a decision. You have asked {self.infs} questions so far. "
                'You can request test results using the format "REQUEST TEST: [test]". '
                'For example, "REQUEST TEST: Chest_X-Ray". Your dialogue will only be '
                '1-3 sentences in length. Once you have decided to make a diagnosis please '
                'type "DIAGNOSIS READY: [diagnosis here]"'
            )

        base += (
            "\n\nSpeak only in natural conversational sentences directed at the patient. "
            "Never output analysis, internal thoughts, <think> tags, XML-style markers, "
            "or bullet points—only the dialogue you would say aloud."
        )
        presentation = (
            f"\n\nBelow is all of the information you have. {self.presentation}. "
            "\n\n Remember, you must discover their disease by asking them questions. "
            "You are also able to provide exams."
        )
        return base + bias_prompt + presentation

    def reset(self) -> None:
        self.agent_hist = ""
        self.presentation = self.scenario.examiner_information()


class MeasurementAgent:
    """Lab/test result reader agent. Returns structured test results."""

    def __init__(self, scenario, backend_str="gpt4") -> None:
        self.agent_hist = ""
        self.backend = backend_str
        self.scenario = scenario
        self.reset()

    def inference_measurement(self, question, model_class) -> str:
        prompt = (
            f"\nHistory: {self.agent_hist}\n"
            f"Doctor request: {question}"
        )
        answer = query_model(self.backend, prompt, self.system_prompt(), model_class)
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer

    def system_prompt(self) -> str:
        return (
            'You are a measurement reader who responds with medical test results. '
            'Please respond in the format "RESULTS: [results here]"'
            f"\n\nBelow is all of the information you have. {self.information}. "
            "\n\n If the requested results are not in your data then you can respond "
            "with NORMAL READINGS."
        )

    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"

    def reset(self) -> None:
        self.agent_hist = ""
        self.information = self.scenario.exam_information()


def compare_results(diagnosis, correct_diagnosis, moderator_llm, model_class):
    """
    Semantic comparison of predicted vs ground-truth diagnosis.
    Uses a moderator LLM as an impartial judge.
    """
    prompt = (
        f"\nCorrect diagnosis: {correct_diagnosis}\n"
        f"Doctor's diagnosis: {diagnosis}\n"
        "Are these the same disease or condition? Consider synonyms and "
        "equivalent medical terminology."
    )
    sys = (
        "You are responsible for determining if the doctor's diagnosis and the "
        "correct diagnosis refer to the same disease or condition. Consider "
        "medical synonyms (e.g., 'Heart Attack' = 'Myocardial Infarction'). "
        "Respond ONLY with 'Yes' or 'No'. Nothing else."
    )
    answer = query_model(moderator_llm, prompt, sys, model_class)
    return answer.lower()


# =============================================================================
# SECTION 7: LANGGRAPH STATE MACHINE
# =============================================================================

class ClinicalState(TypedDict):
    """
    State schema for the LangGraph clinical decision-making state machine.

    Core fields:
        last_input_to_doctor: What the patient/lab just said
        last_doctor_message: What the doctor just asked/said
        differential_diagnoses: Current top differential diagnoses (hidden from patient)
        turn_count: Question counter enforcing max-turn constraint
        diagnosis_ready: Terminal condition flag

    Trajectory fields (for process-oriented metrics):
        differential_trajectory: Accumulated DDx snapshots across all turns
        tests_ordered: Extracted test names from REQUEST TEST messages
        full_dialogue: Complete conversation log with timestamps
    """
    last_input_to_doctor: str
    last_doctor_message: str
    differential_diagnoses: List[str]
    turn_count: int
    diagnosis_ready: bool
    # --- Trajectory tracking ---
    differential_trajectory: List[List[str]]
    tests_ordered: List[str]
    full_dialogue: List[dict]


# =============================================================================
# SECTION 8: MAIN EXECUTION PIPELINE
# =============================================================================

def main(
    api_key: str,
    replicate_api_key: str,
    doctor_llm: str,
    patient_llm: str,
    measurement_llm: str,
    moderator_llm: str,
    num_scenarios: int,
    dataset: str,
    total_inferences: int,
    scenario_id: Optional[int] = None,
    doctor_bias: str = "None",
    patient_bias: str = "None",
    doctor_model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
    patient_model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
    moderator_model_id: Optional[str] = None,
    quantize: bool = True,
    output_file: Optional[str] = None,
    notify_topic: Optional[str] = None,
):
    """
    Main experiment execution pipeline.

    THESIS INNOVATION 2: Heterogeneous Multi-Engine Architecture.
    Separate LLM instances for Doctor (domain-expert), Patient (generalist persona),
    and Moderator (impartial judge). Eliminates 'Lexical Leakage' from homogeneous setups.

    Args:
        doctor_model_id: HuggingFace model ID for the Doctor agent
        patient_model_id: HuggingFace model ID for the Patient agent
        moderator_model_id: HuggingFace model ID for the Moderator (defaults to patient_model_id)
        quantize: Whether to use 4-bit NF4 quantization
        output_file: Path for JSONL results output (auto-generated if None)
    """
    # --- API Key Setup ---
    openai.api_key = api_key
    if patient_llm in ["llama-3-70b-instruct"]:
        os.environ["REPLICATE_API_TOKEN"] = replicate_api_key

    # --- Ensure results directory exists ---
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Model Loading: Heterogeneous Architecture ---
    # Only load HuggingFace models if the LLM string indicates local inference
    doctor_engine = None
    patient_engine = None
    moderator_engine = None

    if doctor_llm.startswith("HF_"):
        print(f"\n{'='*60}")
        print(f"LOADING DOCTOR ENGINE: {doctor_model_id}")
        print(f"{'='*60}")
        doctor_engine = LocalServerLLMWrapper(model_id=doctor_model_id, quantize=quantize)

    if patient_llm.startswith("HF_"):
        if patient_model_id == doctor_model_id and doctor_engine is not None:
            # Share engine if same model (homogeneous baseline experiment)
            print(f"[INFO] Patient shares Doctor engine (homogeneous mode: {doctor_model_id})")
            patient_engine = doctor_engine
        else:
            print(f"\n{'='*60}")
            print(f"LOADING PATIENT ENGINE: {patient_model_id}")
            print(f"{'='*60}")
            patient_engine = LocalServerLLMWrapper(model_id=patient_model_id, quantize=quantize)

    if moderator_llm.startswith("HF_"):
        mod_model = moderator_model_id or patient_model_id
        if mod_model == patient_model_id and patient_engine is not None:
            print(f"[INFO] Moderator shares Patient engine ({mod_model})")
            moderator_engine = patient_engine
        elif mod_model == doctor_model_id and doctor_engine is not None:
            print(f"[INFO] Moderator shares Doctor engine ({mod_model})")
            moderator_engine = doctor_engine
        else:
            print(f"\n{'='*60}")
            print(f"LOADING MODERATOR ENGINE: {mod_model}")
            print(f"{'='*60}")
            moderator_engine = LocalServerLLMWrapper(model_id=mod_model, quantize=quantize)

    # --- Dataset Loading ---
    scenario_loader = ScenarioLoaderMedQA()
    print(f"\nLoaded {scenario_loader.num_scenarios} scenarios from {dataset}")

    # --- Evaluator ---
    evaluator = ClinicalTrajectoryEvaluator()

    # --- Output File ---
    if output_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        doctor_tag = doctor_model_id.split("/")[-1] if doctor_llm.startswith("HF_") else doctor_llm
        bias_tag = f"_bias-{doctor_bias}" if doctor_bias != "None" else ""
        output_file = os.path.join(
            RESULTS_DIR,
            f"results_{doctor_tag}{bias_tag}_{timestamp}.jsonl"
        )
    print(f"Results will be saved to: {output_file}")

    # --- Experiment Loop ---
    total_correct = 0
    total_presents = 0
    recent_case_summaries = []

    scenario_ids = (
        [scenario_id] if scenario_id is not None
        else range(0, min(num_scenarios or 1, scenario_loader.num_scenarios))
    )

    for i, _scenario_id in enumerate(scenario_ids):
        # Progress Notification every 10 scenarios
        if i > 0 and i % 10 == 0:
            msg = f"Experiment Progress: {i}/{len(scenario_ids)} scenarios done. (Model: {doctor_model_id.split('/')[-1]})"
            send_notification(notify_topic, msg)

        scenario_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"SCENARIO {_scenario_id} / {len(scenario_ids)} | Bias: {doctor_bias}")
        print(f"{'='*60}")

        total_presents += 1
        scenario = scenario_loader.get_scenario(id=_scenario_id)

        # --- Instantiate Agents ---
        meas_agent = MeasurementAgent(scenario=scenario, backend_str=measurement_llm)
        patient_agent = PatientAgent(
            scenario=scenario, backend_str=patient_llm, bias_present=patient_bias,
        )
        doctor_agent = DoctorAgent(
            scenario=scenario, backend_str=doctor_llm, max_infs=total_inferences,
            bias_present=doctor_bias, recent_cases=recent_case_summaries,
        )

        # Select the appropriate engine for each agent role
        doc_eng = doctor_engine
        pat_eng = patient_engine
        mod_eng = moderator_engine

        # =====================================================================
        # LANGGRAPH NODE DEFINITIONS (closures over agent instances)
        # =====================================================================

        def doctor_reflection_node(state: ClinicalState) -> dict:
            """
            HIDDEN NODE: Updates differential diagnosis without speaking to patient.
            This is Thesis Innovation 1 — making latent LLM reasoning observable.
            """
            diff = doctor_agent.reflect_metacognition(
                state["last_input_to_doctor"], doc_eng
            )
            print(f"\n  [METACOGNITION] Differential: {diff}")

            # Append to trajectory for Diagnostic Stability computation
            updated_trajectory = list(state.get("differential_trajectory", []))
            updated_trajectory.append(diff)

            return {
                "differential_diagnoses": diff,
                "differential_trajectory": updated_trajectory,
            }

        def doctor_speaker_node(state: ClinicalState) -> dict:
            """Doctor generates spoken dialogue or final diagnosis."""
            is_final = state["turn_count"] >= total_inferences - 1
            msg = doctor_agent.inference_doctor(
                state["last_input_to_doctor"], doc_eng, is_final=is_final
            )

            # Forced diagnosis fallback (already handled in inference_doctor,
            # but double-check for edge cases)
            if is_final and "DIAGNOSIS READY" not in msg.upper():
                force_prompt = (
                    f"History: {doctor_agent.agent_hist}\n"
                    f"Patient: {state['last_input_to_doctor']}\n\n"
                    "You must now provide your final diagnosis.\n"
                    "DIAGNOSIS READY: "
                )
                force_sys = "Output ONLY the diagnosis name. No other words."
                if hasattr(doc_eng, 'generate_response'):
                    forced = doc_eng.generate_response(
                        prompt=force_prompt, system_prompt=force_sys,
                        max_new_tokens=50, temperature=0.1,
                    )
                else:
                    forced = query_model(
                        doctor_llm, force_prompt, force_sys, doc_eng, is_final=True
                    )
                forced = _clean_dialogue_response(forced)
                parts = re.split(r'DIAGNOSIS READY:', forced, flags=re.IGNORECASE)
                diagnosis = parts[-1].strip().split('.')[0].split('\n')[0].strip()
                msg = f"DIAGNOSIS READY: {diagnosis}"

            print(f"\n  Doctor [Turn {state['turn_count']}]: {msg}")

            # Log to full dialogue
            dialogue_entry = {
                "turn": state["turn_count"],
                "role": "doctor",
                "message": msg,
                "differential": state.get("differential_diagnoses", []),
                "timestamp": time.time(),
            }
            updated_dialogue = list(state.get("full_dialogue", []))
            updated_dialogue.append(dialogue_entry)

            return {
                "last_doctor_message": msg,
                "turn_count": state["turn_count"] + 1,
                "diagnosis_ready": "DIAGNOSIS READY" in msg.upper(),
                "full_dialogue": updated_dialogue,
            }

        def patient_node(state: ClinicalState) -> dict:
            """Patient responds to doctor's question."""
            msg = patient_agent.inference_patient(
                state["last_doctor_message"], pat_eng
            )
            meas_agent.add_hist(msg)
            print(f"\n  Patient: {msg}")

            dialogue_entry = {
                "turn": state["turn_count"],
                "role": "patient",
                "message": msg,
                "timestamp": time.time(),
            }
            updated_dialogue = list(state.get("full_dialogue", []))
            updated_dialogue.append(dialogue_entry)

            return {
                "last_input_to_doctor": msg,
                "full_dialogue": updated_dialogue,
            }

        def measurement_node(state: ClinicalState) -> dict:
            """Lab/test reader returns results."""
            # Extract test name for Test Rationality metric
            test_name = _extract_test_name(state["last_doctor_message"])
            msg = meas_agent.inference_measurement(
                state["last_doctor_message"], pat_eng  # Measurement uses patient/generalist engine
            )
            patient_agent.add_hist(msg)
            print(f"\n  Measurement [{test_name}]: {msg}")

            # Update tests_ordered for Test Rationality metric
            updated_tests = list(state.get("tests_ordered", []))
            updated_tests.append(test_name)

            dialogue_entry = {
                "turn": state["turn_count"],
                "role": "measurement",
                "message": msg,
                "test_name": test_name,
                "timestamp": time.time(),
            }
            updated_dialogue = list(state.get("full_dialogue", []))
            updated_dialogue.append(dialogue_entry)

            return {
                "last_input_to_doctor": msg,
                "tests_ordered": updated_tests,
                "full_dialogue": updated_dialogue,
            }

        def route_doctor(state: ClinicalState):
            """
            THESIS INNOVATION 3: Deterministic routing via state machine.
            The doctor MUST go through reflection before every turn.
            Routing prevents hallucinated workflow violations.
            """
            if state["diagnosis_ready"] or state["turn_count"] >= total_inferences:
                return END
            if "REQUEST TEST" in state["last_doctor_message"].upper():
                return "measurement"
            return "patient"

        # =====================================================================
        # BUILD & COMPILE LANGGRAPH
        # =====================================================================

        workflow = StateGraph(ClinicalState)
        workflow.add_node("doctor_reflection", doctor_reflection_node)
        workflow.add_node("doctor_speaker", doctor_speaker_node)
        workflow.add_node("patient", patient_node)
        workflow.add_node("measurement", measurement_node)

        workflow.set_entry_point("doctor_reflection")
        workflow.add_edge("doctor_reflection", "doctor_speaker")
        workflow.add_conditional_edges("doctor_speaker", route_doctor)
        workflow.add_edge("patient", "doctor_reflection")
        workflow.add_edge("measurement", "doctor_reflection")

        app = workflow.compile()

        # =====================================================================
        # EXECUTE GRAPH
        # =====================================================================

        initial_state: ClinicalState = {
            "last_input_to_doctor": "Please begin the examination. What would you like to ask or test?",
            "last_doctor_message": "",
            "differential_diagnoses": [],
            "turn_count": 0,
            "diagnosis_ready": False,
            "differential_trajectory": [],
            "tests_ordered": [],
            "full_dialogue": [],
        }

        final_state = app.invoke(initial_state)

        # =====================================================================
        # EVALUATION
        # =====================================================================

        print("\n  Doctor's diagnosis complete.")
        final_diagnosis_str = final_state["last_doctor_message"].split("DIAGNOSIS READY:")[-1].strip()

        comparison_result = compare_results(
            final_diagnosis_str, scenario.diagnosis_information(),
            moderator_llm, mod_eng
        )
        correctness = "yes" in str(comparison_result).lower()
        if correctness:
            total_correct += 1

        # Compute process metrics
        trajectory_data = {
            "differential_trajectory": final_state.get("differential_trajectory", []),
            "tests_ordered": final_state.get("tests_ordered", []),
            "turn_count": final_state.get("turn_count", 0),
        }
        metrics_report = evaluator.generate_full_report(trajectory_data)

        scenario_duration = time.time() - scenario_start_time

        print(f"\n  Doctor's Diagnosis:  {final_diagnosis_str}")
        print(f"  Ground Truth:        {scenario.diagnosis_information()}")
        print(f"  Result:              {'✅ CORRECT' if correctness else '❌ INCORRECT'}")
        print(f"  Running Accuracy:    {int((total_correct / total_presents) * 100)}%")
        print(f"  Duration:            {scenario_duration:.1f}s")
        print(f"  Metrics:             {metrics_report}")

        # =====================================================================
        # PERSIST RESULTS TO JSONL
        # =====================================================================

        result_record = {
            "scenario_id": _scenario_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "doctor_model": doctor_model_id if doctor_llm.startswith("HF_") else doctor_llm,
            "patient_model": patient_model_id if patient_llm.startswith("HF_") else patient_llm,
            "doctor_bias": doctor_bias,
            "patient_bias": patient_bias,
            "ground_truth": scenario.diagnosis_information(),
            "predicted_diagnosis": final_diagnosis_str,
            "correct": correctness,
            "moderator_verdict": comparison_result,
            "turn_count": final_state.get("turn_count", 0),
            "differential_trajectory": final_state.get("differential_trajectory", []),
            "tests_ordered": final_state.get("tests_ordered", []),
            "metrics": metrics_report,
            "full_dialogue": final_state.get("full_dialogue", []),
            "duration_seconds": round(scenario_duration, 2),
        }

        with open(output_file, "a") as f:
            f.write(json.dumps(result_record, ensure_ascii=False) + "\n")

        # Update recency bias memory
        case_summary = summarize_case_for_recency(scenario)
        recent_case_summaries.append(case_summary)
        if len(recent_case_summaries) > MAX_RECENT_CASES:
            recent_case_summaries.pop(0)

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    final_accuracy = int((total_correct / total_presents) * 100) if total_presents > 0 else 0
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"  Scenarios Run:    {total_presents}")
    print(f"  Correct:          {total_correct}")
    print(f"  Accuracy:         {final_accuracy}%")
    print(f"  Results saved to: {output_file}")
    print(f"{'='*60}")

    # Final Notification
    if notify_topic:
        final_msg = f"EXP COMPLETE: {doctor_model_id.split('/')[-1]} finished {total_presents} scenarios. Accuracy: {final_accuracy}%"
        send_notification(notify_topic, final_msg)


# =============================================================================
# SECTION 9: CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AgentClinic v2.1 — LangGraph Clinical Decision-Making State Machine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Baseline homogeneous experiment (Mistral-7B for all roles):
  python agentic_clinic_v2.py \\
    --doctor_llm HF_mistralai/Mixtral-8x7B-v0.1 \\
    --patient_llm HF_mistralai/Mixtral-8x7B-v0.1 \\
    --measurement_llm HF_mistralai/Mixtral-8x7B-v0.1 \\
    --moderator_llm HF_mistralai/Mixtral-8x7B-v0.1 \\
    --doctor_model_id mistralai/Mistral-7B-Instruct-v0.2 \\
    --num_scenarios 107

  # Heterogeneous experiment (JSL-Med Doctor, Mistral Patient):
  python agentic_clinic_v2.py \\
    --doctor_llm HF_mistralai/Mixtral-8x7B-v0.1 \\
    --patient_llm HF_mistralai/Mixtral-8x7B-v0.1 \\
    --measurement_llm HF_mistralai/Mixtral-8x7B-v0.1 \\
    --moderator_llm HF_mistralai/Mixtral-8x7B-v0.1 \\
    --doctor_model_id johnsnowlabs/JSL-MedMistral-7B-v2.2 \\
    --patient_model_id mistralai/Mistral-7B-Instruct-v0.2 \\
    --num_scenarios 107

  # With recency bias:
  python agentic_clinic_v2.py \\
    --doctor_llm HF_mistralai/Mixtral-8x7B-v0.1 \\
    --patient_llm HF_mistralai/Mixtral-8x7B-v0.1 \\
    --measurement_llm HF_mistralai/Mixtral-8x7B-v0.1 \\
    --moderator_llm HF_mistralai/Mixtral-8x7B-v0.1 \\
    --doctor_model_id mistralai/Mistral-7B-Instruct-v0.2 \\
    --doctor_bias recency \\
    --num_scenarios 107
        """,
    )

    # --- LLM Backend Selection ---
    parser.add_argument('--openai_api_key', type=str,
                        default=os.environ.get("OPENAI_API_KEY", ""),
                        help='OpenAI API Key')
    parser.add_argument('--replicate_api_key', type=str, default="",
                        help='Replicate API Key')
    parser.add_argument('--doctor_llm', type=str, default='HF_mistralai/Mixtral-8x7B-v0.1',
                        help='Backend string for Doctor agent')
    parser.add_argument('--patient_llm', type=str, default='HF_mistralai/Mixtral-8x7B-v0.1',
                        help='Backend string for Patient agent')
    parser.add_argument('--measurement_llm', type=str, default='HF_mistralai/Mixtral-8x7B-v0.1',
                        help='Backend string for Measurement agent')
    parser.add_argument('--moderator_llm', type=str, default='HF_mistralai/Mixtral-8x7B-v0.1',
                        help='Backend string for Moderator (semantic judge)')

    # --- HuggingFace Model IDs (for heterogeneous architecture) ---
    parser.add_argument('--doctor_model_id', type=str,
                        default='mistralai/Mistral-7B-Instruct-v0.2',
                        help='HuggingFace model ID for Doctor agent')
    parser.add_argument('--patient_model_id', type=str,
                        default='mistralai/Mistral-7B-Instruct-v0.2',
                        help='HuggingFace model ID for Patient agent')
    parser.add_argument('--moderator_model_id', type=str, default=None,
                        help='HuggingFace model ID for Moderator (defaults to patient_model_id)')

    # --- Experiment Configuration ---
    parser.add_argument('--agent_dataset', type=str, default='MedQA',
                        help='Dataset to use (MedQA)')
    parser.add_argument('--num_scenarios', type=int, default=1,
                        help='Number of scenarios to simulate')
    parser.add_argument('--total_inferences', type=int, default=10,
                        help='Max turns per scenario')
    parser.add_argument('--scenario_id', type=int, default=None,
                        help='Run a single specific scenario ID')

    # --- Bias Configuration ---
    parser.add_argument('--doctor_bias', type=str, default='None',
                        choices=["None", "recency", "confirmation"],
                        help='Cognitive bias type for Doctor agent')
    parser.add_argument('--patient_bias', type=str, default='None',
                        choices=["None", "recency", "confirmation"],
                        help='Cognitive bias type for Patient agent')

    # --- Quantization ---
    parser.add_argument('--no_quantize', action='store_true',
                        help='Disable 4-bit quantization (use fp16)')

    # --- Output ---
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path for JSONL results output')
    parser.add_argument('--notify_topic', type=str, default=None,
                        help='ntfy.sh topic for mobile notifications')

    args = parser.parse_args()

    main(
        api_key=args.openai_api_key,
        replicate_api_key=args.replicate_api_key,
        doctor_llm=args.doctor_llm,
        patient_llm=args.patient_llm,
        measurement_llm=args.measurement_llm,
        moderator_llm=args.moderator_llm,
        num_scenarios=args.num_scenarios,
        dataset=args.agent_dataset,
        total_inferences=args.total_inferences,
        scenario_id=args.scenario_id,
        doctor_bias=args.doctor_bias,
        patient_bias=args.patient_bias,
        doctor_model_id=args.doctor_model_id,
        patient_model_id=args.patient_model_id,
        moderator_model_id=args.moderator_model_id,
        quantize=not args.no_quantize,
        output_file=args.output_file,
        notify_topic=args.notify_topic,
    )