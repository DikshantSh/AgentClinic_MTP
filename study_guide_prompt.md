# AI Prompt: Generate an Extreme-Detail Internal Study Guide

> **Copy everything below this line and paste it into an AI assistant (Claude, ChatGPT, Gemini, etc.)**

---

You are a world-class academic tutor and technical writer. I need you to create an **extreme-detail internal study guide** for my M.Tech thesis titled:

**"Evaluating Large Language Models for Clinical Decision-Making Using AgentClinic"**
*Author: Dikshant Sharma | IIT Kharagpur | Dept. of AI | April 2026*

The guide must serve as a **complete self-study document** that I can use to prepare for a thesis viva/defence. It must cover EVERY concept — theoretical, technical, and experimental — with both **rigorous academic explanations** AND **simple intuitive "explain-like-I'm-12" analogies**.

---

## THESIS STRUCTURE & CONTENT TO COVER

### Chapter 1: Introduction
- Why static medical QA benchmarks (MedQA, USMLE, PubMedQA) are insufficient for evaluating clinical AI
- The difference between **one-shot factual recall** vs **interactive, sequential, multi-turn clinical reasoning**
- What is **Lexical Leakage** — explain the concept, why it's dangerous, and give a simple analogy (e.g., an exam where the student and answer-key-maker share the same brain)
- Problem statement: 4 key research gaps this thesis addresses
- Contributions: List all 5 contributions with one-liner explanations

### Chapter 2: Literature Review
- **MedQA / USMLE benchmarks**: What they test, what they miss
- **Dialogue-based simulations**: AMIE (Google), Agent Hospital — what they do, their limitations
- **Cognitive bias in clinical AI**: Anchoring bias, confirmation bias, recency bias — define each with a medical example
- **Open-source LLMs**: Llama-3, Mistral-7B, Mixtral — parameter counts, architectures (decoder-only transformers)
- **Quantization techniques**: What is NF4 quantization? What is double quantization? Why is it needed? Walk through the math of how a 7B model goes from 14 GB (FP16) to ~4 GB (NF4). Explain QLoRA.
- **Research gaps** identified — summarize the 5 gaps

### Chapter 3: Dataset — AgentClinic-MedQA
- What is an **OSCE (Objective Structured Clinical Examination)**? How does the dataset simulate it?
- The JSON schema: `Objective_for_Doctor`, `Patient_Actor`, `Test_Results`, `Physical_Examination_Findings`, `Correct_Diagnosis`
- **Knowledge partitioning**: Why each agent only sees its own slice of the data. Give an analogy.
- **Semantic evaluation by the Moderator Agent**: Why exact string matching fails (e.g., "Heart Attack" vs "Acute Myocardial Infarction"). How the Moderator solves this.
- Total scenarios: 107

### Chapter 4: Initial Framework (Phase 1)
- Architecture: Ad-hoc `while`-loop orchestrator
- Models tested: Mistral-7B (44%), BioGPT (35%), JSL-Med (32%), Apollo (23%)
- **Catastrophic forgetting**: Why medically fine-tuned models (BioGPT, JSL-Med) *underperformed* the generalist Mistral-7B. Explain with the concept of narrow SFT destroying general reasoning. Give an analogy (a chess player who studies only one opening forgets how to play the middlegame).
- Bias experiments: Recency (34%), Gender (35%), Confirmation (32%)
- **3 critical architectural flaws identified**: (1) Non-deterministic execution, (2) No trajectory analysis, (3) Lexical Leakage

### Chapter 5: Enhanced Methodology — AgentClinic v2.1
For EVERY concept below, provide: (a) Academic definition, (b) Simple analogy, (c) Why it matters for this thesis.

- **LangGraph Directed Cyclic Graph (DCG)**: What is a state machine? What is LangGraph? Why a cyclic graph and not a DAG? Draw out the flow: `doctor_reflection → doctor_speaker → {patient | measurement | END} → back to doctor_reflection`.
- **Heterogeneous Multi-Engine Architecture**: Why the Doctor, Patient, and Moderator use different model weights. How this kills Lexical Leakage. VRAM budget math.
- **4-bit NF4 Quantization with Double Quantization**: Full technical walkthrough. What is NormalFloat4? Why are NF4 bins non-linearly spaced? What does "double quantization" mean (quantizing the quantization constants)? Memory savings calculation.
- **Flash Attention 2**: What problem does standard attention have (O(n²) memory)? How Flash Attention fixes it.
- **Latent Metacognitive Reflection Node (`doctor_reflection`)**: The core innovation. The Doctor LLM is forced to output a JSON array of differential diagnoses BEFORE generating dialogue. Explain the "think before you speak" principle. Why this makes internal reasoning observable and graphable.
- **ClinicalState TypedDict**: Explain every field: `last_input_to_doctor`, `last_doctor_message`, `differential_diagnoses`, `turn_count`, `diagnosis_ready`, `differential_trajectory`, `tests_ordered`, `full_dialogue`.
- **Deterministic Routing via `route_doctor()`**: How the state machine routes to patient/measurement/END based on the doctor's output. Why this prevents hallucinated workflow violations.
- **Cognitive Bias Injection Manifold**: Recency bias (k=5 case memory queue), Confirmation bias (strong prior hypothesis injection). How each exploits the transformer's attention mechanism.
- **Task-Specific Dynamic LoRA Adapter Routing**: What is LoRA? The math: ΔW = BA where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r << min(d,k). What is PEFT? How two adapters (reasoning + dialogue) are loaded simultaneously and switched at inference time using `model.set_adapter()`. Rank r=16, α=32. VRAM overhead ~50 MB.
- **Theoretical Underpinnings**:
  - Decoupling Reasoning from Generation (Latent Chain-of-Thought): The math of P(w_t | w_1,...,w_{t-1}) and why self-attention heads split capacity.
  - Embedding Manifold Overlap and Lexical Leakage: Shared θ_shared causes zero-shot knowledge transfer.
  - Information-Theoretic Basis of NF4: Gaussian weight distributions and optimal bin placement.

### Chapter 6: Experiments and Results
Present EVERY experiment with full metrics. Create tables.

| Exp | Doctor | Bias | Acc% | DS | TR | IE |
|-----|--------|------|------|-----|-----|-----|
| E1 | Mistral-7B | None | 45.6 | 0.56 | 0.63 | 1.13 |
| E2 | JSL-MedMistral | None | 50.2 | 0.68 | 0.71 | 0.98 |
| E3 | Llama-3.1-8B | None | 53.5 | 0.65 | 0.79 | 1.05 |
| E4 | JSL-MedMistral | Recency | 38.1 | 0.45 | 0.58 | 1.34 |
| E5 | JSL-MedMistral | Confirmation | 31.4 | 0.88 | 0.41 | 0.61 |
| E6 | JSL-Med + LoRA | None | 56.7 | 0.74 | 0.78 | 1.01 |

For each metric, explain:
- **Diagnostic Stability (DS)**: Jaccard similarity of consecutive differential diagnoses. What does DS=0.56 mean vs DS=0.88? Why is high DS with low accuracy (E5) dangerous?
- **Test Rationality (TR)**: Did the model order basic tests (CBC, vitals) before advanced tests (MRI, biopsy)? Binary penalty.
- **Information Efficiency (IE)**: Data points gathered per turn. Why E4's IE=1.34 is paradoxically high (over-testing due to bias panic).

Key findings to explain:
1. Why E1 accuracy (45.6%) *increased* from Phase 1 (44%) despite removing Lexical Leakage
2. Why E3 > E2 (instruction alignment vs domain fine-tuning)
3. E4 Recency Bias: "hypothesis thrashing" — low DS, low accuracy
4. E5 Confirmation Bias: "premature diagnostic closure" — pathologically high DS, lowest accuracy
5. E6 Dynamic LoRA: +6.5% over E2 base, highest overall accuracy

### Chapter 7: Discussion
- **Structured metacognition eclipses Lexical Leakage**: Why this is the most important finding
- **Domain Specialisation vs Instruction Alignment**: The tension, and how LoRA resolves it
- **Dynamic LoRA resolving the trade-off**: E6 proves that separating reasoning from dialogue at the adapter level works
- **Safety threat of cognitive biases**: Why E5 (high stability + low accuracy) is the most dangerous failure mode in clinical deployment
- **Implications for clinical AI evaluation standards**: Why binary accuracy alone is dangerous
- **Limitations**: Text-only measurement agent, single dataset (MedQA)

### Chapter 8: Conclusion & Future Work
- Summarize all findings in 5 bullet points
- **Future Work**:
  - Process Reward Modelling via DPO (Direct Preference Optimisation): What is DPO? How it differs from RLHF. How trajectory pairs (chosen vs rejected) would be generated using ClinicalTrajectoryEvaluator.
  - RL from AI Feedback via Self-Play: Using AgentClinic as an RL environment. ClinicalTrajectoryEvaluator as the reward function. PPO training.
  - Extending LoRA to Bias Resilience: Debiasing adapters. Four-adapter routing manifold.
  - Multimodal Integration: LLaVA-Med, Med-Gemini for visual data (ECGs, X-rays).

---

## FORMAT REQUIREMENTS

1. **Structure**: Use the chapter/section hierarchy above. Add a Table of Contents at the start.
2. **For every technical concept**, provide:
   - 📘 **Academic Definition** (rigorous, cite-ready)
   - 🧠 **Simple Analogy** (explain it to a 12-year-old or a non-CS person)
   - 💡 **Why It Matters for This Thesis** (1-2 sentences connecting it to the specific experiments)
3. **Include worked examples** wherever possible:
   - Example Jaccard similarity calculation for DS
   - Example test ordering sequence for TR
   - Example NF4 memory calculation
   - Example LoRA parameter count calculation
4. **Potential viva questions**: At the end of each chapter section, list 5-10 likely viva/defence questions with concise model answers.
5. **Use tables, bullet points, and bold/italic** liberally for scannability.
6. **Mathematical formulas**: Include all key equations (DS Jaccard, LoRA decomposition, NF4 quantization, accuracy formula).
7. **Length**: Be EXHAUSTIVE. This should be 8000-12000 words. Do NOT summarize or skip sections. Cover EVERYTHING.
8. **Tone**: Academic but accessible. Think "Stanford lecture notes" — rigorous yet clear.

---

**IMPORTANT**: This is for internal study ONLY, not for submission. Be as detailed and explanatory as possible. I need to understand every concept deeply enough to answer any question a professor might ask during a viva defence.
