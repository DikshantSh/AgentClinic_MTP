---
description: Create an extreme-detail internal LaTeX study guide for the AgentClinic M.Tech thesis viva preparation
---

# Create Internal Study Guide

This workflow produces a standalone LaTeX study guide at `Study_Guide/` based on `study_guide_prompt.md` and the thesis content in `Final_Report/`.

## Prerequisites
- `study_guide_prompt.md` exists with the chapter outline and format requirements
- `Final_Report/Chapters/*.tex` contain the thesis content to extract data/metrics from
- LaTeX toolchain (`pdflatex`) is available

## Steps

1. **Read the study guide prompt** to understand the exact chapter structure, format requirements (academic definition, analogy, why-it-matters boxes), and content expectations.

2. **Read all 8 thesis chapters** from `Final_Report/Chapters/` to extract accurate metrics, formulas, and experimental data.

3. **Create directory structure**:
// turbo
```bash
mkdir -p Study_Guide/Chapters
```

4. **Create `Study_Guide/main.tex`** — the root document with:
   - Standalone article class (not Thesis.cls)
   - tcolorbox environments for: Academic Definition (blue), Simple Analogy (green), Why It Matters (red), Viva Questions (purple), Worked Examples (orange)
   - Title page with "Internal Study — Not for Submission"
   - Table of Contents
   - `\input{}` for each chapter file

5. **Create chapter files one-by-one** (each ~2-4 pages):

   - `Chapters/ch1_introduction.tex` — Static benchmarks limitations, one-shot vs multi-turn, Lexical Leakage definition + analogy, 4 research gaps, 5 contributions
   - `Chapters/ch2_literature_review.tex` — MedQA/USMLE, AMIE/Agent Hospital, cognitive biases (anchoring/confirmation/recency), open-source LLMs (Llama-3/Mistral/Mixtral architectures), NF4 quantization math walkthrough (FP16→NF4), QLoRA, 5 research gaps
   - `Chapters/ch3_dataset.tex` — OSCE definition, JSON schema fields, knowledge partitioning, semantic evaluation by Moderator, 107 scenarios
   - `Chapters/ch4_initial_framework.tex` — While-loop architecture, 4 models tested (Mistral 44%, BioGPT 35%, JSL-Med 32%, Apollo 23%), catastrophic forgetting explanation, bias experiments (recency 34%, gender 35%, confirmation 32%), 3 architectural flaws
   - `Chapters/ch5_enhanced_methodology.tex` — LangGraph DCG, state machine, metacognitive reflection node, heterogeneous multi-engine, NF4+double quant math, Flash Attention 2, ClinicalState TypedDict fields, route_doctor(), cognitive bias injection manifold, LoRA math (ΔW=BA), PEFT dual adapters, theoretical underpinnings (latent CoT, embedding overlap, NF4 info theory)
   - `Chapters/ch6_results.tex` — Full experiment table (E1-E6), metric definitions (DS/TR/IE) with worked examples, key findings with explanations
   - `Chapters/ch7_discussion.tex` — Metacognition > Lexical Leakage, domain vs alignment, LoRA resolution, bias safety threat, evaluation standards, limitations
   - `Chapters/ch8_conclusion.tex` — 5-bullet summary, future work (DPO, RL self-play, debiasing adapters, multimodal)

6. **Each section must include** (per study_guide_prompt.md):
   - 📘 Academic Definition box
   - 🧠 Simple Analogy box
   - 💡 Why It Matters box
   - Worked examples where applicable
   - 5-10 viva questions with model answers at end of each chapter

7. **Compile the PDF**:
// turbo
```bash
cd Study_Guide && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
```

8. **Verify** the PDF compiled without errors and all chapters appear in the TOC.
