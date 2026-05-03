# Prompt: Generate Detailed Academic Speaker Notes for M.Tech Thesis Viva

## Role

You are an expert academic presentation coach and a senior researcher in AI and NLP. You are helping an M.Tech student at IIT Kharagpur prepare **detailed, convincing, viva-ready speaker notes** for their thesis defense presentation on "Evaluating Large Language Models for Clinical Decision-Making Using Multi-Agent Simulations."

## Objective

For **every single slide** in the provided Beamer presentation (main.tex + slides_part2.tex), produce a self-contained speaker notes block. These notes are what the student will rehearse and deliver verbally during the 30–40 minute viva. The output must be a **compilable LaTeX document** saved as `speaker_notes_detailed.tex`.

---

## Output Format & Structure

Use this exact LaTeX structure for the output file:

```latex
\documentclass[12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{geometry}
\geometry{a4paper, margin=0.9in}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{hyperref}
\usepackage{mdframed}

\definecolor{spokenblue}{RGB}{0,51,102}
\definecolor{techgray}{RGB}{80,80,80}
\definecolor{qared}{RGB}{180,40,40}
\definecolor{transgreen}{RGB}{30,120,60}

\newmdenv[
  linecolor=spokenblue,
  linewidth=1.5pt,
  backgroundcolor=spokenblue!3,
  roundcorner=5pt,
  innertopmargin=8pt,
  innerbottommargin=8pt
]{spokenbox}

\newmdenv[
  linecolor=qared,
  linewidth=1pt,
  backgroundcolor=qared!3,
  roundcorner=4pt,
  innertopmargin=6pt,
  innerbottommargin=6pt
]{qabox}

\title{Comprehensive Speaker Notes\\AgentClinic v2.1 — M.Tech Thesis Viva}
\author{Dikshant Sharma — IIT Kharagpur}
\date{April 2026}

\begin{document}
\maketitle
\tableofcontents
\newpage

% === FOR EACH SLIDE, USE THIS TEMPLATE: ===

\section{Slide N: [Slide Title]}

\subsection*{Timing}
\textit{Target: X minutes | Cumulative: Y minutes}

\subsection*{Spoken Script}
\begin{spokenbox}
[Write the EXACT words the student should say aloud. Written in first person. Natural academic tone — confident but not arrogant. Use full sentences, not bullet points. Include pause markers like [PAUSE] and gesture cues like [POINT TO DIAGRAM].]
\end{spokenbox}

\subsection*{Technical Deep-Dive (Internal Reference)}
{\color{techgray}
[Detailed technical explanation the student must KNOW but does NOT say unless asked. Include: underlying math, implementation details from the codebase, design decisions and their rationale, exact file/function references, numerical values and their derivations.]
}

\subsection*{Transition to Next Slide}
{\color{transgreen}
\textit{[One sentence that naturally bridges to the next slide's topic.]}
}

\subsection*{Anticipated Examiner Questions}
\begin{qabox}
\textbf{Q1:} [Likely examiner question] \\
\textbf{A1:} [Prepared, confident, technically precise answer — 3–5 sentences] \\[6pt]
\textbf{Q2:} [Another likely question] \\
\textbf{A2:} [Answer]
\end{qabox}

\end{document}
```

---

## Detailed Instructions for Each Section

### 1. Spoken Script (the most important section)

- **Write EXACTLY what the student should say out loud** — full natural sentences in the first person.
- **Tone:** Confident, knowledgeable, academically precise. Not robotic. Not overly casual. Sound like a researcher who built this system and deeply understands every line of code.
- **Length:** Each slide's spoken script should be **150–300 words** (roughly 1–2 minutes of speaking).
- **Opening slides** (Title, Outline) should be brief (~30 seconds each).
- **Core technical slides** (Architecture, Reflection Node, Metrics, Results) should be the longest and most detailed (~2–3 minutes each).
- **Include delivery cues:**
  - `[PAUSE]` — for dramatic effect or to let a key point land
  - `[POINT TO LEFT COLUMN]`, `[GESTURE TO DIAGRAM]`, `[INDICATE TABLE ROW]` — reference on-screen content
  - `[SLOW DOWN]` — for especially important statements
  - `[MAKE EYE CONTACT]` — before key claims
- **Never just read out bullet points from the slide.** The spoken script must ADD context, narrative, and insight beyond what the slide already shows.
- **Use storytelling structure:** Problem → What I did → Why it matters → What it revealed.
- **Weave in specific numbers** naturally: "...accuracy jumped from 43.9 to 45.8 percent, which is remarkable because we simultaneously *removed* an artificial advantage..."
- **Acknowledge limitations proactively** where relevant — this shows intellectual maturity.

### 2. Technical Deep-Dive

- This section is the student's **private reference sheet** — material they must internalize but only articulate if questioned.
- Include:
  - **Mathematical formulations** (e.g., the exact Jaccard formula, the POMDP tuple, LoRA decomposition W = W₀ + BA)
  - **Implementation details** from the actual codebase (e.g., "The `_clean_dialogue_response()` function in `agentic_clinic_v2.py` line 92 uses regex to strip `<think>` tags that DeepSeek models leak")
  - **Design decision rationale** (e.g., "Temperature 0.1 was chosen for the reflection node because structured JSON generation requires near-deterministic sampling, while dialogue uses 0.7 for natural variation")
  - **Exact numerical results** with their n/107 derivations where applicable
  - **Hardware specifics** (H100 NVL, 95.8 GB VRAM, NF4 reducing 14 GB to ~4 GB per model)
  - **Why alternatives were rejected** (e.g., "AutoGen was considered but rejected because it assumes pre-built agent roles; LangGraph provides raw graph execution without opinionated abstractions")

### 3. Transition Sentences

- Must feel **natural and conversational**, not mechanical.
- Should create a logical narrative arc across the entire presentation.
- Examples of good transitions:
  - "Having established that static benchmarks are insufficient, let me now show you the research questions that guided my investigation."
  - "This limitation in Phase 1 is precisely what motivated the complete architectural redesign I'll describe next."
  - "Now that we've seen the raw accuracy numbers, let me show you something far more interesting — what happens when we look at *how* the model reasoned, not just what it concluded."

### 4. Anticipated Examiner Q&A

- For each slide, provide **2–4 likely examiner questions** with **pre-prepared answers**.
- Questions should range from:
  - **Clarification questions:** "Why did you choose X over Y?"
  - **Probing questions:** "What happens if the model doesn't produce valid JSON?"
  - **Critical challenges:** "Isn't your DS metric flawed because it uses exact string matching?"
  - **Extension questions:** "How would this generalize beyond MedQA?"
  - **Foundational questions:** "Can you explain what a POMDP is in simple terms?"
- **Answers must be:**
  - Concise (3–5 sentences)
  - Technically precise
  - Honest about limitations ("That's a valid concern. In this thesis, we used X because Y, but I acknowledge that Z would be a stronger approach and is planned as future work.")
  - Never defensive or dismissive

---

## Slide-by-Slide Content Requirements

The presentation has approximately **26 slides** across `main.tex` and `slides_part2.tex`. Below are the specific content expectations for key slides:

| Slide | Title | Key Content the Notes MUST Cover |
|-------|-------|----------------------------------|
| 1 | Title | Brief, confident opening. State name, department, thesis title. |
| 2 | Outline | Quick overview of what's coming. Set expectations for the committee. |
| 3 | The Problem | Why static benchmarks ≠ clinical reasoning. The fundamental gap. |
| 4 | Why This Matters | AMIE, Agent Hospital, MedAgents limitations. Lexical Leakage preview. |
| 5 | Research Questions | RQ1–RQ4. Map each to specific experiments. |
| 6 | POMDP Formulation | Explain the theoretical foundation clearly. Define each tuple element. |
| 7 | NF4 & LoRA | Technical enablers. Memory math. Why these were necessary. |
| 8 | Phase 1 Architecture | Flowchart walkthrough. The while-loop problem. Catastrophic forgetting. |
| 8b | Phase 1 Results Chart | Read and interpret the accuracy numbers. Bias preliminary results. |
| 9 | Lexical Leakage | The original discovery. Shared embedding manifold. Why it inflates accuracy. |
| 10 | Phase 1→2 Transition | Three showstoppers. Why a complete redesign was necessary. |
| 11 | LangGraph DCG | Full architecture deep dive. Node definitions. TypedDict state. Routing. |
| 12 | Metacognitive Reflection | Most important contribution. Decoupling reasoning from generation. |
| 13 | Heterogeneous Engines | Multi-engine architecture. Memory math. Cognitive asymmetry. |
| 14 | Engineering Deep-Dive | JSON fallback. Response cleaning. Forced diagnosis. Practical robustness. |
| 15 | Process Metrics | DS, TR, IE formulations. What each measures and why it matters. |
| 16 | Experiment Matrix | E1–E6 setup. Why each experiment exists. Control variables. |
| 17 | Phase 2 Accuracy | Read and interpret results. The metacognition win. Domain vs alignment. |
| 18 | Model Comparison | Visual summary. Connect to broader patterns. |
| 19 | Process Metrics Results | Table 6.3 deep dive. The E5 "confidently wrong" finding. |
| 20 | Cognitive Bias Impact | E4 vs E5 mechanisms. Attention-based explanation. Clinical danger. |
| 21 | Bias Failure Modes | Specific behavioral pathologies. Safety implications. |
| 22 | Dynamic LoRA | The fine-tuning pipeline. SDG, QLoRA, dynamic swapping. +4.6pp result. |
| 23 | Limitations | Honest self-critique of DS, TR, IE. Single dataset. Text-only modality. |
| 24 | Future Work | DPO, RLAIF, debiasing adapters, multimodal. All build on existing infra. |
| 25 | Contributions Summary | C1–C5. The overarching thesis message. |
| 26 | Thank You | Graceful close. Final Q&A prep. |

---

## Global Constraints

1. **Total presentation time budget:** 30–35 minutes for the spoken portion. Allocate time per slide accordingly.
2. **Academic register:** This is an IIT Kharagpur M.Tech thesis defense. The language must be formal but not stiff. No slang. No filler words like "basically" or "so yeah."
3. **First person singular:** The student did this work. Use "I designed...", "I identified...", "My experiments showed..."
4. **Cite related work naturally:** When mentioning AMIE, Agent Hospital, MedAgents, BioGPT, LoRA (Hu et al.), NF4 (Dettmers et al.), weave citations into speech: "Building on Dettmers et al.'s NormalFloat4 quantization..."
5. **Numbers must be exact:** Use the actual results from the thesis (43.9%, 45.8%, 50.5%, 51.4%, 38.3%, 31.8%, 55.1%).
6. **The narrative arc should be:** Crisis in clinical AI evaluation → My Phase 1 attempt → Fundamental flaws discovered → Complete rebuild (Phase 2) → Novel architecture + metrics → Rigorous experiments → Key findings → Limitations & future → Contribution summary.
7. **The document must compile cleanly** with pdflatex without errors.

---

## What Makes These Notes "Convincing" to an Examiner

- The student demonstrates they **understand every design decision**, not just the output.
- The student can **connect implementation details to theoretical motivation** (e.g., "I used temperature 0.1 here because the reflection node requires near-deterministic JSON extraction, aligned with the POMDP's observation function").
- The student **preemptively addresses weaknesses** rather than being caught off-guard.
- The student uses **precise technical vocabulary** correctly (e.g., "catastrophic forgetting," "premature closure," "Jaccard similarity," "attention heads," "softmax distribution").
- The student shows **intellectual ownership** — they didn't just run code, they made deliberate architectural decisions and can justify each one.


keep detiallic, and also keep many extra details for each slide in speaker notes, which may also look like internal study guide