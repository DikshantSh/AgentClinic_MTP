---
description: Humanize the Final Report LaTeX thesis to reduce AI-detection scores while preserving all technical content, structure, and meaning
---

# Humanize Final Report Workflow

## Step 1: Set Humanization Level

Before proceeding, the user should specify their desired level on this 1-5 scale:

| Level | Name | Description | AI Detection Risk |
|-------|------|-------------|-------------------|
| 1 | Light Polish | Fix only the most egregious AI tells (superlatives, filler transitions). Keep 90% of phrasing intact. | Medium-High |
| 2 | Moderate Rewrite | Replace formulaic openers, vary sentence structure, add occasional colloquial academic phrasing. Keep all content. | Medium |
| 3 | Substantial Reframe | Rewrite most paragraphs in a natural graduate-student voice. Introduce imperfect phrasing, shorter sentences, occasional hedging. Preserve all data, equations, figures. | Low-Medium |
| 4 | Deep Humanization | Rewrite from scratch using the same outline and data. Introduce personal research narrative ("we observed that...", "this was unexpected because..."). Vary paragraph length dramatically. | Low |
| 5 | Full Ghost-Rewrite | Complete rewrite as if dictated by the student after deeply understanding the material. Conversational academic tone, personal anecdotes about debugging, "we initially expected X but found Y". | Very Low |

**Ask the user**: "What level of humanization do you want? (1-5)"

## Step 2: Identify AI Tells in Current Report

Before rewriting, analyze each chapter for these common AI-generated patterns.
Use the dedicated prompt in Step 4 to perform the rewrite.

### Common AI Tells to Fix:

**Vocabulary tells:**
- "Furthermore", "Moreover", "Additionally" as paragraph openers (humans rarely chain these)
- "It is widely recognised that..." (hedging filler)  
- "This carries fundamental implications" (grandiose claims)
- "Critically" used as a sentence opener repeatedly
- "Notably", "Crucially", "Importantly" — overused intensifiers
- "Yields immense performance benefits" — superlatives
- "Constitutes a vector for" — unnecessarily formal  
- "Prohibitively expensive" — cliché
- "Bridging the final gap toward" — dramatic filler

**Structural tells:**
- Every paragraph starts with a topic sentence + expansion + conclusion (too uniform)
- Bullet points all have the same length
- Transitions are always explicit ("To address this...", "This result reveals that...")
- Every section ends with a forward-looking statement
- Sentences are all roughly the same length (15-25 words)

**Stylistic tells:**
- Heavy use of em-dashes for parenthetical remarks
- Passive voice throughout with zero first-person references
- No contractions, no colloquialisms
- Every claim is immediately backed by a qualifier
- Everything sounds confident and complete — no uncertainty

## Step 3: Chapter-by-Chapter Rewrite Order

Process files in this order (dependency order — later chapters reference earlier ones):

1. `Chapters/1_Introduction.tex`
2. `Chapters/2_Literature_Review.tex`
3. `Chapters/3_Dataset.tex`
4. `Chapters/4_Initial_Framework.tex`
5. `Chapters/5_Enhanced_Methodology.tex`
6. `Chapters/6_Results.tex`
7. `Chapters/7_Discussion.tex`
8. `Chapters/8_Conclusion.tex`

**Rules during rewrite:**
- NEVER change any numerical values, equations, table data, figure references, or citation keys
- NEVER change section/subsection/chapter titles (these are referenced elsewhere)
- NEVER remove or add \label{} commands
- NEVER modify tikzpicture/figure environments
- ONLY modify the prose text inside paragraphs and \item descriptions
- Preserve all LaTeX formatting commands (\textbf, \textit, \texttt, etc.)

## Step 4: Use the Humanization Prompt

// turbo
Read the prompt file at:
`/home/ai21im3ai29/mtp-agentclinic/AgentClinic/.agents/workflows/humanize_prompt.md`

Apply it to each chapter file sequentially.

## Step 5: Compile and Verify

// turbo
After all chapters are rewritten:
```bash
cd /home/ai21im3ai29/mtp-agentclinic/AgentClinic/Final_Report
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
grep -c "^!" main.log
```

Verify 0 LaTeX errors and that the PDF size is comparable to the original.
