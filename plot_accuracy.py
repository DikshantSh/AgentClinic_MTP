import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── Data from Chapter 6 / Table 2 of Final_Report_Revised ──────────────────
labels   = ['E1\nMistral-7B\n(Baseline)', 'E2\nJSL-Med\n(Domain FT)', 'E3\nLlama-3.1-8B\n(Reasoning)']
accuracy = [45.8, 50.5, 51.4]

# Colour palette that matches the report aesthetic
COLORS = ['#4C72B0', '#55A868', '#C44E52']

fig, ax = plt.subplots(figsize=(9, 6))

bars = ax.bar(labels, accuracy, color=COLORS, width=0.5,
              edgecolor='white', linewidth=1.2, zorder=3)

# Value labels above each bar
for bar, val in zip(bars, accuracy):
    ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.6,
            f'{val:.1f}%', ha='center', va='bottom',
            fontsize=13, fontweight='bold', color='#2d2d2d')

# Horizontal reference line at Phase-1 baseline (43.9%)
ax.axhline(43.9, color='grey', linewidth=1.4, linestyle='--', zorder=2)
ax.text(2.42, 44.5, 'Phase 1 baseline (43.9%)',
        ha='right', va='bottom', fontsize=9, color='grey', style='italic')

ax.set_ylim(0, 62)
ax.set_ylabel('Diagnostic Accuracy (%)', fontsize=12, labelpad=8)
ax.set_title('Diagnostic Accuracy — Unbiased Experiments E1–E3\n(AgentClinic-MedQA, 107 Scenarios)',
             fontsize=13, fontweight='bold', pad=14)

ax.yaxis.grid(True, linestyle='--', alpha=0.6, zorder=0)
ax.set_axisbelow(True)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()

out_dir = '/home/ai21im3ai29/mtp-agentclinic/AgentClinic/Final_Report_Revised/Pictures'
png_path = os.path.join(out_dir, 'phase2_acc.png')
pdf_path = os.path.join(out_dir, 'phase2_acc.pdf')
plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, bbox_inches='tight')
print(f"Saved:\n  {png_path}\n  {pdf_path}")
