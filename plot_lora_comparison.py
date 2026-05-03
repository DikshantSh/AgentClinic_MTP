import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ── Data from Chapter 6, Table 3 ───────────────────────────────────────────
metrics    = ['Accuracy (%)', 'Diagnostic\nStability', 'Test\nRationality', 'Information\nEfficiency']
e2_vals    = [50.5,  0.68,  0.71,  0.98]
e6_vals    = [55.1,  0.72,  0.76,  1.02]
y_limits   = [(40, 62), (0, 1.0), (0, 1.0), (0, 1.3)]

E2_COLOR = '#4C72B0'   # blue
E6_COLOR = '#DD8452'   # orange

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.patch.set_facecolor('#FAFAFA')

for i, (ax, metric, v2, v6, ylim) in enumerate(
        zip(axes, metrics, e2_vals, e6_vals, y_limits)):

    x = [0.3, 0.7]
    bars = ax.bar(x, [v2, v6], width=0.28,
                  color=[E2_COLOR, E6_COLOR],
                  edgecolor='white', linewidth=1.5,
                  zorder=3)

    # Value labels
    for bar, val in zip(bars, [v2, v6]):
        label = f'{val:.0f}%' if i == 0 else f'{val:.2f}'
        ax.text(bar.get_x() + bar.get_width() / 2, val + ylim[1] * 0.02,
                label, ha='center', va='bottom',
                fontsize=13, fontweight='bold', color='#2d2d2d')

    # Delta arrow annotation
    delta = v6 - v2
    delta_str = f'+{delta:.1f}%' if i == 0 else f'+{delta:.2f}'
    ax.annotate('', xy=(0.7, v6), xytext=(0.7, v2),
                arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=2.5))
    ax.text(0.82, (v2 + v6) / 2, delta_str,
            ha='left', va='center', fontsize=11,
            fontweight='bold', color='#2ca02c')

    ax.set_xlim(0, 1.05)
    ax.set_ylim(ylim)
    ax.set_xticks([0.3, 0.7])
    ax.set_xticklabels(['E2\n(JSL-Med)', 'E6\n(JSL-Med\n+LoRA)'], fontsize=11)
    ax.set_title(metric, fontsize=13, fontweight='bold', pad=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_facecolor('#FAFAFA')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=E2_COLOR, label='E2 — JSL-Med (Base)'),
    Patch(facecolor=E6_COLOR, label='E6 — JSL-Med + Dynamic LoRA'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=2,
           fontsize=12, frameon=True, bbox_to_anchor=(0.5, -0.05))

fig.suptitle('E2 vs E6: Impact of Dynamic LoRA Adapter Routing',
             fontsize=15, fontweight='bold', y=1.02)

plt.tight_layout()

out_dir = '/home/ai21im3ai29/mtp-agentclinic/AgentClinic/Final_Report_Revised/Pictures'
png_path = os.path.join(out_dir, 'lora_comparison.png')
pdf_path = os.path.join(out_dir, 'lora_comparison.pdf')
plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, bbox_inches='tight')
print(f"Saved:\n  {png_path}\n  {pdf_path}")
