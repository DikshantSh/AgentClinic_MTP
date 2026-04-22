import matplotlib.pyplot as plt
import numpy as np
import os

# Create Pictures directory if it doesn't exist
os.makedirs('Pictures', exist_ok=True)

# Common settings
plt.rcParams.update({'font.size': 14})

def plot_phase1_acc():
    labels = ['Mistral-7B', 'BioGPT', 'JSL-Med', 'Apollo']
    values = [44, 35, 32, 23]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, width=0.5, color=['#4C72B0', '#DD8452', '#55A868', '#C44E52'], edgecolor='black', alpha=0.9)
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Diagnostic Accuracy on AgentClinic-MedQA (Initial Framework)')
    ax.set_ylim(0, 60)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., 1.01 * height,
                '%d' % int(height), ha='center', va='bottom')
                
    plt.tight_layout()
    plt.savefig('Pictures/phase1_acc.pdf')
    plt.close()

def plot_phase2_acc():
    labels = ['E1', 'E2', 'E3', 'E4', 'E5']
    values = [45.6, 54.2, 58.5, 38.1, 31.4]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, width=0.5, color=['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3'], edgecolor='black', alpha=0.9)
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('AgentClinic v2.1 Diagnostic Accuracy by Experiment')
    ax.set_ylim(0, 70)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., 1.01 * height,
                '%.1f' % height, ha='center', va='bottom')
                
    plt.tight_layout()
    plt.savefig('Pictures/phase2_acc.pdf')
    plt.close()

def plot_ds_metric():
    labels = ['E1 (Mistral)', 'E2 (JSL-Med)', 'E3 (Llama-3.1)']
    values = [0.56, 0.71, 0.68]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, width=0.4, color=['#4C72B0', '#DD8452', '#55A868'], edgecolor='black', alpha=0.9)
    
    ax.set_ylabel('Diagnostic Stability (DS)')
    ax.set_title('Diagnostic Stability across Model Configurations')
    ax.set_ylim(0, 1.0)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., 1.02 * height,
                '%.2f' % height, ha='center', va='bottom')
                
    plt.tight_layout()
    plt.savefig('Pictures/ds_metric.pdf')
    plt.close()

def plot_bias_combined():
    labels = ['E2 (Unbiased)', 'E4 (Recency)', 'E5 (Confirm.)']
    acc_values = [54.2, 38.1, 31.4]
    ds_values = [71.0, 45.0, 88.0]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, acc_values, width, label='Diagnostic Accuracy (%)', color='#4C72B0', edgecolor='black', alpha=0.9)
    rects2 = ax.bar(x + width/2, ds_values, width, label='Diagnostic Stability (x 100)', color='#DD8452', edgecolor='black', alpha=0.9)
    
    ax.set_ylabel('Metric Score')
    ax.set_title('Effect of Cognitive Bias on Accuracy and Diagnostic Stability')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    
    plt.tight_layout()
    plt.savefig('Pictures/bias_combined.pdf', bbox_inches='tight')
    plt.close()

def plot_langgraph():
    # Since we might consider it a "plot of latex" (tikzpicture), we can generate a simple visual representation
    # However, it's a node diagram. I will generate an image using matplotlib patches to mimic it.
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # Coordinates
    reflection_pos = (2, 4)
    speaker_pos = (7, 4)
    patient_pos = (2, 1)
    measurement_pos = (7, 1)
    end_pos = (9.5, 4)
    
    # Def blocks
    def add_node(ax, xy, text, bg_color='white'):
        box = mpatches.FancyBboxPatch((xy[0]-1.5, xy[1]-0.5), 3.0, 1.0, boxstyle="round,pad=0.1", 
                                      ec="black", fc=bg_color, lw=2)
        ax.add_patch(box)
        ax.text(xy[0], xy[1], text, ha='center', va='center', weight='bold')
    
    add_node(ax, reflection_pos, "doctor_reflection\n(Latent DDx)", '#d3d3d3')
    add_node(ax, speaker_pos, "doctor_speaker\n(Utterance)", 'white')
    add_node(ax, patient_pos, "patient_node", 'white')
    add_node(ax, measurement_pos, "measurement_node", 'white')
    
    # END node is smaller
    box = mpatches.FancyBboxPatch((end_pos[0]-0.8, end_pos[1]-0.5), 1.6, 1.0, boxstyle="round,pad=0.1", 
                                  ec="black", fc='#a9a9a9', lw=2)
    ax.add_patch(box)
    ax.text(end_pos[0], end_pos[1], "END", ha='center', va='center', weight='bold')
    
    # Arrows and text
    def add_arrow(ax, start, end, text, offset=(0,0)):
        ax.annotate(text, xy=end, xytext=start,
                    arrowprops=dict(facecolor='black', shrink=0.1, width=1.5, headwidth=8),
                    ha='center', va='center')
                    
    # The annotations are simpler to just draw with arrows
    # Reflection -> Speaker
    ax.annotate("Updates State", xy=(speaker_pos[0]-1.6, speaker_pos[1]), xytext=(reflection_pos[0]+1.6, reflection_pos[1]),
                arrowprops=dict(facecolor='black', width=1.5, headwidth=8), ha='center', va='bottom')
                
    # Speaker -> Measurement
    ax.annotate("REQUEST TEST", xy=(measurement_pos[0], measurement_pos[1]+0.6), xytext=(speaker_pos[0], speaker_pos[1]-0.6),
                arrowprops=dict(facecolor='black', width=1.5, headwidth=8), ha='center', va='center', rotation=-90)

    # Speaker -> Patient
    ax.annotate("Ask Question", xy=(patient_pos[0]+1.6, patient_pos[1]+0.5), xytext=(speaker_pos[0]-1.6, speaker_pos[1]-0.5),
                arrowprops=dict(facecolor='black', width=1.5, headwidth=8), ha='center', va='center', rotation=30)
                
    # Patient -> Reflection
    ax.annotate("Response", xy=(reflection_pos[0], reflection_pos[1]-0.6), xytext=(patient_pos[0], patient_pos[1]+0.6),
                arrowprops=dict(facecolor='black', width=1.5, headwidth=8), ha='right', va='center')

    # Measurement -> Reflection
    ax.annotate("Results", xy=(reflection_pos[0]+0.5, reflection_pos[1]-0.6), xytext=(measurement_pos[0]-1.0, measurement_pos[1]+0.6),
                arrowprops=dict(facecolor='black', width=1.5, headwidth=8), ha='right', va='center')
                
    # Speaker -> End
    ax.annotate("DIAGNOSIS READY", xy=(end_pos[0]-0.9, end_pos[1]), xytext=(speaker_pos[0]+1.6, speaker_pos[1]),
                arrowprops=dict(facecolor='black', width=1.5, headwidth=8), ha='center', va='bottom')
                
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    plt.tight_layout()
    plt.savefig('Pictures/langgraph.pdf', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    plot_phase1_acc()
    plot_phase2_acc()
    plot_ds_metric()
    plot_bias_combined()
    print("Plots generated successfully in Pictures/")
