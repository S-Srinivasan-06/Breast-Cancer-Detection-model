import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

# ─── Shared style ──────────────────────────────────────────────────────────
BG   = "#fdf8f2"
GRID = "#e8e0d6"
BEN  = "#6dbf88"
MAL  = "#c0624a"
BEN_A = "#6dbf8888"
MAL_A = "#c0624a88"
TITLE_COLOR = "#8b2a2a"
FONT = "DejaVu Sans"

def apply_style(fig, ax):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#c8beb4')
    ax.tick_params(colors='#555555')
    ax.xaxis.label.set_color('#444444')
    ax.yaxis.label.set_color('#444444')
    ax.title.set_color(TITLE_COLOR)
    ax.title.set_fontweight('bold')

df = pd.read_csv("data.csv")
df = df.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')
benign    = df[df['diagnosis'] == 'B']
malignant = df[df['diagnosis'] == 'M']

# ─── Chart 1: Smoothness distribution ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(9.5, 5))
bins = np.linspace(df['smoothness_mean'].min(), df['smoothness_mean'].max(), 30)
ax.hist(benign['smoothness_mean'],    bins=bins, alpha=0.65, color=BEN, label='Benign (Smoothness)',    zorder=3)
ax.hist(malignant['smoothness_mean'], bins=bins, alpha=0.65, color=MAL, label='Malignant (Smoothness)', zorder=3)
ax.set_xlabel('Smoothness Mean (Surface Uniformity)')
ax.set_ylabel('Cell Sample Count')
ax.set_title('Nuclear Smoothness Distribution (Surface Uniformity)')
ax.legend(framealpha=0.9, facecolor=BG)
apply_style(fig, ax)
fig.tight_layout()
fig.savefig('charts/smoothness_group.png', dpi=130, facecolor=BG)
plt.close()
print("Saved smoothness_group.png")

# ─── Chart 2: Symmetry distribution ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(9.5, 5))
bins = np.linspace(df['symmetry_mean'].min(), df['symmetry_mean'].max(), 28)
ax.hist(benign['symmetry_mean'],    bins=bins, alpha=0.65, color=BEN, label='Benign (Symmetry)',    zorder=3)
ax.hist(malignant['symmetry_mean'], bins=bins, alpha=0.65, color=MAL, label='Malignant (Symmetry)', zorder=3)
ax.set_xlabel('Symmetry Mean')
ax.set_ylabel('Cell Sample Count')
ax.set_title('Nuclear Symmetry Distribution')
ax.legend(framealpha=0.9, facecolor=BG)
apply_style(fig, ax)
fig.tight_layout()
fig.savefig('charts/symmetry_group.png', dpi=130, facecolor=BG)
plt.close()
print("Saved symmetry_group.png")

# ─── Chart 3: Compactness distribution ────────────────────────────────────
fig, ax = plt.subplots(figsize=(9.5, 5))
bins = np.linspace(df['compactness_mean'].min(), df['compactness_mean'].max(), 30)
ax.hist(benign['compactness_mean'],    bins=bins, alpha=0.65, color=BEN, label='Benign (Compactness)',    zorder=3)
ax.hist(malignant['compactness_mean'], bins=bins, alpha=0.65, color=MAL, label='Malignant (Compactness)', zorder=3)
ax.set_xlabel('Compactness Mean (Perimeter²/Area − 1)')
ax.set_ylabel('Cell Sample Count')
ax.set_title('Nuclear Compactness Distribution')
ax.legend(framealpha=0.9, facecolor=BG)
apply_style(fig, ax)
fig.tight_layout()
fig.savefig('charts/compactness_group.png', dpi=130, facecolor=BG)
plt.close()
print("Saved compactness_group.png")

# ─── Chart 4: Box plot — top 6 features by class ──────────────────────────
features = ['radius_mean','texture_mean','perimeter_mean',
            'area_mean','concavity_mean','concave points_mean']
labels   = ['Radius','Texture','Perimeter','Area','Concavity','Concave Pts']

# normalise each to 0-1 for comparison
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_norm = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)
df_norm['diagnosis'] = df['diagnosis'].values
b_data = [df_norm[df_norm['diagnosis']=='B'][f].values for f in features]
m_data = [df_norm[df_norm['diagnosis']=='M'][f].values for f in features]

fig, ax = plt.subplots(figsize=(10, 5.5))
positions_b = np.arange(len(features)) * 2.4
positions_m = positions_b + 0.9

bp_b = ax.boxplot(b_data, positions=positions_b, widths=0.7,
                  patch_artist=True, notch=False,
                  boxprops=dict(facecolor=BEN+'99', color=BEN),
                  medianprops=dict(color='#2a6e45', linewidth=2),
                  whiskerprops=dict(color=BEN), capprops=dict(color=BEN),
                  flierprops=dict(marker='o', color=BEN, alpha=0.3, markersize=3))
bp_m = ax.boxplot(m_data, positions=positions_m, widths=0.7,
                  patch_artist=True, notch=False,
                  boxprops=dict(facecolor=MAL+'99', color=MAL),
                  medianprops=dict(color='#7a2018', linewidth=2),
                  whiskerprops=dict(color=MAL), capprops=dict(color=MAL),
                  flierprops=dict(marker='o', color=MAL, alpha=0.3, markersize=3))

ax.set_xticks(positions_b + 0.45)
ax.set_xticklabels(labels)
ax.set_ylabel('Normalised Value (0–1)')
ax.set_title('Feature Distribution by Class (Normalised)')
ax.legend(handles=[mpatches.Patch(color=BEN, label='Benign'),
                   mpatches.Patch(color=MAL, label='Malignant')],
          facecolor=BG, framealpha=0.9)
apply_style(fig, ax)
fig.tight_layout()
fig.savefig('charts/boxplot_comparison.png', dpi=130, facecolor=BG)
plt.close()
print("Saved boxplot_comparison.png")

print("All new charts generated successfully.")
