#!/usr/bin/env python3.12
"""Generate two new figures from paper tables: temporal evolution + LLM behaviour heatmap."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

# Style setup
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 7,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'axes.linewidth': 0.8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

PAL = {
    'collapse': '#D64541',
    'healthy': '#3B7DD8',
    'scale': '#7F8C8D',
    'highlight': '#E8873D',
}

OUT_DIR = Path('/tmp/overleaf_nmi/images')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fig_temporal_evolution():
    """Figure 6: Temporal evolution of Moltbook topological indicators."""
    data_path = Path(__file__).resolve().parents[2] / "results" / "study1_temporal" / "temporal_metrics.json"
    with open(data_path) as f:
        data = json.load(f)

    weeks = [d['week'].replace('2026-', '') for d in data]
    gini = [d['gini'] for d in data]
    closure = [d['triadic_closure'] for d in data]
    ho_frac = [d['frac_higher_order'] for d in data]
    edge_size = [d['edge_size_mean'] for d in data]
    x = np.arange(len(weeks))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 3.8), sharex=True,
                                     gridspec_kw={'hspace': 0.12, 'height_ratios': [1, 1]})

    # Top panel: Gini + Closure
    ln1 = ax1.plot(x, gini, 'o-', color=PAL['collapse'], markersize=5, linewidth=1.5,
                   label='Degree Gini', zorder=3)
    ln2 = ax1.plot(x, closure, 's-', color=PAL['healthy'], markersize=5, linewidth=1.5,
                   label='Triadic closure', zorder=3)
    ax1.axvline(2, color='#999999', linestyle='--', linewidth=0.7, alpha=0.7)
    ax1.text(2.15, 0.82, 'Structural\ntransition', fontsize=6, color='#666666', va='top')
    ax1.set_ylabel('Coefficient')
    ax1.set_ylim(0.35, 0.88)
    ax1.legend(loc='lower right', frameon=False)
    ax1.text(-0.12, 1.08, 'a', transform=ax1.transAxes, fontsize=10, fontweight='bold')

    # Bottom panel: HO fraction + Edge size
    ln3 = ax2.plot(x, ho_frac, color=PAL['healthy'], marker='D', markersize=5, linewidth=1.5,
                   linestyle='--', label='Higher-order fraction', zorder=3)
    ax2_twin = ax2.twinx()
    ln4 = ax2_twin.plot(x, edge_size, '^-', color=PAL['scale'], markersize=5, linewidth=1.5,
                        label='Mean edge size', zorder=3)
    ax2.set_ylabel('Higher-order fraction', color=PAL['healthy'])
    ax2_twin.set_ylabel('Mean edge size', color=PAL['scale'])
    ax2.set_ylim(0.25, 0.95)
    ax2_twin.set_ylim(2.0, 6.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(weeks, rotation=0)
    ax2.set_xlabel('Week')
    ax2.axvline(2, color='#999999', linestyle='--', linewidth=0.7, alpha=0.7)

    lns = ln3 + ln4
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc='upper right', frameon=False)
    ax2.text(-0.12, 1.08, 'b', transform=ax2.transAxes, fontsize=10, fontweight='bold')

    # Despine
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    ax2_twin.spines['top'].set_visible(False)
    ax2_twin.spines['left'].set_visible(False)

    out = OUT_DIR / 'fig_temporal_evolution.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def fig_llm_behaviour_heatmap():
    """Figure 7: LLM behaviour heatmap (22 models × 4 conditions)."""

    # Data from Table 11 in paper
    round1 = {
        'GPT-5': {'vendor': 'OpenAI', 'A': 1.000, 'B': 1.000, 'C': 1.000, 'D': 0.990},
        'Claude Sonnet 4.6': {'vendor': 'Anthropic', 'A': 0.854, 'B': 0.802, 'C': 0.719, 'D': 0.615},
        'MiMo-v2-flash': {'vendor': 'Xiaomi', 'A': 0.823, 'B': 0.802, 'C': 0.781, 'D': 0.812},
        'Gemini 2.5 Pro': {'vendor': 'Google', 'A': 0.719, 'B': 0.823, 'C': 0.708, 'D': 0.646},
    }

    round2 = {
        'GPT-5.4': {'vendor': 'OpenAI', 'A': 0.875, 'B': 1.000, 'C': 0.875, 'D': 0.906},
        'Seed-OSS-36B': {'vendor': 'ByteDance', 'A': 0.938, 'B': 1.000, 'C': 0.938, 'D': 0.812},
        'MiniMax-M2.7': {'vendor': 'MiniMax', 'A': 0.906, 'B': 0.594, 'C': 0.875, 'D': 0.844},
        'Claude Opus 4.5': {'vendor': 'Anthropic', 'A': 0.719, 'B': 0.531, 'C': 0.531, 'D': 0.531},
        'Qwen3-max': {'vendor': 'Alibaba', 'A': 0.594, 'B': 0.469, 'C': 0.531, 'D': 0.688},
        'Claude Sonnet 4.5': {'vendor': 'Anthropic', 'A': 0.562, 'B': 0.469, 'C': 0.500, 'D': 0.594},
        'DeepSeek-R1': {'vendor': 'DeepSeek', 'A': 0.500, 'B': 0.531, 'C': 0.469, 'D': 0.375},
        'DeepSeek-V3.2': {'vendor': 'DeepSeek', 'A': 0.500, 'B': 0.562, 'C': 0.500, 'D': 0.438},
        'Kimi-K2': {'vendor': 'Moonshot', 'A': 0.500, 'B': 0.469, 'C': 0.531, 'D': 0.531},
        'GLM-5': {'vendor': 'Zhipu', 'A': 0.500, 'B': 0.219, 'C': 0.188, 'D': 0.000},
        'MiMo-v2.5-Pro': {'vendor': 'Xiaomi', 'A': 0.469, 'B': 0.438, 'C': 0.406, 'D': 0.375},
        'o4-mini': {'vendor': 'OpenAI', 'A': 0.375, 'B': 0.375, 'C': 0.375, 'D': 0.375},
        'GPT-5-mini': {'vendor': 'OpenAI', 'A': 0.375, 'B': 0.375, 'C': 0.375, 'D': 0.375},
    }

    # Build matrix
    all_models = list(round1.keys()) + list(round2.keys())
    all_data = {**round1, **round2}
    conditions = ['A', 'B', 'C', 'D']
    n_models = len(all_models)

    matrix = np.zeros((n_models, 4))
    vendors = []
    for i, model in enumerate(all_models):
        for j, cond in enumerate(conditions):
            matrix[i, j] = all_data[model][cond]
        vendors.append(all_data[model]['vendor'])

    # Vendor colors
    vendor_colors = {
        'OpenAI': '#10A37F',
        'Anthropic': '#D4A574',
        'Google': '#4285F4',
        'Xiaomi': '#FF6900',
        'DeepSeek': '#5B6ACD',
        'Alibaba': '#FF6A00',
        'ByteDance': '#000000',
        'Moonshot': '#6B4FBB',
        'Zhipu': '#2E86AB',
        'MiniMax': '#E94560',
    }

    # Figure
    fig, ax = plt.subplots(figsize=(5.0, 5.5))

    norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1.0)
    im = ax.imshow(matrix, cmap='RdYlBu', norm=norm, aspect='auto')

    # Cell annotations
    for i in range(n_models):
        for j in range(4):
            val = matrix[i, j]
            color = 'white' if val < 0.2 or val > 0.85 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6, color=color)

    # Separator line between Round 1 and Round 2
    n_r1 = len(round1)
    ax.axhline(n_r1 - 0.5, color='white', linewidth=2)

    # Labels
    ax.set_xticks(range(4))
    ax.set_xticklabels(['A (Pairs)', 'B (Star)', 'C (Triads)', 'D (5-Clique)'], fontsize=7)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(all_models, fontsize=6.5)

    # Vendor color indicators on left
    for i, v in enumerate(vendors):
        color = vendor_colors.get(v, '#333333')
        ax.text(-0.7, i, '|', ha='center', va='center', fontsize=12, color=color,
                fontweight='bold', transform=ax.get_yaxis_transform())

    # Mean rho bars on right
    means = matrix.mean(axis=1)
    ax_right = ax.twinx()
    ax_right.set_ylim(ax.get_ylim())
    ax_right.set_yticks(range(n_models))
    ax_right.set_yticklabels([f'{m:.2f}' for m in means], fontsize=6, color='#555555')
    ax_right.tick_params(axis='y', length=0)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.set_ylabel('Mean $\\rho$', fontsize=7, color='#555555')

    # Round labels (in data coordinates, inverted y-axis for imshow)
    ax.text(-0.8, (n_r1 - 1) / 2, 'Round 1\n(49 configs)',
            fontsize=6, color='#666666', ha='center', va='center', style='italic')
    ax.text(-0.8, n_r1 + (n_models - n_r1 - 1) / 2, 'Round 2\n(9 configs)',
            fontsize=6, color='#666666', ha='center', va='center', style='italic')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.12)
    cbar.set_label('Norm adoption rate ($\\rho$)', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax.set_title('Cross-model norm adoption under four topological conditions', fontsize=8, pad=10)

    # Despine
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    out = OUT_DIR / 'fig_llm_behaviour_heatmap.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def fig_robustness():
    """Figure: Parameter robustness (tornado sensitivity + pass rate bar)."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5),
                                     gridspec_kw={'width_ratios': [1.2, 1], 'wspace': 0.4})

    # Panel a: Tornado sensitivity
    params = [r'$\alpha$ (overlap-closure)', r'$\lambda$ (attention decay)', r'$C$ (context window)']
    spans = [0.573, 0.000, 0.000]
    colors = [PAL['highlight'] if s > 0.1 else PAL['scale'] for s in spans]

    y_pos = np.arange(len(params))
    ax1.barh(y_pos, spans, color=colors, height=0.5, edgecolor='none')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(params)
    ax1.set_xlabel('Influence span on $\\Phi$ ratio')
    ax1.set_xlim(0, 0.7)
    ax1.axvline(0.1, color='#cccccc', linestyle='--', linewidth=0.7)
    ax1.text(0.12, 2.3, 'Negligible\nthreshold', fontsize=5.5, color='#999999')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.text(-0.2, 1.12, 'a', transform=ax1.transAxes, fontsize=10, fontweight='bold')

    # Panel b: Claim pass rates
    claims = ['$\\Phi_{H}/\\Phi_{AI} > 1.5$', '$\\Phi_{H}/\\Phi_{AI} > 2.0$',
              'SP bistable,\nMoltbook not', '$\\rho^*_H > \\rho^*_{AI}$', 'All claims\nsimultaneously']
    rates = [100.0, 55.2, 31.2, 100.0, 100.0]
    colors_b = [PAL['healthy'] if r == 100 else (PAL['highlight'] if r > 50 else PAL['scale']) for r in rates]

    y_pos2 = np.arange(len(claims))
    ax2.barh(y_pos2, rates, color=colors_b, height=0.55, edgecolor='none')
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(claims, fontsize=6.5)
    ax2.set_xlabel('Pass rate (%, $N=500$ LHS samples)')
    ax2.set_xlim(0, 110)
    ax2.axvline(100, color='#cccccc', linestyle='--', linewidth=0.7)
    for i, r in enumerate(rates):
        ax2.text(r + 1.5, i, f'{r:.0f}%', va='center', fontsize=6, color='#555555')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.text(-0.25, 1.12, 'b', transform=ax2.transAxes, fontsize=10, fontweight='bold')

    out = OUT_DIR / 'fig_robustness.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == '__main__':
    print("Generating figures from table data...")
    fig_temporal_evolution()
    fig_llm_behaviour_heatmap()
    fig_robustness()
    print("Done! Output in:", OUT_DIR)
