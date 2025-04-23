import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ——— Define your three CSV‑pairs here ———
file_pairs = [
    ('ds7.csv',  'ds7_defense.csv'),
    ('ds32.csv',  'ds32_defense.csv'),
    ('gemini.csv',  'gemini_defense.csv'),
]

# ——— Collect average %‑increases for each (model, scenario) ———
records = []
for attack_csv, defense_csv in file_pairs:
    df_a = pd.read_csv(attack_csv)
    df_d = pd.read_csv(defense_csv)

    model_name = df_a['model'].iloc[0]

    # Attacked runs
    baseline_a = (
        df_a[df_a['run_type'] == 'clean']
        .groupby('question_id')['total_output_tokens']
        .mean()
    )
    attack_mean = (
        df_a[df_a['run_type'] == 'attacked']
        .groupby('question_id')['total_output_tokens']
        .mean()
    )
    pct_attacked = (attack_mean - baseline_a) / baseline_a * 100

    # Attacked+Defense runs
    baseline_d = (
        df_d[df_d['run_type'] == 'clean']
        .groupby('question_id')['total_output_tokens']
        .mean()
    )
    def_mean = (
        df_d[df_d['run_type'] == 'attacked+defended']
        .groupby('question_id')['total_output_tokens']
        .mean()
    )
    pct_defended = (def_mean - baseline_d) / baseline_d * 100

    records.append({
        'model': model_name,
        'scenario': 'Attacked',
        'avg_percent_increase': pct_attacked.mean()
    })
    records.append({
        'model': model_name,
        'scenario': 'Attacked+Defense',
        'avg_percent_increase': pct_defended.mean()
    })

# ——— Build DataFrame ———
results_df = pd.DataFrame(records)

# ——— Remap long model names to shorter display names ———
name_map = {
    'deepseek-ai_DeepSeek-R1-Distill-Qwen-7B': 'DeepSeek-R1-7B',
    'deepseek-ai_DeepSeek-R1-Distill-Qwen-32B':   'DeepSeek-R1-32B',
    'gemini': 'Gemini-2.5-Pro',
    # add your other raw→short mappings here
}

# create a new column for plotting
results_df['model_short'] = results_df['model'].map(name_map).fillna(results_df['model'])

print(results_df)

# ——— Plot with shorter names on x‑axis ———
import matplotlib.ticker as mtick

# ——— Styling ———
sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.5})
sns.set_context("talk")  # bigger fonts

# choose a colorblind‑friendly palette for your two scenarios
palette = {
    'Attacked':         '#4C72B0',  # blue
    'Attacked+Defense':'#55A868'   # green
}

# ——— Plot ———
plt.figure(figsize=(10, 6))
ax = sns.barplot(
    data=results_df,
    x='model_short',
    y='avg_percent_increase',
    hue='scenario',
    palette=palette
)

# remove top & right spines
sns.despine(ax=ax, top=True, right=True)

# format y‑axis as percentages
ax.yaxis.set_major_formatter(mtick.PercentFormatter())

# annotate each bar with its height
for container in ax.containers:        # one container per hue level
    ax.bar_label(container,            # matplotlib helper
                 fmt='%.0f%%',
                 padding=3,
                 fontsize=12)

# axis labels & title
ax.set_xlabel("Model", fontsize=16)
ax.set_ylabel("Average % Increase", fontsize=16)
ax.set_title("Token Increase by Model: Attack vs. Attack+Defense", fontsize=18, pad=15)

# legend formatting
leg = ax.legend(title="Scenario", frameon=True, fontsize=12, title_fontsize=14)
leg._legend_box.align = "left"

# save the plot
plt.savefig("token_increase_plot.png", dpi=300, bbox_inches='tight')
# save to pdf
plt.savefig("token_increase_plot.pdf", dpi=300, bbox_inches='tight')


plt.tight_layout()
plt.show()

