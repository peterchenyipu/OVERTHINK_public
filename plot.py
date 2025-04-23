import pandas as pd
import matplotlib.pyplot as plt

# Load data
df_attack = pd.read_csv('ds7.csv')
df_def = pd.read_csv('ds7_defense.csv')

# ——— Attacked runs ———

# 1) baseline = mean of all clean runs per question
baseline_attack = (
    df_attack[df_attack['run_type'] == 'clean']
    .groupby('question_id')['total_output_tokens']
    .mean()
)

# 2) attacked = mean of all attacked runs per question
attack_mean = (
    df_attack[df_attack['run_type'] == 'attacked']
    .groupby('question_id')['total_output_tokens']
    .mean()
)

# 3) percent increase
percent_attacked = (attack_mean - baseline_attack) / baseline_attack * 100



# ——— Defended runs ———

# 1) baseline = mean of all clean runs per question
baseline_def = (
    df_def[df_def['run_type'] == 'clean']
    .groupby('question_id')['total_output_tokens']
    .mean()
)

# 2) defended = mean of all attacked+defended runs per question
def_mean = (
    df_def[df_def['run_type'] == 'attacked+defended']
    .groupby('question_id')['total_output_tokens']
    .mean()
)

# 3) percent increase
percent_defended = (def_mean - baseline_def) / baseline_def * 100


# Combine results into a DataFrame
results = pd.DataFrame({
    'Attacked (%)': percent_attacked,
    'Attacked+Defense (%)': percent_defended
})

# Display per-question percentages and compute averages
print("Percentage increase per question:")
print(results)
avg_attack = percent_attacked.mean()
avg_def = percent_defended.mean()
print(f"\nAverage percentage increase for Attacked: {avg_attack:.2f}%")
print(f"Average percentage increase for Attacked+Defense: {avg_def:.2f}%")

# Plot bar chart of the average increases
fig, ax = plt.subplots()
ax.bar(['Attacked', 'Attacked+Defense'], [avg_attack, avg_def])
ax.set_ylabel('Average Percent Increase (%)')
ax.set_title('Average Token Increase: Attack vs. Attack+Defense')
plt.tight_layout()
plt.show()
