import pandas as pd
import matplotlib.pyplot as plt

# Load data
df_attack = pd.read_csv('ds7.csv')
df_def = pd.read_csv('ds7_defense.csv')

# Baseline and mean total tokens for attacked runs
baseline_attack = df_attack[df_attack['run_type'] == 'clean'].set_index('question_id')['total_output_tokens']
attack_mean = df_attack[df_attack['run_type'] == 'attacked'].groupby('question_id')['total_output_tokens'].mean()
percent_attacked = ((attack_mean - baseline_attack) / baseline_attack) * 100

# Baseline and mean total tokens for defended runs
baseline_def = df_def[df_def['run_type'] == 'clean'].set_index('question_id')['total_output_tokens']
def_mean = df_def[df_def['run_type'] == 'attacked+defended'].groupby('question_id')['total_output_tokens'].mean()
percent_defended = ((def_mean - baseline_def) / baseline_def) * 100

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
