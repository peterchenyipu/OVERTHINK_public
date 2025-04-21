#!/usr/bin/env python3
import sys
import pandas as pd
import matplotlib.pyplot as plt

def parse_attack_log(filepath):
    """Parses an attack log file into a pandas DataFrame."""
    # --- read all lines ---
    with open(filepath, "r") as f:
        lines = [line.rstrip("\n") for line in f]

    # --- detect defense flag ---
    is_defended = any("Enabling System Prompt Defense" in line for line in lines)
    run_status = 'defended' if is_defended else 'unattacked'
    print(f"Detected {run_status} run in {filepath}")

    # --- locate metadata block via lines of asterisks ---
    star_idxs = [i for i, L in enumerate(lines) if set(L) == {"*"}]
    if len(star_idxs) >= 2:
        meta_start = star_idxs[0] + 1
        meta_end   = star_idxs[1]
    else:
        # fallback to "all leading lines with a colon"
        meta_start = 0
        meta_end = 0
        for i, L in enumerate(lines):
            if ":" not in L:
                break
            meta_end = i + 1

    # --- parse metadata key:value ---
    metadata = {}
    for L in lines[meta_start:meta_end]:
        if ":" in L:
            k, v = L.split(":", 1)
            metadata[k.strip()] = v.strip()

    attack_type    = metadata.get("attack_type", "unknown")
    model          = metadata.get("model",       "unknown")
    num_samples    = int(metadata.get("num_samples", 1))
    # assume metadata["runs"] == runs_per_sample, else compute:
    runs_per_sample = None

    # --- parse triples of token counts ---
    entries = []
    i = meta_end + 1
    while i + 2 < len(lines):
        l1, l2, l3 = lines[i:i+3]
        if l1.startswith("input_tokens") and l2.startswith("output_tokens") and l3.startswith("reasoning_tokens"):
            in_t = int(l1.split(":",1)[1])
            out_t = int(l2.split(":",1)[1])
            r_t = int(l3.split(":",1)[1])
            entries.append((in_t, out_t, r_t))
            i += 3
        else:
            i += 1

    # --- if runs_per_sample not in metadata, derive it ---
    if runs_per_sample is None:
        total = len(entries)
        runs_per_sample = total // num_samples if num_samples else total
        if runs_per_sample == 0:
             print(f"Warning: Could not determine runs_per_sample for {filepath}. Assuming 1.", file=sys.stderr)
             runs_per_sample = 1 # Avoid division by zero

    # --- build DataFrame rows with corrected run_type logic ---
    rows = []
    for idx, (in_t, out_t, r_t) in enumerate(entries):
        qid  = idx // runs_per_sample
        ridx = idx %  runs_per_sample

        if ridx == 0:
            run_type = "clean"
        else:
            # Use the previously detected defense status
            run_type = "attacked+defended" if is_defended else "attacked"

        rows.append({
            "question_id":         qid,
            "run_type":            run_type,
            "attack_type":         attack_type,
            "model":               model,
            "input_tokens":        in_t,
            "output_tokens":       out_t,
            "reasoning_tokens":    r_t,
            "total_output_tokens": out_t + r_t
        })

    print(f"Parsed {len(rows)} rows from {filepath}")
    return pd.DataFrame(rows)

def plot_comparison(df_attack, df_def, output_filename="comparison_plot.png"):
    """Calculates and plots the token increase comparison."""
    # Baseline and mean total tokens for attacked runs
    baseline_attack = df_attack[df_attack['run_type'] == 'clean'].set_index('question_id')['total_output_tokens']
    attack_mean = df_attack[df_attack['run_type'] == 'attacked'].groupby('question_id')['total_output_tokens'].mean()
    percent_attacked = ((attack_mean - baseline_attack) / baseline_attack) * 100

    # Baseline and mean total tokens for defended runs
    baseline_def = df_def[df_def['run_type'] == 'clean'].set_index('question_id')['total_output_tokens']
    def_mean = df_def[df_def['run_type'] == 'attacked+defended'].groupby('question_id')['total_output_tokens'].mean()
    percent_defended = ((def_mean - baseline_def) / baseline_def) * 100

    # Combine results into a DataFrame
    # Use outer join to handle cases where question_ids might not perfectly overlap
    results = pd.DataFrame({
        'Attacked (%)': percent_attacked,
        'Attacked+Defense (%)': percent_defended
    }).fillna(0) # Fill NaNs that might arise from joins or divisions by zero

    # Display per-question percentages and compute averages
    print("\nPercentage increase per question:")
    print(results)
    avg_attack = results['Attacked (%)'].mean()
    avg_def = results['Attacked+Defense (%)'].mean()
    print(f"\nAverage percentage increase for Attacked: {avg_attack:.2f}%")
    print(f"Average percentage increase for Attacked+Defense: {avg_def:.2f}%")

    # Plot bar chart of the average increases
    fig, ax = plt.subplots()
    ax.bar(['Attacked', 'Attacked+Defense'], [avg_attack, avg_def], color=['#ff7f0e', '#1f77b4']) # Use distinct colors
    ax.set_ylabel('Average Percent Increase in Total Output Tokens (%)')
    ax.set_title('Average Token Increase: Attack vs. Attack+Defense')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"\n📊 Plot saved to {output_filename}")


if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python parse_and_plot.py <attack_log_file> <defense_log_file> [<output_plot_filename>]")
        sys.exit(1)

    attack_log_file = sys.argv[1]
    defense_log_file = sys.argv[2]
    output_plot_file = sys.argv[3] if len(sys.argv) == 4 else "comparison_plot.png"

    try:
        df_attack = parse_attack_log(attack_log_file)
        df_defense = parse_attack_log(defense_log_file)

        if df_attack.empty or df_defense.empty:
             print("Error: One or both parsed dataframes are empty. Cannot proceed.", file=sys.stderr)
             sys.exit(1)

        # Basic validation: Check if 'clean' runs exist in both dataframes
        if 'clean' not in df_attack['run_type'].unique() or 'clean' not in df_defense['run_type'].unique():
            print("Warning: Missing 'clean' runs in one or both logs. Percentage calculation might be inaccurate.", file=sys.stderr)

        plot_comparison(df_attack, df_defense, output_plot_file)
        print("\n✅ Analysis and plotting complete.")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1) 