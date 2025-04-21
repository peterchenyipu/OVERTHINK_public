#!/usr/bin/env python3
import sys
import pandas as pd

def parse_attack_log(filepath):
    # --- read all lines ---
    with open(filepath, "r") as f:
        lines = [line.rstrip("\n") for line in f]

    # --- detect defense flag ---
    is_defended = any("Enabling System Prompt Defense" in line for line in lines)
    print(f"Detected {'defended' if is_defended else 'unattacked'} run")
    # --- locate metadata block via lines of asterisks ---
    star_idxs = [i for i, L in enumerate(lines) if set(L) == {"*"}]
    if len(star_idxs) >= 2:
        meta_start = star_idxs[0] + 1
        meta_end   = star_idxs[1]
    else:
        # fallback to “all leading lines with a colon”
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

    # --- build DataFrame rows with corrected run_type logic ---
    rows = []
    for idx, (in_t, out_t, r_t) in enumerate(entries):
        qid  = idx // runs_per_sample
        ridx = idx %  runs_per_sample

        if ridx == 0:
            run_type = "clean"
        else:
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

    return pd.DataFrame(rows)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_attack_log.py <input_file> [<output_csv>]")
        sys.exit(1)

    input_file  = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "parsed_output.csv"

    df = parse_attack_log(input_file)
    df.to_csv(output_file, index=False)
    print(f"✅ Parsed {len(df)} rows and saved to {output_file}")
