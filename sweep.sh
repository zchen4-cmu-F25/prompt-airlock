#!/bin/bash

PYTHON=${PYTHON:-python}

# types=('RandomSwapPerturbation' 'RandomInsertPerturbation' 'RandomPatchPerturbation')
# pcts=(5 10 15 20)
# num_copies=(2 4 6 8 10)
# trials=(1 2 3 4)
# detectors=('binary' 'three_class')
types=('RandomSwapPerturbation')
pcts=(5 10 15 20 25 30)
num_copies=(10)
trials=(1)
detectors=('binary' 'three_class')
target_model=qwen2.5-3b
results_root=./results

for trial in "${trials[@]}"; do
    for detector in "${detectors[@]}"; do
        for type in "${types[@]}"; do
            for pct in "${pcts[@]}"; do
                for n in "${num_copies[@]}"; do

                    dir=$results_root/$target_model/trial-$trial/det-$detector/n-$n-type-$type-pct-$pct
                    echo $dir

                    $PYTHON main.py \
                        --results_dir $dir \
                        --target_model $target_model \
                        --attack GCG \
                        --attack_logfile data/GCG/vicuna_behaviors.json \
                        --smoothllm_pert_type $type \
                        --smoothllm_pert_pct $pct \
                        --smoothllm_num_copies $n \
                        --smoothllm_detector $detector \
                        --trial $trial

                done
            done
        done
    done
done

echo ""
echo "========================================"
echo "  All runs complete. Collecting results  "
echo "========================================"
echo ""

$PYTHON - "$results_root" "$target_model" <<'PYEOF'
import sys, glob, os, pandas as pd

results_root = sys.argv[1]
target_model = sys.argv[2]

search_dir = os.path.join(results_root, target_model)
files = sorted(glob.glob(f'{search_dir}/**/summary.pd', recursive=True))
if not files:
    print("No summary.pd files found.")
    sys.exit(0)

df = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)

cols = [
    'Perturbation percentage', 'Perturbation type',
    'Number of smoothing copies', 'Detector', 'Trial index',
    'Refusal %', 'Confusion %', 'Jailbreak %',
]
cols = [c for c in cols if c in df.columns]
sort_keys = [c for c in ['Detector', 'Perturbation percentage'] if c in df.columns]
if sort_keys:
    df = df.sort_values(sort_keys)

print("=" * 80)
print("  RESULTS SUMMARY")
print("=" * 80)
print(df[cols].to_string(index=False))
print()

if 'Detector' in df.columns and df['Detector'].nunique() == 2:
    pivot = df.pivot_table(
        index=['Perturbation type', 'Perturbation percentage'],
        columns='Detector',
        values='Jailbreak %',
        aggfunc='mean'
    )
    if 'binary' in pivot.columns and 'three_class' in pivot.columns:
        pivot['Δ Jailbreak%'] = pivot['binary'] - pivot['three_class']
        print("=" * 80)
        print("  BINARY vs THREE_CLASS COMPARISON (Jailbreak %)")
        print("=" * 80)
        print(pivot.to_string())
        print()

csv_path = os.path.join(search_dir, 'all_results.csv')
df.to_csv(csv_path, index=False)
print(f"Full results saved to: {csv_path}")
PYEOF
