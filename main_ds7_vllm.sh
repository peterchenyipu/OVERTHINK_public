#################################
# Set model and num_samples
ATTACK="context_agnostic" # context_agnostic, context_aware, heuristic_context_agnostic, heuristic_context_aware
MODEL="deepseek-ai_DeepSeek-R1-Distill-Qwen-7B" # o1, o1-mini, deepseek, deepseek_firework
ATTACK_CONTEXT_MODEL="deepseek-ai_DeepSeek-R1-Distill-Qwen-7B" # o1, o1-mini, deepseek, deepseek_firework
NUM_SAMPLES=3
NUM_SHOTS=None
RUN=1
REASONING_EFFORT=None
#################################

# Automatically generate output file names based on LANG and DATASET
LOG="${ATTACK}_${MODEL}_${ATTACK_CONTEXT_MODEL}_${NUM_SAMPLES}_${REASONING_EFFORT}.log"

# Run the command
python main.py \
  --attack $ATTACK \
  --model $MODEL \
  --attack_context_model $ATTACK_CONTEXT_MODEL \
  --num_samples $NUM_SAMPLES \
  --runs $RUN \
  --reasoning_effort $REASONING_EFFORT \
  | tee $LOG
