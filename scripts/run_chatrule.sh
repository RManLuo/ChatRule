DATA="family" # umls wn-18rr yago
SUMMARY_MODEL="none"
RANK_MODE="all"
FEW_SHOT=50
REPEAT=15
MEM="50G"
# PREFIX=""


MODEL_NAME="gpt-3.5-turbo"
N_PROCESS=10


# MODEL_NAME="gpt-4"
# N_PROCESS=10

# MODEL_NAME="Qwen-7B-Chat"
# MODEL_PATH="Qwen/Qwen-7B-Chat"
# N_PROCESS=1
# QUANT=none

# MODEL_NAME="Qwen-14B-Chat"
# MODEL_PATH="Qwen/Qwen-14B-Chat"
# N_PROCESS=1
# QUANT=none

# MODEL_NAME="Mistral-7B-Instruct-v0.1"
# MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.1"
# N_PROCESS=1
# QUANT=none

# MODEL_NAME="llama2-7B-chat-hf"
# MODEL_PATH="meta-llama/Llama-2-7b-chat-hf"
# N_PROCESS=1
# QUANT=none

# MODEL_NAME="llama2-13B-chat-hf"
# MODEL_PATH="meta-llama/Llama-2-13b-chat-hf"
# N_PROCESS=1
# QUANT=none

# MODEL_NAME="vicuna-33b-v1.3"
# MODEL_PATH="lmsys/vicuna-33b-v1.3"
# N_PROCESS=1
# QUANT=none

# MODEL_NAME="llama2-70B-chat-hf"
# MODEL_PATH="meta-llama/Llama-2-70b-chat-hf"
# N_PROCESS=1
# QUANT=none



MODEL_PREFIX=${MODEL_NAME}-top-0-f-${FEW_SHOT}-l-${REPEAT}
if [[ $MODEL_NAME == *"gpt"* ]]; then
    python chat_rule_generator.py --dataset $DATA --model_name $MODEL_NAME -f $FEW_SHOT -l $REPEAT -n $N_PROCESS && \ 
    python clean_rule.py --dataset $DATA --model $SUMMARY_MODEL -p $MODEL_PREFIX && \
    python rank_rule.py --dataset $DATA -p clean_rules/${DATA}/${MODEL_PREFIX}/${SUMMARY_MODEL} --eval_mode ${RANK_MODE} && \
    python kg_completion.py --dataset $DATA -p ranked_rules/${DATA}/${MODEL_PREFIX}/${SUMMARY_MODEL}/${RANK_MODE}
else
    python chat_rule_generator.py --dataset $DATA --model_name $MODEL_NAME --model_path ${MODEL_PATH} -f $FEW_SHOT -l $REPEAT -n $N_PROCESS --quant $QUANT && \ 
    python clean_rule.py --dataset $DATA --model $SUMMARY_MODEL -p $MODEL_PREFIX && \
    python rank_rule.py --dataset $DATA -p clean_rules/${DATA}/${MODEL_PREFIX}/${SUMMARY_MODEL} --eval_mode ${RANK_MODE} && \
    python kg_completion.py --dataset $DATA -p ranked_rules/${DATA}/${MODEL_PREFIX}/${SUMMARY_MODEL}/${RANK_MODE}
fi


