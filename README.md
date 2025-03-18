# 🤯 OverThink: Slowdown Attacks on Reasoning LLMs

## 💡 Introduction 
This is an official repository of our paper "*OverThink: Slowdown Attacks on Reasoning LLMs*". In this attack, we aim to increase the number of reasoning tokens without manipulating the generated output. 

Please follow the steps below to test our **OverThink** attack.

## 📝 Note 
* To conduct ICL-Genetic Context-Agnostic or ICL-Genetic Context-Aware attack, first complete the attack without ICL-Genetic to create the pickle files.
* We generated the pickle files in the `/pickle` folder in advance for convenience.
* Since our attack only utilizies APIs from OpenAI's o1, o1-mini and DeepSeek-R1, it does not require any CUDA environment. Feel free to run the attack in your local environment.
* You can download the FreshQA dataset from https://github.com/freshllms/freshqa

## 1. Prerequisites ✅
All experiments were done on `python==3.9.21` version. Use the following command to setup a conda environment and download required pacakages.
```
conda create -n overthink python==3.9.21 -y
conda activate overthink
pip install -r requirements.txt
```

## 2. Overthinking Attack ☠️
#### Before running the attack, make sure to complete the following steps to be able to run our attack:

### a. API keys 📍
In the `utils.py` file, either fill in the OpenAI (`openai_api_key`), official DeepSeek API (`deepseek_api_key`), or DeepSeek Firework AI API (`deepseek_firework_api_key`) depending on your preference. *(The reason why we have two different APIs for the DeepSeek-R1 model is because official DeepSeek API faced a malicious attack, which caused an error in terms of generation*. 

### b. Hyperparameters 🛠
Edit the following parameters in the `main.sh` bash file:
```
#################################
# Set model and num_samples
ATTACK="context_agnostic"      # context_agnostic, context_aware, heuristic_context_agnostic, heuristic_context_aware
MODEL="deepseek_firework"      # o1, o1-mini, o3-mini, deepseek, deepseek_firework
ATTACK_CONTEXT_MODEL="o1" # o1, o1-mini, o3-mini, deepseek, deepseek_firework
NUM_SAMPLES=5
NUM_SHOTS=None
RUN=1
REASONING_EFFORT=None
#################################
```
The "heuristic_context_agnostic" hyperparamter conducts ICL-genetic Context-Agnostic attack and "heuristic_context_aware" conducts ICL-Genetic Context-Aware attack.

### d. Run Attack 🏃‍♂️‍➡️
Run the following command the test our attack:
```
chmod +x main.sh
./main.sh
```
## To Do
Add Sudoku based attack, the current repository only contains MDP based attacks.
