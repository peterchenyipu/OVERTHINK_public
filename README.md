# OVERTHINK_public

### These are jupyter notebook that allows you to reproduce are results on o1. For every test, add your openAI key in the notebook

1. context-agnostic-o1.ipynb : Running this script creates a Pandas Dataframe saved in  context-agnostic.pkl. Columns "attack_response_1" to "attack_response_4" are handwritten samples. Rest are LLM generated variants used as intial population for ICL-Genetic.

2. context-agnostic-icl-o1.ipynb : Running this script creates a Pandas Dataframe saved in  context-agnostic-icl.pkl, saving the best performing templates and its responses.


