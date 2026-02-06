#---------------------------------------
# Date          : 29 Dec 25
#Author         : Elton Tay, Chatgpt as part of AI driven solution
#Dependencies   : llama_cpp
#Purpose        : Loading Llama Model
#Output         : -
#----------------------------------------
from llama_cpp import Llama

#Path to model
model_path ="/Users/eltontay/Documents/Github/capstone.nosync/Capstone_1_Healthcare_Rag/llm/meta-llama-3.1-8b-instruct-q4_k_m.gguf"

#Initialize the model
llm = Llama(model_path=model_path, n_ctx = 2048, n_threads =4)

'''
n_ctx - sets the context window = input + output combined
    tokens(prompts + context + rules + question + separators + history) + tokens (generated answer) <= n_ctx
n_threads - controls CPU threads
'''

#Test prompt
prompt = "What are the symptons of spesis of children?"

#Generate output
response = llm(prompt=prompt, max_tokens = 200) # llama_cpp inference engine syntax in Python
print(response['choices'][0]['text'])

'''
output (response) is a dictionary, not just a string. It contains multiple fields,
eg. choices, usage etc
[0] -> take the first generated choice
['text'] -> extract the actual text string from that choice

the choices list in the output is part of the LLM API design. A prompt can produce multiple candidate completions.
max_tokens = output only
'''