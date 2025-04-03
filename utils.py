from openai import OpenAI
import tiktoken
import requests
import json
from bs4 import BeautifulSoup
import re

openai_api_key = ""
deepseek_api_key = ""
deepseek_firework_api_key = ""

openai_client = OpenAI(api_key=openai_api_key)
deepseek_client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

def extract_description(text, tag):
    # Use a regular expression to find the content between <DESCRIPTION></DESCRIPTION>
    search_string = f'<{tag}>(.*?)</{tag}>'
    #print(search_string)
    match = re.search(search_string, text,re.DOTALL)
    if match:
        return match.group(1)  # Return the content inside the tags
    else:
        return None  # If no match is found, return None

def fetch_wikipedia_article(url):
    # Send a GET request to the Wikipedia page
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors

    # Parse the page content with BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")

    # Extract the title of the article
    title = soup.find("h1", {"id": "firstHeading"}).text

    # Extract the main content of the article
    content_div = soup.find("div", {"id": "mw-content-text"})
    paragraphs = content_div.find_all("p")

    # Combine all paragraph texts into a single string
    article_text = "\n".join([para.text for para in paragraphs if para.text.strip()])

    return title, article_text
    
# Function to create prompts
def create_prompts(question, context, target_context_templates):
    """
    Generate prompts for the original context and multiple target contexts.
    """
    prompts = {
        "original_prompt": f'You are an helpful agent who will answer the following user question "{question}" Use the following retrieved context: "{context}".'
    }
    # print(prompts)
    for i, target_template in enumerate(target_context_templates):
        target_context = target_template + context
        prompts[f"attack_prompt_{i+1}"] = f'You are an helpful agent who will answer the following user question"{question}". Use the following retrieved context: "{target_context}"'
    return prompts

# Function to create prompts
def create_context(question, context, target_context_templates):
    """
    Generate prompts for the original context and multiple target contexts.
    """
    prompts = {
        "original_prompt": f'You are an helpful agent who will answer the following user question "{question}" Use the following retrieved context: "{context}".'
    }
    # print(prompts)
    for i, target_template in enumerate(target_context_templates):
        target_context = target_template + context
        prompts[f"attack_context_{i+1}"] = target_context
    return prompts

def create_context_weaving(question, contex_index, context, problem_templates):
    """
    Generate prompts for the original context and multiple target contexts.
    """
    prompts = {
        "original_prompt": f'You are an helpful agent who will answer the following user question "{question}" Use the following retrieved context: "{context}".'
    }
    with open("weaving_template_freshQA.json", "r") as f:
        weaving_templates_freshQA= json.load(f)
    print(prompts)
    for i, target_template in enumerate(problem_templates):
        weaving_template = weaving_templates_freshQA["sample_" + str(contex_index + 1)]['template'].replace("<MDP>", problem_templates[i])
        if weaving_templates_freshQA["sample_" + str(contex_index + 1)]['context_position'] == 1:
            weaving_context = weaving_template + context
        else:
            weaving_context = context + weaving_template
        prompts[f"attack_context_{i+1}"] = weaving_context
    return prompts

# Function to create prompts
def create_prompts_weaving(question, contex_index, context, problem_templates):
    """
    Generate prompts for the original context and multiple target contexts.
    """
    prompts = {
        "original_prompt": f'You are an helpful agent who will answer the following user question "{question}" Use the following retrieved context: "{context}".'
    }
    with open("weaving_template_freshQA.json", "r") as f:
        weaving_templates_freshQA = json.load(f)
    # print(prompts)
    for i, target_template in enumerate(problem_templates):
        weaving_template = weaving_templates_freshQA["sample_" + str(contex_index + 1)]['template'].replace("<MDP>", problem_templates[i])
        if weaving_templates_freshQA["sample_" + str(contex_index + 1)]['context_position'] == 1:
            target_context = weaving_template + context
        else:
            target_context = context + weaving_template
        prompts[f"attack_prompt_{i+1}"] = f'You are an helpful agent who will answer the following user question"{question}". Use the following retrieved context: "{target_context}"'
    return prompts

def count_tokens(text, model="gpt-3.5-turbo"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print(f"Warning: model {model} not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    
    token_count = len(encoding.encode(text))
    return token_count


def run_command(prompt, model, reasoning_effort='low'):
    # print('prompt: ', prompt)
    messages=[{"role": "user", "content": prompt}]

    if model == "deepseek_firework":
        # Fireworks DeepSeek-R1 generation
        payload = {
                "model": "accounts/fireworks/models/deepseek-r1",
                "max_tokens": 1000000000,
                "top_p": 1,
                "top_k": 35,
                "presence_penalty": 0,
                "frequency_penalty": 0,
                "temperature": 0.6,
                "messages": messages
        }

        headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": deepseek_firework_api_key
        }

        url = "https://api.fireworks.ai/inference/v1/chat/completions"
        response = requests.request("POST", url, headers=headers, data=json.dumps(payload))

        # Reasoning content
        response_dict = json.loads(response.text)
        # print(response_dict)
        content = response_dict["choices"][0]['message']['content']
        completion_tokens = response_dict['usage']['completion_tokens']

        # print(response_dict)
        # print("*"*100)
        # print(content)
        text = (content.split('<think>')[1]).split('</think>')[1]
        reasoning_content = (content.split('<think>')[1]).split('</think>')[0]
    
        # Count tokens
        input_tokens = count_tokens(prompt)
        reasoning_tokens = count_tokens(reasoning_content)
        output_tokens = count_tokens(text)

        print("input_tokens: ", input_tokens)
        print("output_tokens: ", output_tokens)
        print("reasoning_tokens: ", reasoning_tokens)
        print("*"*100)

        return {'text': text, 
                'input tokens': input_tokens,
                'output tokens': output_tokens,
                'reasoning tokens':reasoning_tokens, 
                "entire respose":response.text}

    elif model == 'deepseek':
        print(f'prompt: {prompt}', )
        client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

        # Round 1
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages,
            max_tokens=8192
        )

        text = response.choices[0].message.content
        reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens - reasoning_tokens

        print("input_tokens: ", input_tokens)
        print("output_tokens: ", output_tokens)
        print("reasoning_tokens: ", reasoning_tokens)
        
        return {'text': text, 
                'input tokens': input_tokens,
                'output tokens': output_tokens,
                'reasoning tokens':reasoning_tokens, 
                "entire respose":response}

    else:
        print(f"reasoning effort: {reasoning_effort}")
        response = openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    reasoning_effort=reasoning_effort
                    )
        
        text = response.choices[0].message.content
        reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens - reasoning_tokens

        print("input_tokens: ", input_tokens)
        print("output_tokens: ", output_tokens)
        print("reasoning_tokens: ", reasoning_tokens)

        return {'text': text, 
                'input tokens': input_tokens,
                'output tokens': output_tokens,
                'reasoning tokens':reasoning_tokens, 
                "entire respose":response}


    
