from openai import OpenAI
import openai
import time
import tiktoken
import requests
import json
from bs4 import BeautifulSoup
import re
from dotenv import load_dotenv
import os
from google import genai
from google.genai import types
from google.genai.errors import ClientError
load_dotenv()
        
openai_api_key = ""
deepseek_api_key = ""
deepseek_firework_api_key = ""
gemini_api_key = os.getenv('GEMINI_TOKEN')

openai_client = OpenAI(api_key=openai_api_key)
deepseek_client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
gemini_client = OpenAI(api_key=gemini_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")


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
def create_prompts(question, context, target_context_templates, duplicates=0):
    """
    Generate prompts for the original context and multiple target contexts.
    """
    prompts = {
        "original_prompt": f'You are an helpful agent who will answer the following user question "{question}" Use the following retrieved context: "{context}".'
    }
    # repeat the original prompt
    for i in range(duplicates):
        prompts[f"original_prompt_{i}"] = prompts["original_prompt"]
    
    
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

def count_tokens_gemini(text):
    client = genai.Client(api_key=gemini_api_key)
    count = client.models.count_tokens(
        model="gemini-2.5-pro-exp-03-25",
        contents=text,
    ).total_tokens
    return count
    
def run_command(prompt, model, reasoning_effort='low', system_prompt=None):
    # print('prompt: ', prompt)
    if system_prompt is not None:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    else:
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
    elif model.startswith('deepseek-ai'): # vllm serving
        # print(f'prompt: {prompt}', )
        # print('Begin Asking')
        model = model.replace('_', '/')
        client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            # max_tokens=8192
        )
        # print("Got Response")

        text = response.choices[0].message.content
        reasoning_content = response.choices[0].message.reasoning_content
        
        total_output_tokens_by_usage = response.usage.completion_tokens

        
        input_tokens = response.usage.prompt_tokens
        output_tokens = count_tokens(text)
        reasoning_tokens = response.usage.completion_tokens - output_tokens

        print("input_tokens: ", input_tokens)
        print("output_tokens: ", output_tokens)
        print("reasoning_tokens: ", reasoning_tokens)
        
        return {'text': text, 
                'input tokens': input_tokens,
                'output tokens': output_tokens,
                'reasoning tokens':reasoning_tokens, 
                "entire respose":response}
    
    elif model.startswith('deepseek-r1'):
        print(f'prompt: {prompt}')
        # exit()
        client = OpenAI(api_key='ollama', base_url="http://localhost:11434/v1")

        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )

        text = response.choices[0].message.content

        try:
            # Attempt to extract reasoning and output content
            reasoning_content = text.split('<think>')[1].split('</think>')[0].strip()
            output_content = text.split('<think>')[1].split('</think>')[1].strip()

            input_tokens = count_tokens(prompt)
            reasoning_tokens = count_tokens(reasoning_content)
            output_tokens = count_tokens(output_content)

            print("input_tokens: ", input_tokens)
            print("output_tokens: ", output_tokens)
            print("reasoning_tokens: ", reasoning_tokens)

            return {
                'text': text,
                'input tokens': input_tokens,
                'output tokens': output_tokens,
                'reasoning tokens': reasoning_tokens,
                'entire response': response
            }

        except Exception as e:
            print(f"[ERROR] Failed to extract <think> tags: {e}")

            # Log failed prompt + response to file for later inspection
            log_file = 'failed_prompts.txt'
            with open(log_file, "a") as f:
                f.write("=== FAILED PROMPT ===\n")
                f.write(f"Prompt:\n{prompt}\n")
                f.write(f"Response:\n{text}\n")
                f.write(f"Error: {str(e)}\n\n")

            return {
                'text': text,
                'input tokens': count_tokens(prompt),
                'output tokens': None,
                'reasoning tokens': None,
                'entire response': response,
                'error': str(e)
            }
    
    elif model == 'gemini':
        # print(messages[0])
        # exit()
        wait_time = 60
        while True:
            try:

                client = genai.Client(api_key=gemini_api_key)
                
                model = "gemini-2.5-pro-exp-03-25"
                contents = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=prompt),
                        ],
                    ),
                ]

                if system_prompt is not None:
                    system_instruction = [
                            types.Part.from_text(text=system_prompt),
                        ]
                else:
                    system_instruction = None

                generate_content_config = types.GenerateContentConfig(
                    response_mime_type="text/plain",
                    system_instruction=system_instruction
                )

                response = client.models.generate_content(
                    model="gemini-2.5-pro-exp-03-25",  # or gemini-2.0-flash-thinking-exp
                    contents=contents,
                    config=generate_content_config,
                )

                text = response.text
                total_output_tokens = response.usage_metadata.candidates_token_count
                input_tokens = response.usage_metadata.prompt_token_count
                reasoning_tokens = response.usage_metadata.thoughts_token_count
                output_tokens = total_output_tokens - reasoning_tokens


                # response = gemini_client.chat.completions.create(
                #     model="models/gemini-2.5-pro-exp-03-25",
                #     n=1,
                #     messages=messages
                # )
                # text = response.choices[0].message.content
                # total_output_tokens = response.usage.completion_tokens
                # input_tokens = response.usage.prompt_tokens
                # output_tokens = count_tokens_gemini(text)
                # reasoning_tokens = total_output_tokens - output_tokens

                print("input_tokens: ", input_tokens)
                print("output_tokens: ", output_tokens)
                print("reasoning_tokens: ", reasoning_tokens)

                return {
                    'text': text,
                    'input tokens': input_tokens,
                    'output tokens': output_tokens,
                    'reasoning tokens': reasoning_tokens,
                    "entire respose": response
                }

            except ClientError as e:
                print(f"[ClientError] Retrying in {wait_time} seconds: {e}")
                time.sleep(wait_time)
        # response = gemini_client.chat.completions.create(
        #     model="models/gemini-2.5-pro-exp-03-25",
        #     n=1,
        #     messages=messages
        # )
        # text = response.choices[0].message.content
        # total_output_tokens = response.usage.completion_tokens
        # input_tokens = response.usage.prompt_tokens
        # output_tokens = count_tokens_gemini(text)
        # reasoning_tokens = total_output_tokens - output_tokens
        
        # print("input_tokens: ", input_tokens)
        # print("output_tokens: ", output_tokens)
        # print("reasoning_tokens: ", reasoning_tokens)

        # return {'text': text, 
        #         'input tokens': input_tokens,
        #         'output tokens': output_tokens,
        #         'reasoning tokens':reasoning_tokens, 
        #         "entire respose":response}
    
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


    
