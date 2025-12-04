import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import random
from openai import AzureOpenAI
import time

argparser = argparse.ArgumentParser()
argparser.add_argument('--model', type=str, default='0613', help='model version to use: 0125, 1106, 0613, Qwen_Qwen3-8B or Qwen_Qwen3-14B')
argparser.add_argument('--version', type=str, default='new', help='which story version to use: original, new')
argparser.add_argument('--promptstyle', type=str, default='default', help='which prompt style to use: default, concise, detailed')
argparser.add_argument('--ordering', type=str, default='default', help='which ordering to use: ab, ba, random')
argparser.add_argument('--verbose', action='store_true', help='whether to print verbose output')
args = argparser.parse_args()


versions = {'0125':{'resource_name':'0125-Preview', 'deployment_name':'0125-Preview'},
            '1106':{'resource_name':'MMResearch', 'deployment_name':'gpt-4-1106-Preview'},
            '0613':{'resource_name':'0613', 'deployment_name':'0613'}}
if not args.model.startswith('Qwen'):
    id=args.model

no=args.version
start_full_run = time.time()

# Helper function to return the generated response of the model in a clean format
def clean_text(text: str) -> str:
    if not text:
        return text
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.strip("`").strip()
    if len(text) >= 2 and (
        (text[0] == '"' and text[-1] == '"')
        or (text[0] == "“" and text[-1] == "”")
    ):
        text = text[1:-1].strip()
    return text

if args.model.startswith('Qwen'):
    print(f"Loading model {args.model}...")
    MAX_NEW_TOKENS = 1024 #512

    # Check available GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU memory available: {gpu_memory:.2f} GB", flush=True)
    # Clear any cached memory
    torch.cuda.empty_cache()

    # Prepare loading kwargs
    load_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": "auto" if torch.cuda.is_available() else "cpu",
    }

    # Try to use Flash Attention 2 for faster inference
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            attn_implementation="flash_attention_2",
            **load_kwargs
        )
        print("Model loaded with Flash Attention 2", flush=True)
    except Exception as e:
        print(f"Flash Attention 2 not available ({e}), using default attention", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            **load_kwargs
        )
        print("Model loaded with default attention", flush=True)

    print(f"Model device: {next(model.parameters()).device}", flush=True)
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"GPU memory allocated: {allocated:.2f} GB, reserved: {reserved:.2f} GB", flush=True)
    print("Model loaded. Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    print(f"Model {args.model} and tokenizer loaded.", flush=True)

    # Set model to eval mode and disable gradients for inference
    model.eval()
    torch.set_grad_enabled(False)
else:
    client = AzureOpenAI(
        azure_endpoint = os.getenv(f"AZURE_OPENAI_ENDPOINT_{id}"), 
        api_key=os.getenv(f"AZURE_OPENAI_API_KEY_{id}"),  
        api_version="2024-02-01"
    )
    print(os.getenv(f"AZURE_OPENAI_ENDPOINT_{id}"))

# load json dict
with open(f'all_tasks_dict_{args.version}_answers.json', 'r') as f:
    story_dict = json.load(f)

if not args.model.startswith('Qwen'):
    gpt_responses = {}

    for k in story_dict:

        if k.startswith("Attn"):
            continue  # skip attention tasks

        gpt_responses[k] = {}
        story_1 = story_dict[k]['Story_1']
        story_a = story_dict[k]['Story_A']
        story_b = story_dict[k]['Story_B']
        prompt_1 = f"Consider the following story:\n\nStory 1: {story_1}\n\nNow consider two more stories:\n\nStory A: {story_a}\n\nStory B: {story_b}\n\nWhich of Story A and Story B is a better analogy to Story 1? Is the best answer Story A, Story B, or both are equally analogous?"
        prompt_2 = f"Consider the following story:\n\nStory 1: {story_1}\n\nNow consider two more stories:\n\nStory A: {story_b}\n\nStory B: {story_a}\n\nWhich of Story A and Story B is a better analogy to Story 1? Is the best answer Story A, Story B, or both are equally analogous?"
        response = client.chat.completions.create(
            model= versions[id]['deployment_name'],#"gpt-4-1106-Preview", # model = "deployment_name".
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_1},
                ]
        )
        gpt_responses[k]['order_1'] = response.choices[0].message.content
        
        response = client.chat.completions.create(
            model= versions[id]['deployment_name'], # model = "deployment_name".
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_2},
                ]
        )
        gpt_responses[k]['order_2'] = response.choices[0].message.content

    json_string = json.dumps(gpt_responses, indent=2)

    with open(f'gpt_results/gpt_{id}_responses_dict_{no}.json', 'w') as json_f:
        json_f.write(json_string)
        
elif args.model.startswith('Qwen'):
    print(f"Generating responses with model {args.model}...", flush=True)
    model_responses = {}

    model_responses["ordering"] = args.ordering
    model_responses["promptstyle"] = args.promptstyle

    for k in story_dict:

        if k.startswith("Attn"):
            continue  # skip attention tasks

        print(f"------------ {k} ------------")
        start = time.time()
        model_responses[k] = {}
        story_1 = story_dict[k]['Story_1']
        story_a = story_dict[k]['Story_A']
        story_b = story_dict[k]['Story_B']
        correct_ind = story_dict[k]["correct"]
        
        if args.ordering == 'ab':
            prompt = f"Consider the following story:\n\nStory 1: {story_1}\n\nNow consider two more stories:\n\nStory A: {story_a}\n\nStory B: {story_b}\n\nWhich of Story A and Story B is a better analogy to Story 1? Is the best answer Story A, Story B, or both are equally analogous?"
        elif args.ordering == 'ba':    
            prompt = f"Consider the following story:\n\nStory 1: {story_1}\n\nNow consider two more stories:\n\nStory A: {story_b}\n\nStory B: {story_a}\n\nWhich of Story A and Story B is a better analogy to Story 1? Is the best answer Story A, Story B, or both are equally analogous?"
            correct_ind = 1 - correct_ind  # flip the correct index
        elif args.ordering == "random":
            if random.random() < 0.5:
                prompt = f"Consider the following story:\n\nStory 1: {story_1}\n\nNow consider two more stories:\n\nStory A: {story_a}\n\nStory B: {story_b}\n\nWhich of Story A and Story B is a better analogy to Story 1? Is the best answer Story A, Story B, or both are equally analogous?"
            else:
                prompt = f"Consider the following story:\n\nStory 1: {story_1}\n\nNow consider two more stories:\n\nStory A: {story_b}\n\nStory B: {story_a}\n\nWhich of Story A and Story B is a better analogy to Story 1? Is the best answer Story A, Story B, or both are equally analogous?"
                correct_ind = 1 - correct_ind  # flip the correct index

        if args.promptstyle == "analogical":
            prompt += "\n\nBefore answering, recall 3 relevant examples, then provide your answer."

        if args.verbose or k == "Task 1":
            print(f"Prompt for task {k}:\n{prompt}\n", flush=True)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        # Tokenize
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        pad_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        # Generate
        with torch.inference_mode():  # Faster than torch.no_grad()
            gen = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=pad_id,
                # use_cche=True,  # Enable KV caching for faster generation
                # num_beams=1,  # Greedy decoding (faster than beam search)
            )
        end = time.time()
        out = tokenizer.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        clean_out = clean_text(out)
        if args.verbose or k == "Task 1":
            print(f"Full Qwen output ({end-start:.2f} seconds): {clean_out}", flush=True)
        response_text = clean_out
        
        # Clean up GPU memory after generation
        del inputs, gen
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model_responses[k]['response'] = response_text
        model_responses[k]['correct_ind'] = correct_ind
    
    json_string = json.dumps(model_responses, indent=2)

    with open(f'../qwen_results/{args.model.replace("/", "_")}_{args.promptstyle}_{args.ordering}_responses_{no}.json', 'w') as json_f:
        json_f.write(json_string)
else: 
    print("Model not recognized.")

end_full_run = time.time()
print(f"Total time for full run: {end_full_run - start_full_run:.2f} seconds.")