import numpy as np
import pandas as pd
import argparse
import json

argparser = argparse.ArgumentParser()
argparser.add_argument('--model', type=str, required=True, help='Model name (e.g., GPT-4, Qwen)')
argparser.add_argument('--promptstyle', type=str, default='default', help='Prompt style used')
argparser.add_argument('--version', type=str, default='new', help='Story version: original or new')
argparser.add_argument('--ordering', type=str, default='default', help='Ordering used: ab, ba, random')
args = argparser.parse_args()

def extract_answer(response):
    """
    Extract the answer choice from GPT response text.
    Returns 0 for Story A, 1 for Story B, or None if unclear.
    """
    response = response.lower()
    if 'story a' in response and 'story b' in response:
        # Check which story is indicated as better analogy
        if 'story a' in response.split('story b')[0]:
            # Story A mentioned first
            return 0
        elif 'story b' in response.split('story a')[0]:
            # Story B mentioned first
            return 1
    elif 'story a' in response:
        return 0
    elif 'story b' in response:
        return 1
    return None


if args.model.startswith('Qwen'):
    fname = f'qwen_results/{args.model}_{args.promptstyle}_{args.ordering}_responses_{args.version}.json'
else:
    fname = f'gpt_results/gpt_{args.model}_responses_dict_{args.version}.json'

with open(fname, 'r') as f:
    responses = json.load(f)

total_correct = 0

for k in responses:
    if k == "ordering" or k == "promptstyle":
        continue

    response_text = responses[k]['response']
    predicted = extract_answer(response_text)
    correct = responses[k]['correct']

    print(f'Task: {k}, Predicted: {predicted}, Correct: {correct}')
    
    if predicted == correct:
        total_correct += 1

accuracy = total_correct / (len(responses) - 2)  # exclude metadata entries
print(f'Model: {args.model}, Version: {args.version}, Prompt Style: {args.promptstyle}, Ordering: {args.ordering} => Accuracy: {accuracy:.2f}')