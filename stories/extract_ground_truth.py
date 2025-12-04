"""
Extract ground truth (correct answer) for story analogy tasks.

Since the JSON files don't explicitly mark which story is correct,
we infer it from the human data CSVs and cross-reference with GPT responses.
"""

import json
import pandas as pd
import numpy as np
from collections import Counter

# Load human data to infer ground truth
human_orig = pd.read_csv('human_data/data_stories_original.csv')
human_rewritten = pd.read_csv('human_data/data_stories_rewritten.csv')

# Load GPT responses
with open('gpt_results/gpt_0613_responses_dict_orig.json', 'r') as f:
    gpt_orig_responses = json.load(f)

with open('gpt_results/gpt_0613_responses_dict_new.json', 'r') as f:
    gpt_new_responses = json.load(f)

# Load task definitions
with open('gpt_experiment/all_tasks_dict_orig.json', 'r') as f:
    tasks_orig = json.load(f)

with open('gpt_experiment/all_tasks_dict_new.json', 'r') as f:
    tasks_new = json.load(f)

def extract_gpt_choice(response_text):
    """
    Extract whether GPT chose Story A or Story B from its response text.
    Returns 0 for A, 1 for B, or None if unclear.
    """
    text_lower = response_text.lower()
    
    # Look for explicit "Story A" or "Story B" mentions
    has_story_a = 'story a' in text_lower
    has_story_b = 'story b' in text_lower
    
    # Find which comes first or is emphasized
    idx_a = text_lower.find('story a')
    idx_b = text_lower.find('story b')
    
    # Look for "the best answer is"
    best_answer_idx = text_lower.find('the best answer is')
    if best_answer_idx != -1:
        snippet = text_lower[best_answer_idx:best_answer_idx+100]
        if 'story a' in snippet:
            return 0
        elif 'story b' in snippet:
            return 1
    
    # If one clearly mentioned more emphatically
    if has_story_a and has_story_b:
        # Check which is mentioned first as the answer
        if idx_a < idx_b:
            return 0
        else:
            return 1
    elif has_story_a:
        return 0
    elif has_story_b:
        return 1
    
    return None

def infer_ground_truth_from_human_data(human_df):
    """
    Infer ground truth by finding majority response for each task.
    Assumes human consensus indicates the correct answer.
    """
    ground_truth = {}
    
    # Group by task and find most common response
    for task_id in human_df['task'].unique():
        task_data = human_df[human_df['task'] == task_id]
        responses = task_data['response'].values
        
        # Most common response
        counter = Counter(responses)
        most_common = counter.most_common(1)[0][0]
        ground_truth[f"Task {task_id}"] = most_common
    
    return ground_truth

def infer_ground_truth_from_gpt(gpt_responses):
    """
    Extract ground truth by analyzing GPT's choices across two orderings.
    If GPT consistently picks the same story regardless of ordering, that's likely correct.
    """
    ground_truth = {}
    
    for task_id, task_data in gpt_responses.items():
        order_1_response = task_data.get('order_1', '')
        order_2_response = task_data.get('order_2', '')
        
        choice_1 = extract_gpt_choice(order_1_response)
        choice_2 = extract_gpt_choice(order_2_response)
        
        # In order_1: Story A is story_a, Story B is story_b
        # In order_2: Story A is story_b, Story B is story_a
        # So if GPT picked the same underlying story, choices should differ
        
        # For now, just record what GPT chose
        ground_truth[task_id] = {
            'order_1_choice': choice_1,  # 0=A (story_a), 1=B (story_b)
            'order_2_choice': choice_2,  # 0=A (story_b), 1=B (story_a)
        }
    
    return ground_truth

# Infer from human data
print("=" * 60)
print("GROUND TRUTH FROM HUMAN DATA (Original Stories)")
print("=" * 60)
gt_human_orig = infer_ground_truth_from_human_data(human_orig)
for task, answer in sorted(gt_human_orig.items()):
    print(f"{task}: Story {'A' if answer == 0 else 'B'} (response={answer})")

print("\n" + "=" * 60)
print("GROUND TRUTH FROM HUMAN DATA (Rewritten Stories)")
print("=" * 60)
gt_human_rewritten = infer_ground_truth_from_human_data(human_rewritten)
for task, answer in sorted(gt_human_rewritten.items()):
    print(f"{task}: Story {'A' if answer == 0 else 'B'} (response={answer})")

print("\n" + "=" * 60)
print("GROUND TRUTH FROM GPT RESPONSES (Original)")
print("=" * 60)
gt_gpt_orig = infer_ground_truth_from_gpt(gpt_orig_responses)
for task, choices in sorted(gt_gpt_orig.items()):
    o1 = 'A' if choices['order_1_choice'] == 0 else 'B' if choices['order_1_choice'] == 1 else '?'
    o2 = 'A' if choices['order_2_choice'] == 0 else 'B' if choices['order_2_choice'] == 1 else '?'
    print(f"{task}: order_1 picked {o1}, order_2 picked {o2}")

print("\n" + "=" * 60)
print("GROUND TRUTH FROM GPT RESPONSES (New)")
print("=" * 60)
gt_gpt_new = infer_ground_truth_from_gpt(gpt_new_responses)
for task, choices in sorted(gt_gpt_new.items()):
    o1 = 'A' if choices['order_1_choice'] == 0 else 'B' if choices['order_1_choice'] == 1 else '?'
    o2 = 'A' if choices['order_2_choice'] == 0 else 'B' if choices['order_2_choice'] == 1 else '?'
    print(f"{task}: order_1 picked {o1}, order_2 picked {o2}")

# Save ground truth
with open('ground_truth.json', 'w') as f:
    json.dump({
        'human_original': {k: int(v) for k, v in gt_human_orig.items()},
        'human_rewritten': {k: int(v) for k, v in gt_human_rewritten.items()},
    }, f, indent=2)

print("\n✓ Ground truth saved to ground_truth.json")
