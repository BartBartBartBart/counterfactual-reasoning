import openai
import numpy as np
import builtins
import argparse
import os
import time
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

start = time.time()

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--promptstyle', help='Give a prompt style: human, minimal, hw, webb, webbplus')
parser.add_argument('--model', help='give model name', default=None)
parser.add_argument('--verbose', action='store_true', help="Print verbose output.")
parser.add_argument('--debug', action='store_true', help="Debug mode - do not load large models")
args = parser.parse_args()

def check_path(path):
	if not os.path.exists(path):
		os.mkdir(path)

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

def create_prompt(promptstyle, prob, alph_string):
	prompt=''
	if promptstyle not in ["minimal", "hw", "webb","webbplus", "analogical"]:			
		prompt+='Use the following alphabet to guess the missing piece.\n\n' \
			+ alph_string \
			+ '\n\nNote that the alphabet may be in an unfamiliar order. Complete the pattern using this order.\n\n'
	elif promptstyle == 'minimal':			
		prompt+='Use the following alphabet to complete the pattern.\n\n' \
			+ alph_string \
			+ '\n\nNote that the alphabet may be in an unfamiliar order. Complete the pattern using this order. Answer with only the final answer and nothing else. Put your final answer between double brackets.\n\n'
	elif promptstyle == 'hw':			
		prompt+='Use this fictional alphabet: \n\n' \
			+ alph_string \
			+ "\n\nLet's try to complete the pattern:\n\n"
	elif promptstyle == "webb":
		prompt += "Let's try to complete the pattern:\n\n"
	elif promptstyle == "webbplus":
		prompt += "Let's try to complete the pattern. Just give the letters that complete the pattern and nothing else at all. Do not describe the pattern.\n\n"
	elif promptstyle == "analogical":
		prompt += "Use the following alphabet to complete the pattern.\n\n"
		prompt += alph_string \
			+ '\n\nFirst, describe 3 relevant exemplars that are distinct from this problem, then give the final answer. Answer with only the examples and the final answer with no further explanation. Put your final answer between double brackets. Note that the alphabet may be in an unfamiliar order. Complete the pattern using this order.\n\n'
	prompt += '['
	for i in range(len(prob[0][0])):
		prompt += str(prob[0][0][i])
		if i < len(prob[0][0]) - 1:
			prompt += ' '
	prompt += '] ['
	for i in range(len(prob[0][1])):
		prompt += str(prob[0][1][i])
		if i < len(prob[0][1]) - 1:
			prompt += ' '
	prompt += ']\n['
	for i in range(len(prob[1][0])):
		prompt += str(prob[1][0][i])
		if i < len(prob[1][0]) - 1:
			prompt += ' '
	if promptstyle in ["minimal", "hw", "webb","webbplus", "analogical"]:
		prompt += '] ['
	else:
		prompt += '] [ ? ]'
	if promptstyle == "human":
		messages = [{'role': 'system', 'content':'You are able to solve letter-string analogies'},
						{'role': 'user', 'content': "In this study, you will be presented with a series of patterns involving alphanumeric characters, together with an example alphabet.\n\n" +
						"Note that the alphabet may be in an unfamiliar order.\n" + 
						"Each pattern will have one missing piece marked by [ ? ].\n"+
						"For each pattern, you will be asked to guess the missing piece.\n" +
						"Use the given alphabet when guessing the missing piece.\n" +
						"You do not need to include the '[ ]' or spaces between letters in your response.\n\n"+
						"a b c h e f g d i j k l m n o p q r s t u v w x y z \n\n" +
						"[a a a] [b b b]\n[c c c] [ ? ]"},
						{'role':'assistant', 'content': 'h h h'},
						{'role':'user', 'content': "In this case, the missing piece is 'h h h'\nNote that in the given alphabet, 'b' is the letter after 'a' and 'h' is the letter after 'c'"},
						{'role':'user', 'content':prompt}]
	elif promptstyle in ["minimal", "hw", "webb","webbplus", "analogical"]:
		messages = [{'role': 'system', 'content':'You are able to solve letter-string analogies'},
						{'role':'user', 'content':prompt}]
	else:
		print("please enter a promptstyle")
	
	return messages

def get_probs(messages, model, tokenizer):
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
		outputs = model.generate(
			**inputs,
			max_new_tokens=MAX_NEW_TOKENS,
			do_sample=False,
			temperature=1.0, # 1 for accurate probabilities, because 0 can lead to overconfident outputs
			top_p=1.0,
			eos_token_id=tokenizer.eos_token_id,
			pad_token_id=pad_id,
			use_cache=True,  # Enable KV caching for faster generation
			num_beams=1,  # Greedy decoding (faster than beam search)
			output_scores=True,  # KEY: Returns logits for each step
			return_dict_in_generate=True,  # KEY: Returns structured output
		)
	
	# Extract generated tokens and scores
	generated_ids = outputs.sequences[0, inputs["input_ids"].shape[1]:]
	scores = outputs.scores  # Tuple of tensors, one per generation step
	full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
	return generated_ids, scores, full_text

def exemplar_probs(tokenizer, scores, generated_ids, verbose=False):
	# Find probabilities of exemplars using the actual generated token IDs
	prob_per_exemplar = []
	total_probs = []
	in_exemplar = False
	opening_count = 0
	closing_count = 0
	exemplar_count = 0
	exemplar_token_indices = []  # List of indices into generated_ids for current exemplar

	if verbose:
		print(f"Total generated tokens: {len(generated_ids)}")
		print(f"Generated text: {tokenizer.decode(generated_ids)}")
		print(f"Total scores available: {len(scores)}\n")

	# Loop through actual generated tokens
	for token_idx, token_id in enumerate(generated_ids):
		if exemplar_count == 3:
			break

		token_text = tokenizer.decode([token_id])

		# Check for opening bracket - start/continue exemplar
		if "[" in token_text:
			in_exemplar = True
			opening_count += 1
			exemplar_token_indices.append(token_idx)
			# print(f"Opening bracket at generated token {token_idx}, opening_count={opening_count}")
			continue

		# Collect tokens while in exemplar
		if in_exemplar:
			exemplar_token_indices.append(token_idx)

			# Check for closing bracket
			if "]" in token_text:
				closing_count += 1
				# print(f"Closing bracket at generated token {token_idx}, closing_count={closing_count}")

				# Check if exemplar is complete (2 opening, 2 closing)
				if opening_count == closing_count and opening_count == 2:
					# Calculate probabilities for this exemplar
					prob_per_word = []
					logprob = 0.0
					exemplar_count += 1

					if verbose:
						print(f"\nComplete exemplar with tokens: {[tokenizer.decode([generated_ids[i]]) for i in exemplar_token_indices]}")

					for tok_idx in exemplar_token_indices:
						tok_id = generated_ids[tok_idx].item() if hasattr(generated_ids[tok_idx], 'item') else generated_ids[tok_idx]
						step = tok_idx

						if step >= 0 and step < len(scores):
							token_logprob = torch.log_softmax(scores[step][0], dim=-1)[tok_id].item()
							token_decoded = tokenizer.decode([tok_id])
							token_prob = np.exp(token_logprob)
							prob_per_word.append((token_decoded, token_prob))
							if verbose:
								print(f"  token '{token_decoded}' (id {tok_id}) at step {step}, logprob {token_logprob:.4f}, prob {token_prob:.2e}")
							logprob += token_logprob
						else:
							print(f"  WARNING: step {step} out of range for scores (len={len(scores)})")

					# Store results for this exemplar
					total_prob = np.exp(logprob)
					total_probs.append(total_prob)
					prob_per_exemplar.append(prob_per_word)

					if verbose:
						print(f"Total probability for this exemplar: {total_prob}\n")

					# Reset for next exemplar
					exemplar_token_indices = []
					opening_count = 0
					closing_count = 0
					in_exemplar = False

	return prob_per_exemplar, total_probs

if args.promptstyle == "webb" and int(args.num_permuted) >1:
	print("promptstyle webb can only be used with an unpermuted alphabet")
	sys.exit()
	
# Load Qwen3  
elif args.model is not None and not args.debug:
	print(f"Loading model {args.model}...")
	MAX_NEW_TOKENS = 128  # Reduced from 256 - letterstring answers are short

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
		"device_map": "cuda:0" if torch.cuda.is_available() else "cpu",
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

average_exemplar_probs = {}

# Collect average exemplar probability for each number of permuted letters
for num_permuted in [1, 2, 5, 10, 20]:
	exemplar_probs_list = []
	
	all_prob = np.load(f'./problems/nogen/all_prob_{num_permuted}_7_human.npz', allow_pickle=True)['all_prob']

	for alph in all_prob.item().keys():
		if (all_prob.item()[alph]['shuffled_letters'] is not None):
			shuffled_letters = builtins.list(all_prob.item()[alph]['shuffled_letters'])	
		else:
			shuffled_letters = None
			
		shuffled_alphabet = builtins.list(all_prob.item()[alph]['shuffled_alphabet'])
		prob_types = builtins.list(all_prob.item()[alph].keys())[2:] # first two items are list of shuffled letters and shuflled alphabet: skip this
		N_prob_types = len(prob_types) # -1 # minus 1 to skip attention problems
		alph_string = ' '.join(shuffled_alphabet)

		# Evaluate
		N_trials_per_prob_type = 10
		count = 0
		for p in range(N_prob_types):
			if prob_types[p] == 'attn':
				# SKIP ATTENTION PROBLEMS
				continue

			print(f"Problem type: {prob_types[p]} - {str(p+1)}/{str(N_prob_types)}", flush=True)

			for t in range(N_trials_per_prob_type):
				print('trial ' + str(t+1) + ' of ' + str(N_trials_per_prob_type) + '...', flush=True)
				prob = all_prob.item()[alph][prob_types[p]]['prob'][t]
				full_tgt_letters = all_prob.item()[alph][prob_types[p]]['tgt_letters'][t]
				current_target = all_prob.item()[alph][prob_types[p]]['prob'][t][1][1]

				# Create prompt
				messages = create_prompt(args.promptstyle, prob, alph_string)

				# If verbose or first trial
				if args.verbose or t == 0:
					print("\n=== PROMPT ===\n", flush=True)
					print(f"System message: {messages[0]['content']}\n", flush=True)
					print(f"User message: {messages[1]['content']}\n", flush=True)
					print("\n--- TARGET LETTERS ---\n", flush=True)
					print(current_target, flush=True)

				# Get response
				if args.model.startswith("Qwen"):
					generated_ids, scores, full_text = get_probs(messages, model, tokenizer)

					if args.verbose or t == 0:
						print("\n=== RESPONSE ===\n", flush=True)
						clean_out = clean_text(full_text)
						print(clean_out, flush=True)

					if args.verbose or t == 0:
						print("Calculating probabilities...", flush=True)						
					probs_per_exemplar, total_probs = exemplar_probs(tokenizer, scores, generated_ids, args.verbose)
					# exemplar_probs_list.append(total_probs)
					exemplar_probs_list.extend(total_probs)

					# Clean up GPU memory after generation
					del generated_ids, scores, full_text
					if torch.cuda.is_available():
						torch.cuda.empty_cache()

	average_exemplar_probs[num_permuted] = exemplar_probs_list
	print(f"Completed exemplar probabilities for {num_permuted} permuted letters.", flush=True)
	print(f"Average exemplar probability: {np.mean(exemplar_probs_list):.6f}", flush=True)

# Print average exemplar probabilities
for num_permuted, probs_list in average_exemplar_probs.items():
	avg_prob = np.mean(probs_list)
	print(f"Num permuted letters: {num_permuted}, Average exemplar probability: {avg_prob:.6f}", flush=True)

end = time.time()
print(f"Total time: {end-start} seconds.", flush=True)