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
parser.add_argument('--gen', help='gen or nogen', default='nogen')
parser.add_argument('--use_saved', action='store_true', help="Use saved responses instead of generating new ones.")
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

def check_partly_correct(true, pred):
    """
    Manual check for partly correct predictions (pred != true, but true in pred.)
    e.g. true = 'abcd', pred = 'abc][abcd' --> correct
    Returns True if partly correct, False otherwise. 
    """
    # if multiple indices, take last one
    indices = [i for i in range(len(pred)) if pred.startswith(true, i)]
    index = indices[-1] if len(indices) > 1 else indices[0] if len(indices) == 1 else -1
    if index != -1:
        before = pred[index-1] if index > 0 else ' '
        after = pred[index+len(true)] if index + len(true) < len(pred) else ' '
        if before in ['['] and after in [' ', ']']:
            # print(f"partly correct: True: {true}, Pred: {pred}")
            return True
        # print(f"Not partly correct due to surrounding chars: True: {true}, Pred: {pred}, before {before}, after {after}")
    return False

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

def determine_correctness(pred, current_target):
	"""Filter output for answer and compare with ground truth"""
	# Filter the answer, take only the content inside double brackets [[ answer ]]
	if '[[' in pred and ']]' in pred:
		start_idx = pred.index('[[') + 2
		end_idx = pred.index(']]')
		pred = pred[start_idx:end_idx].strip()
		pred = pred.strip(" '").replace(" ", "").lower()
	else:
		pred = pred.replace(" ", "").lower()
	# true = full_tgt_letters
	true = current_target
	if type(true[0]) == np.int64:
		true = [str(x) for x in true]
	true = ''.join(true).lower()
	if args.verbose:
		print(f'Pred: {pred}, True: {true}')
	if pred == true:
		correct = True
	elif true in pred:
		correct = check_partly_correct(true, pred)
	else:
		correct = False
	return correct, true

def find_final_answer(generated_ids, verbose=False):
	"""Search the answer for double or single brackets."""
	# Search for opening double bracket
	final_answer_start = None
	for token_idx, token_id in enumerate(generated_ids):
		token_text = tokenizer.decode([token_id])
		if "[[" in token_text:
			final_answer_start = token_idx
			break
	
	if final_answer_start is None:
		if verbose:
			print("No opening double bracket found for final answer\n")
		# Take last open bracket as start
		for token_idx in range(len(generated_ids)-1, -1, -1):
			token_text = tokenizer.decode([generated_ids[token_idx]])
			if "[" in token_text:
				final_answer_start = token_idx
				if verbose:
					print(f"Using last single opening bracket at token index {final_answer_start} as start for final answer\n")
				break
		if final_answer_start is None:
			return None, None
	
	# Search for closing double bracket
	final_answer_end = None
	for token_idx in range(final_answer_start + 1, len(generated_ids)):
		token_text = tokenizer.decode([token_id for token_id in generated_ids[final_answer_start:token_idx+1]])
		if "]]" in token_text:
			final_answer_end = token_idx
			break
	
	if final_answer_end is None:
		if verbose:
			print("No closing double bracket found for final answer\n")
		# Take last closing bracket as end
		for token_idx in range(len(generated_ids)-1, final_answer_start, -1):
			token_text = tokenizer.decode([generated_ids[token_idx]])
			if "]" in token_text:
				final_answer_end = token_idx
				if verbose:
					print(f"Using last single closing bracket at token index {final_answer_end} as end for final answer\n")
				break
		if final_answer_end is None:
			return None, None

	return final_answer_start, final_answer_end	

def calculate_token_probability(tokenizer, scores, generated_ids, token_indices, verbose=False, label=""):
	"""Calculate total probability for a sequence of token indices."""
	logprob = 0.0
	
	if verbose:
		print(f"\n{label} tokens: {[tokenizer.decode([generated_ids[i]]) for i in token_indices]}")
	
	for tok_idx in token_indices:
		tok_id = generated_ids[tok_idx].item() if hasattr(generated_ids[tok_idx], 'item') else generated_ids[tok_idx]
		step = tok_idx
		
		if step >= 0 and step < len(scores):
			token_logprob = torch.log_softmax(scores[step][0], dim=-1)[tok_id].item()
			token_decoded = tokenizer.decode([tok_id])
			token_prob = np.exp(token_logprob)
			if verbose:
				print(f"  token '{token_decoded}' (id {tok_id}) at step {step}, logprob {token_logprob:.4f}, prob {token_prob:.2e}")
			logprob += token_logprob
		else:
			print(f"  WARNING: step {step} out of range for scores (len={len(scores)})")
	
	total_prob = np.exp(logprob)
	if verbose:
		print(f"Total probability for {label}: {total_prob}\n")
	
	return total_prob

def extract_exemplar_probs(tokenizer, output_ap, verbose=False):
	"""Extract probabilities for the first 3 exemplars."""
	prob_per_exemplar = []
	total_probs = []
	in_exemplar = False
	opening_count = 0
	closing_count = 0
	exemplar_count = 0
	exemplar_token_indices = []

	generated_ids = output_ap["generated_ids"]
	scores = output_ap["scores"]

	if verbose:
		print(f"Total generated tokens: {len(generated_ids)}")
		print(f"Generated text: {tokenizer.decode(generated_ids)}")
		print(f"Total scores available: {len(scores)}\n")

	# Loop through actual generated tokens
	for token_idx, token_id in enumerate(generated_ids):
		if exemplar_count >= 3:
			break
			
		token_text = tokenizer.decode([token_id])

		# Check for opening bracket - start/continue exemplar
		if "[" in token_text:
			in_exemplar = True
			opening_count += 1
			exemplar_token_indices.append(token_idx)
			continue

		# Collect tokens while in exemplar
		if in_exemplar:
			exemplar_token_indices.append(token_idx)

			# Check for closing bracket
			if "]" in token_text:
				closing_count += 1

				# Check if exemplar is complete (2 opening, 2 closing)
				if opening_count == closing_count and opening_count == 2:
					prob_per_word = []
					exemplar_count += 1
					
					# Calculate probability for this exemplar
					total_prob = calculate_token_probability(tokenizer, scores, generated_ids, exemplar_token_indices, 
														   verbose=verbose, label=f"exemplar {exemplar_count}")
					total_probs.append(total_prob)
					prob_per_exemplar.append(prob_per_word)

					# Reset for next exemplar
					exemplar_token_indices = []
					opening_count = 0
					closing_count = 0
					in_exemplar = False

	return prob_per_exemplar, total_probs

def extract_final_answer_prob(tokenizer, output, verbose=False):
	"""Extract probability for the final answer between double brackets."""
	final_answer_prob = None

	generated_ids = output["generated_ids"]
	scores = output["scores"]
	
	# Filter for final answer
	final_answer_start, final_answer_end = find_final_answer(generated_ids, verbose)

	# No final answer found
	if final_answer_start is None or final_answer_end is None: 
		return final_answer_prob

	# Extract token indices between brackets
	final_answer_token_indices = list(range(final_answer_start, final_answer_end + 1))
	
	# Calculate probability
	final_answer_prob = calculate_token_probability(tokenizer, scores, generated_ids, final_answer_token_indices,
													verbose=verbose, label="final answer")

	return final_answer_prob

def get_relevant_tokens(final_answer_start, final_answer_end, correct_answer, output):
	generated_ids = output["generated_ids"]
	scores = output["generated_ids"]

	# Identify the generated final inner answer (the part between the last '[' and the next ']')
	last_open = None
	for token_idx in range(final_answer_start, final_answer_end + 1):
		token_text = tokenizer.decode([generated_ids[token_idx]])
		if "[" in token_text:
			last_open = token_idx
	
	if last_open is None:
		# fallback: use the range between final_answer_start and final_answer_end
		gen_answer_start = final_answer_start + 1
	else:
		gen_answer_start = last_open + 1
	
	# Find corresponding closing bracket
	gen_answer_end = None
	for token_idx in range(gen_answer_start, final_answer_end + 1):
		token_text = tokenizer.decode([generated_ids[token_idx]])
		if "]" in token_text:
			gen_answer_end = token_idx
			break
	
	if gen_answer_end is None:
		# fallback: use up to final_answer_end (exclusive)
		gen_answer_end = final_answer_end
	
	# Indices of tokens that form the generated final answer (excluding brackets)
	gen_indices = list(range(gen_answer_start, gen_answer_end))

	# Tokenize correct answer as if it was written with spaces: 'pqrt' -> tokenize as 'p q r t'
	# First character without space, subsequent characters with leading space
	correct_answer_tokens = []
	clean_answer = correct_answer.replace(" ", "")  # Remove any existing spaces
	for i, char in enumerate(clean_answer):
		if i <= len(gen_indices) - 1:
			gen_token_id = generated_ids[gen_indices[i]].item()
			generated_token = tokenizer.decode([gen_token_id])
			if generated_token.startswith(" "):
				# Subsequent characters with leading space
				char_tokens = tokenizer(" " + char, return_tensors="pt")["input_ids"][0]
			else:
				# First character without space
				char_tokens = tokenizer(char, return_tensors="pt")["input_ids"][0]
		else: 
			break				
		correct_answer_tokens.extend(char_tokens.tolist())
	correct_answer_tokens = torch.tensor(correct_answer_tokens, device=generated_ids.device)

	return gen_indices, correct_answer_tokens		


def get_ratio(tokenizer, scores, generated_ids, gen_indices, correct_answer_tokens, verbose=False):
	"""
	Calculate the following ratio using log probability:
	Ratio = p(correct answer)/p(given answer)

	In case of Analogical Prompting:
	Ratio = p(correct answer|exemplars)/p(given answer|exemplars)
	"""
	# LEFT-align the correct answer to the generated final answer
	correct_log_sum = 0.0
	gen_log_sum = 0.0 
	for i, idx in enumerate(gen_indices):
		if i >= len(correct_answer_tokens):
			break
		if idx >= len(scores):
			if verbose:
				print(f"  WARNING: no score for token index {idx}")
			continue
		logits = scores[idx][0]
		logprobs = torch.log_softmax(logits, dim=-1)
		gen_token_id = generated_ids[idx].item()
		gen_logp = logprobs[gen_token_id].item()
		correct_token_id = correct_answer_tokens[i].item()
		correct_logp = logprobs[correct_token_id].item()
		if verbose:
			gen_text = tokenizer.decode([gen_token_id])
			correct_text = tokenizer.decode([correct_token_id])
			print(f"Position {i}: generated '{gen_text}' (id {gen_token_id}) w logp={gen_logp}, correct '{correct_text}' (id {correct_token_id}), correct_logp={correct_logp}")
		gen_log_sum += gen_logp
		correct_log_sum += correct_logp
	
	# generated_answer_prob = np.exp(gen_log_sum)
	# correct_answer_prob = np.exp(correct_log_sum)
	ratio = np.exp(correct_log_sum - gen_log_sum)

	if verbose: 
		print(f"logp(correct)={correct_log_sum}, logp(given)={gen_log_sum}")
		print(f"log Ratio=logp(correct)-logp(given)={correct_log_sum}-{gen_log_sum}={correct_log_sum-gen_log_sum}")
		print(f"Ratio = np.exp(log Ratio) = np.exp({correct_log_sum-gen_log_sum}) = {ratio}")
	
	return ratio

def compare_prompting_ratios(tokenizer, output_ap, output_bl, correct_answer=None, verbose=False): 
	"""
	Compute ratio using the non-similar tokens between ap and bl.  
		
	:param tokenizer: Description
	:param output_ap: Description
	:param output_bl: Description
	:param correct_answer: Description
	:param verbose: Description
	"""
	final_answer_start_ap, final_answer_end_ap = find_final_answer(output_ap["generated_ids"], verbose)
	final_answer_start_bl, final_answer_end_bl = find_final_answer(output_bl["generated_ids"], verbose)
	gen_indices_ap, gen_indices_bl = None, None
	ratio_ap, ratio_bl = None, None
	flag_ap, flag_bl = None, None

	# If correct answer is provided, calculate probability for the correct answer specifically
	if final_answer_start_ap is not None and final_answer_end_ap is not None:
		flag_ap = None

		gen_indices_ap, correct_answer_tokens_ap = get_relevant_tokens(final_answer_start_ap, final_answer_end_ap, correct_answer, output_ap)

		if len(correct_answer_tokens_ap) < len(gen_indices_ap):
			flag_ap = "stopped late"
		elif len(correct_answer_tokens_ap) > len(gen_indices_ap):
			flag_ap = "stopped early"
		else: 
			flag_ap = "same length"
	
	# If correct answer is provided, calculate probability for the correct answer specifically
	if final_answer_start_bl is not None and final_answer_end_bl is not None:
		flag_bl = None

		gen_indices_bl, correct_answer_tokens_bl = get_relevant_tokens(final_answer_start_bl, final_answer_end_bl, correct_answer, output_bl)

		if len(correct_answer_tokens_bl) < len(gen_indices_bl):
			flag_bl = "stopped late"
		elif len(correct_answer_tokens_bl) > len(gen_indices_bl):
			flag_bl = "stopped early"
		else: 
			flag_bl = "same length"
	
	if gen_indices_ap is not None and gen_indices_bl is not None:
		if len(gen_indices_ap) != len(gen_indices_bl):
			if verbose: 
				print(f"Answer of bl and ap different length. No ratio computation.")
			return None, None, None, None

		# Focus on different tokens between ap and bl -> remove same tokens
		idx_to_remove = []
		for idx, (tok_ap, tok_bl) in enumerate(zip(gen_indices_ap, gen_indices_bl)):
			if output_ap["generated_ids"][tok_ap] == output_bl["generated_ids"][tok_bl]: 
				idx_to_remove.append(idx)
				if verbose:
					print(f"\nRemoving token {output_ap["generated_ids"][tok_ap]}")

			else:
				ap_token = tokenizer.decode([output_ap["generated_ids"][tok_ap].item()])
				bl_token = tokenizer.decode([output_bl["generated_ids"][tok_bl].item()])
				if " "+ap_token == bl_token:
					idx_to_remove.append(idx)

		to_remove = set(idx_to_remove)
		n = len(gen_indices_ap)
		n_ap_tokens = int(correct_answer_tokens_ap.size(0))
		n_bl_tokens = int(correct_answer_tokens_bl.size(0))

		keep = [i for i in range(n) if i not in to_remove and i < n_ap_tokens and i < n_bl_tokens]

		if len(keep) == 0:
			if verbose:
				print(f"All tokens have been removed. Skipping ratio calculation \n")
			return None, None, None, None

		gen_indices_ap = [gen_indices_ap[i] for i in keep]
		gen_indices_bl = [gen_indices_bl[i] for i in keep]

		idx_tensor_ap = torch.tensor(keep, dtype=torch.long, device=correct_answer_tokens_ap.device)
		idx_tensor_bl = torch.tensor(keep, dtype=torch.long, device=correct_answer_tokens_bl.device)

		correct_answer_tokens_ap = correct_answer_tokens_ap[idx_tensor_ap]
		correct_answer_tokens_bl = correct_answer_tokens_bl[idx_tensor_bl]

	# compute ratios 
	if final_answer_start_ap is not None:
		if verbose:
			print(f"\nCalculating Ratio for AP")
		ratio_ap = get_ratio(tokenizer, scores_ap, generated_ids_ap, gen_indices_ap, correct_answer_tokens_ap, verbose)
	
	if final_answer_start_bl is not None:
		if verbose: 
			print(f"\nCalculating Ratio for baseline")
		ratio_bl = get_ratio(tokenizer, scores_bl, generated_ids_bl, gen_indices_bl, correct_answer_tokens_bl, verbose)
	
	return ratio_ap, flag_ap, ratio_bl, flag_bl


if args.use_saved and (not args.model or not args.promptstyle):
	print("When using --use_saved, both --model and --promptstyle must be specified.")
	sys.exit()

if args.promptstyle == "webb" and int(args.num_permuted) >1:
	print("promptstyle webb can only be used with an unpermuted alphabet")
	sys.exit()
	
# Load Qwen3  
if args.model is not None and not args.debug and not args.use_saved:
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

# Determine generation types to process
if args.gen == "nogen":
	gen_types = ["0gen"]
	problem_dir = "nogen"
else:
	gen_types = ["1gen", "2gen", "3gen"]
	problem_dir = "gen"

# Initialize data structures
average_exemplar_probs = {gen: {} for gen in gen_types}
average_final_answer_probs = {gen: {} for gen in gen_types}
average_ratios = {gen: {} for gen in gen_types}
# response_dict_all = {} # Stores responses for all problems, together with their scores and generation_ids in npz format

# Collect average exemplar probability for each number of permuted letters
for num_permuted in [1, 2, 5, 10, 20]:
	# response_dict = {}

	# Initialize lists for this num_permuted
	exemplar_probs_list = {gen: {"analogical": {'correct': [], 'incorrect': [], 'total': []}, "minimal": {'correct': [], 'incorrect': [], 'total': []}} for gen in gen_types}
	final_answer_probs = {gen: {"analogical": {'correct': [], 'incorrect': [], 'total': []}, "minimal": {'correct': [], 'incorrect': [], 'total': []}} for gen in gen_types}
	ratios_list = {gen: {"analogical": {'same length': [], 'stopped early': [], 'stopped late': []}, "minimal": {'same length': [], 'stopped early': [], 'stopped late': []}} for gen in gen_types}

	if args.gen == "nogen":
		all_prob = np.load(f'./problems/nogen/all_prob_{num_permuted}_7_human.npz', allow_pickle=True)['all_prob']
	else:
		all_prob = np.load(f'./problems/gen/all_prob_{num_permuted}_7_gpt_human_alphs.npz', allow_pickle=True)['all_prob']

	for alph in all_prob.item().keys():
		print(alph, flush=True)

		if (all_prob.item()[alph]['shuffled_letters'] is not None):
			shuffled_letters = builtins.list(all_prob.item()[alph]['shuffled_letters'])	
		else:
			shuffled_letters = None
			
		shuffled_alphabet = builtins.list(all_prob.item()[alph]['shuffled_alphabet'])

		# response_dict[alph] = {'shuffled_letters': shuffled_letters,
		# 					   'shuffled_alphabet': shuffled_alphabet,
		# 					   'problems': {}}

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
				# messages = create_prompt(args.promptstyle, prob, alph_string)
				messages_ap = create_prompt("analogical", prob, alph_string) # Analogical Prompting
				messages_bl = create_prompt("minimal", prob, alph_string) # Baseline

				# If verbose or first trial
				if args.verbose or t == 0:
					print("\n=== PROMPT ===\n", flush=True)
					print(f"System message: {messages_ap[0]['content']}\n", flush=True)
					print(f"User message: {messages_ap[1]['content']}\n", flush=True)
					print("\n--- TARGET LETTERS ---\n", flush=True)
					print(current_target, flush=True)

				# Get response
				if args.model.startswith("Qwen") and not args.use_saved:
					generated_ids_ap, scores_ap, full_text_ap = get_probs(messages_ap, model, tokenizer)
					generated_ids_bl, scores_bl, full_text_bl = get_probs(messages_bl, model, tokenizer)
					output_ap = {"generated_ids": generated_ids_ap, "scores": scores_ap}
					output_bl = {"generated_ids": generated_ids_bl, "scores": scores_bl}
					pred_ap = clean_text(full_text_ap)
					pred_bl = clean_text(full_text_bl)

					if args.verbose or t == 0:
						print("\n=== RESPONSE AP ===\n", flush=True)
						print(pred_ap, flush=True)
						print("\n=== RESPONSE Baseline ===\n", flush=True)
						print(pred_bl, flush=True)
					
					# Determine correctness
					correct_ap, correct_answer = determine_correctness(pred_ap, current_target)
					correct_bl, correct_answer = determine_correctness(pred_bl, current_target)

					if args.verbose:
						print(f"Final decision on correctness for AP: {correct_ap}", flush=True)
						print(f"Final decision on correctness for BL: {correct_bl}", flush=True)

					probs_per_exemplar, total_exemplar_probs = extract_exemplar_probs(tokenizer, output_ap, args.verbose or t == 0)
					final_answer_prob_ap = extract_final_answer_prob(tokenizer, output_ap, args.verbose or t == 0)
					final_answer_prob_bl = extract_final_answer_prob(tokenizer, output_bl, args.verbose or t == 0)

					ratio_ap, flag_ap, ratio_bl, flag_bl = None, None, None, None
					if pred_ap != pred_bl:
						ratio_ap, flag_ap, ratio_bl, flag_bl = compare_prompting_ratios(tokenizer, output_ap, output_bl, correct_answer, args.verbose or t == 0)
					elif args.verbose: 
						print(f"Skipping ratio calculation because answers are the same.")	

					if args.gen == "nogen":
						gen_key = "0gen"
					elif prob_types[p].startswith("2gen"):
						gen_key = "2gen"
					elif prob_types[p].startswith("3gen"):
						gen_key = "3gen"
					else:
						gen_key = "1gen"

					# response_dict[alph]['problems'][(prob_types[p], t)] = {
					# 	'prompt': messages,
					# 	'generated_ids': generated_ids.cpu().numpy(),
					# 	'scores': [s.cpu().numpy() for s in scores],
					# 	'full_text': full_text,
					# 	'predicted_answer': pred,
					# 	'correct': correct,
					# 	'final_answer_prob': final_answer_prob
					# }

					# if args.promptstyle == "analogical":
						# # Store exemplar probabilities
						# response_dict[alph]['problems'][(prob_types[p], t)]['exemplar_probs'] = probs_per_exemplar
						# response_dict[alph]['problems'][(prob_types[p], t)]['total_exemplar_probs'] = total_exemplar_probs

					for method in ["analogical", "minimal"]:
						if method == "analogical":
							correct_key  = "correct" if correct_ap else "incorrect"
							final_answer_prob = final_answer_prob_ap
							if ratio_ap is not None: 
								ratio = ratio_ap
								flag = flag_ap
							else: 
								ratio = None

							exemplar_probs_list[gen_key][method][correct_key].extend(total_exemplar_probs)
							exemplar_probs_list[gen_key][method]["total"].extend(total_exemplar_probs)

						elif method == "minimal":
							correct_key = "correct" if correct_bl else "incorrect"
							final_answer_prob = final_answer_prob_bl
							if ratio_bl is not None: 
								ratio = ratio_bl
								flag = flag_bl
							else:
								ratio = None

						if final_answer_prob is not None:
							final_answer_probs[gen_key][method][correct_key].append(final_answer_prob)
							final_answer_probs[gen_key][method]["total"].append(final_answer_prob)

						if ratio is not None: 
							ratios_list[gen_key][method][flag].append(ratio)
							if args.verbose:
								print(f"Ratio for {method} with flag {flag}: {ratio}")

					# Clean up GPU memory after generation
					del generated_ids_ap, generated_ids_bl
					del scores_ap, scores_bl
					del full_text_ap, full_text_bl
					del output_ap, output_bl
					if torch.cuda.is_available():
						torch.cuda.empty_cache()

				elif args.use_saved:
					print("Not implemented.")

	# Store all responses for this num_permuted
	# response_dict_all[num_permuted] = response_dict
	
	# Save incrementally after each num_permuted iteration
	# temp_output_filename = f'./prob_results/probs_{args.promptstyle}_{problem_dir}_{args.model.replace("/", "_")}_responses.npz'
	# np.savez_compressed(temp_output_filename, response_dict_all=response_dict_all)
	# print(f"Checkpoint saved after {num_permuted} permuted letters: {temp_output_filename}", flush=True)

	# Store results for this num_permuted
	for gen in gen_types:
		average_exemplar_probs[gen][num_permuted] = exemplar_probs_list[gen]
		average_final_answer_probs[gen][num_permuted] = final_answer_probs[gen]
		average_ratios[gen][num_permuted] = ratios_list[gen]
		
		print(f"Completed {gen} probabilities for {num_permuted} permuted letters.", flush=True)

# Print final results
for gen in gen_types:
	print(f"\n{'='*60}", flush=True)
	print(f"Results for {gen}:", flush=True)
	print(f"{'='*60}", flush=True)
	
	for num_permuted in [1, 2, 5, 10, 20]:
		if num_permuted in average_final_answer_probs[gen]:
			print(f"\nNum permuted letters: {num_permuted}", flush=True)

			probs_by_method = average_exemplar_probs[gen].get(num_permuted, {})
				
			print(f"  Exemplar probabilities:", flush=True)
			
			for method, probs_dict in probs_by_method.items():
				print(f"    {method}:", flush=True)
				for key in ['correct', 'incorrect', 'total']:
					probs_list = probs_dict.get(key, [])
					if probs_list:
						avg_prob = np.mean(probs_list)
						std_prob = np.std(probs_list)
						print(f"      {key}: {avg_prob:.6f} +- {std_prob:.6f} (n={len(probs_list)})", flush=True)
		
		final_probs_by_method = average_final_answer_probs[gen].get(num_permuted, {})
		print(f"  Final answer probabilities:", flush=True)
		
		for method, final_probs_dict in final_probs_by_method.items():
			print(f"    {method}:", flush=True)
			for key in ['correct', 'incorrect', 'total']:
				final_probs_list = final_probs_dict.get(key, [])
				if final_probs_list:
					avg_final_prob = np.mean(final_probs_list)
					std_final_prob = np.std(final_probs_list)
					print(f"      {key}: {avg_final_prob:.6f} +- {std_final_prob:.6f} (n={len(final_probs_list)})", flush=True)
		
		ratios_by_method = average_ratios[gen].get(num_permuted, {})
		print(f"  Correct/given probability ratios:", flush=True)
		
		for method, ratios_dict in ratios_by_method.items():
			print(f"    {method}:", flush=True)
			for key in ['same length', 'stopped early', 'stopped late']:
				ratio_list = ratios_dict.get(key, [])
				if ratio_list:
					avg_ratio = np.mean(ratio_list)
					std_ratio = np.std(ratio_list)
					print(f"      {key}: {avg_ratio:.6f} +- {std_ratio:.6f} (n={len(ratio_list)})", flush=True)


end = time.time()
print(f"\nTotal time: {end-start} seconds.", flush=True)

# Save all responses to npz file
# output_filename = f'./prob_results/probs_{args.promptstyle}_{problem_dir}_{args.model.replace("/", "_")}_responses.npz'
# np.savez_compressed(output_filename, response_dict_all=response_dict_all)
# print(f"All responses saved to {output_filename}", flush=True)
