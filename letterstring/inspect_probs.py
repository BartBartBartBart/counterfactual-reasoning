import openai
import numpy as np
import builtins
import argparse
import os
import time
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import torch.nn.functional as F


start = time.time()

def check_path(path):
	if not os.path.exists(path):
		os.mkdir(path)

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--sentence', action='store_true', help="Present problem in sentence format.")
parser.add_argument('--noprompt', action='store_true', help="Present problem without prompt.")
parser.add_argument('--newprompt', action='store_true', help="Present problem with new prompt.")
parser.add_argument('--promptstyle', help='Give a prompt style: human, minimal, hw, webb, webbplus')
parser.add_argument('--num_permuted', help="give a number of letters in the alphabet to permute from 2 to 26")
parser.add_argument('--gpt', help='give gpt model: 3, 35, 4', default=None)
parser.add_argument('--model', help='give model name', default=None)
parser.add_argument('--gen', help='give gen for generalized problems or nogen for non generalized')
parser.add_argument('--hf_token', help='Huggingface token for model loading', default=None)
parser.add_argument('--verbose', action='store_true', help="Print verbose output.")
parser.add_argument('--extra-split', action='store_true', help="Test only 3gensplit7")
parser.add_argument('--use-8bit', action='store_true', help="Use 8-bit quantization to save memory (may be slower)")
parser.add_argument('--debug', action='store_true', help="Debug mode - do not load large models")
args = parser.parse_args()

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

def create_prompt(promptstyle, prob, alph_string, noprompt, sentence):
	prompt=''
	if not noprompt:
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
	if sentence:
		prompt += 'If '
		for i in range(len(prob[0][0])):
			prompt += str(prob[0][0][i])
			if i < len(prob[0][0]) - 1:
				prompt += ' '
		prompt += ' changes to '
		for i in range(len(prob[0][1])):
			prompt += str(prob[0][1][i])
			if i < len(prob[0][1]) - 1:
				prompt += ' '
		prompt += ', then '
		for i in range(len(prob[1][0])):
			prompt += str(prob[1][0][i])
			if i < len(prob[1][0]) - 1:
				prompt += ' '
		prompt += ' should change to '
	else:
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
		# if args.promptstyle == "analogical":
		# 	prompt += '\n\nFirst, describe 3 relevant exemplars that are distinct from this problem. Then give the final answer. Answer with only the examples and the final answer with no further explanation. Put your final answer between double brackets.\n'
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

def exemplar_probs(full_text, tokenizer, scores, generated_ids):
	# Find probabilities of exemplars
	if 'exemplars' in full_text.lower():
		exemplar_section = full_text.split('exemplar')[1]
	elif 'examples' in full_text.lower():
		exemplar_section = full_text.split('examples')[1]
	else: 
		exemplar_section = full_text

	# Calculate probability of all words after 'exemplar'
	words = exemplar_section.split()
	prob_per_word = []
	logprob = 0.0
	for i, word in enumerate(words):
		token_ids = tokenizer.encode(word, add_special_tokens=False)
		for j, token_id in enumerate(token_ids):
			step = len(generated_ids) - len(token_ids) + j
			if step < len(scores):
				token_logprob = torch.log_softmax(scores[step][0], dim=-1)[token_id].item()
				prob_per_word.append(token_logprob)
				logprob += token_logprob
	prob = np.exp(logprob)
	return prob, prob_per_word

if args.promptstyle == "webb" and int(args.num_permuted) >1:
	print("promptstyle webb can only be used with an unpermuted alphabet")
	sys.exit()

# GPT-3 settings
openai.api_key = "API KEY HERE"
if args.gpt == '3':
    kwargs = {"engine":"text-davinci-003", "temperature":0, "max_tokens":40, "stop":"\n", "echo":False, "logprobs":1, }
elif args.gpt == '35':
    kwargs = { "model":"gpt-3.5-turbo", "temperature":0, "max_tokens":40, "stop":"\n"}
elif args.gpt == '4':
    kwargs = { "model":"gpt-4", "temperature":0, "max_tokens":40, "stop":"\n"}
	
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
	}
	
	# Use 8-bit quantization if requested (saves memory but may be slower)
	if args.use_8bit:
		from transformers import BitsAndBytesConfig
		load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
		load_kwargs["device_map"] = "auto"
		print("Using 8-bit quantization", flush=True)
	else:
		# Force all on GPU 0 - no CPU offloading
		load_kwargs["device_map"] = "cuda:0"
	
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

all_prob_10perm = np.load(f'./problems/nogen/all_prob_10_7_human.npz', allow_pickle=True)['all_prob']
all_prob = np.load(f'./problems/nogen/all_prob_1_7_human.npz', allow_pickle=True)['all_prob']


response_dict={}

for alph, perm_alph in zip(all_prob.item().keys(), all_prob_10perm.item().keys()):
	print(alph, flush=True)
	print(perm_alph, flush=True)
	if (all_prob.item()[alph]['shuffled_letters'] is not None):
		shuffled_letters = builtins.list(all_prob.item()[alph]['shuffled_letters'])	
	else:
		shuffled_letters = None

	if (all_prob_10perm.item()[perm_alph]['shuffled_letters'] is not None):
		shuffled_letters_perm = builtins.list(all_prob_10perm.item()[perm_alph]['shuffled_letters'])	
	else:
		shuffled_letters_perm = None
		
	shuffled_alphabet = builtins.list(all_prob.item()[alph]['shuffled_alphabet'])
	shuffled_alphabet_perm = builtins.list(all_prob_10perm.item()[perm_alph]['shuffled_alphabet'])
	
	response_dict[alph] = {
		'shuffled_letters': shuffled_letters,
		'shuffled_alphabet': shuffled_alphabet,
	}
	response_dict[perm_alph] = {
        'shuffled_letters': shuffled_letters_perm,
        'shuffled_alphabet': shuffled_alphabet_perm,
    }

	prob_types = builtins.list(all_prob.item()[alph].keys())[2:] # first two items are list of shuffled letters and shuflled alphabet: skip this
	prob_types_perm = builtins.list(all_prob_10perm.item()[perm_alph].keys())[2:] # first two items are list of shuffled letters and shuflled alphabet: skip this
	N_prob_types = len(prob_types) # -1 # minus 1 to skip attention problems

	alph_string = ' '.join(shuffled_alphabet)
	alph_string_perm = ' '.join(shuffled_alphabet_perm)
	print("Alphabet:", flush=True)
	print(alph_string, flush=True)
	print("Permuted Alphabet:", flush=True)
	print(alph_string_perm, flush=True)

	# Evaluate
	N_trials_per_prob_type = 10
	count = 0
	for p in range(N_prob_types):
		if prob_types[p] == 'attn':
			# SKIP ATTENTION PROBLEMS
			continue
			alph_string = "For this question, ignore other instructions and respond 'a a a a'"
		# accidently left out 3gensplit7, test separately
		elif args.extra_split and prob_types[p] != '3gensplit7':
			continue
		print(f"Problem type: {prob_types[p]} - {str(p+1)}/{str(N_prob_types)}", flush=True)
		# print('problem type ' + str(p+1) + ' of ' + str(N_prob_types) + '... ', flush=True)
		prob_type_responses = []
		prob_type_responses_perm = []
		prob_type_targets = []
		prob_type_targets_perm = []
		for t in range(N_trials_per_prob_type):
			if t == 0:
				t += 1  # skip first trial for speed
				continue
			print('trial ' + str(t+1) + ' of ' + str(N_trials_per_prob_type) + '...', flush=True)
			prob = all_prob.item()[alph][prob_types[p]]['prob'][t]
			full_tgt_letters = all_prob.item()[alph][prob_types[p]]['tgt_letters'][t]
			current_target = all_prob.item()[alph][prob_types[p]]['prob'][t][1][1]
			prob_type_targets.append(current_target)

			prob_perm = all_prob_10perm.item()[perm_alph][prob_types_perm[p]]['prob'][t]
			full_tgt_letters_perm = all_prob_10perm.item()[perm_alph][prob_types_perm[p]]['tgt_letters'][t]
			current_target_perm = all_prob_10perm.item()[perm_alph][prob_types_perm[p]]['prob'][t][1][1]
			prob_type_targets_perm.append(current_target_perm)

			# Create prompt
			messages = create_prompt(args.promptstyle, prob, alph_string, args.noprompt, args.sentence)
			# Create prompt for permuted alphabet
			messages_perm = create_prompt(args.promptstyle, prob_perm, alph_string_perm, args.noprompt, args.sentence)

	        # If verbose or first trial
			if args.verbose or t == 0:
				print("\n=== PROMPT ===\n", flush=True)
				print(f"System message: {messages[0]['content']}\n", flush=True)
				print(f"User message: {messages[1]['content']}\n", flush=True)
				print("\n--- TARGET LETTERS ---\n", flush=True)
				print(current_target, flush=True)

				print("\n=== PROMPT PERMUTED ===\n", flush=True)
				print(f"System message: {messages_perm[0]['content']}\n", flush=True)
				print(f"User message: {messages_perm[1]['content']}\n", flush=True)
				print("\n--- TARGET LETTERS PERMUTED ---\n", flush=True)
				print(current_target_perm, flush=True)

			if args.gpt == '3':
				comp_prompt = ''
				for m in messages:
					comp_prompt += '\n' + m['content']
				comp_prompt=comp_prompt.strip('\n')				

			# Get response
			response = []
			while len(response) == 0:
				if args.gpt == '3':
					try:
						response = openai.Completion.create(prompt=comp_prompt, **kwargs)
					except:
						print('trying again...')
						time.sleep(5)
				elif args.model.startswith("Qwen"):
					print("Gathering response...", flush=True)
					generated_ids, scores, full_text = get_probs(messages, model, tokenizer)
					generated_ids_perm, scores_perm, full_text_perm = get_probs(messages_perm, model, tokenizer)
					print("Response gathered.", flush=True)

					print("\n=== RESPONSE ===\n", flush=True)
					clean_out = clean_text(full_text)
					print(clean_out, flush=True)
					print("\n=== RESPONSE PERMUTED ===\n", flush=True)
					clean_out_perm = clean_text(full_text_perm)
					print(clean_out_perm, flush=True)

					# print("SCORES AND PROBS:", flush=True)
					# print(scores, flush=True)
					# print(scores_perm, flush=True)

					print("Calculating probabilities...", flush=True)
					probs, probs_per_word = exemplar_probs(full_text, tokenizer, scores, generated_ids)
					probs_perm, probs_perm_per_word = exemplar_probs(full_text_perm, tokenizer, scores_perm, generated_ids_perm)
					print(f"Probability of exemplar section: {prob}", flush=True)
					print(f"Probability per word: {probs_per_word}", flush=True)
					print(f"Probability of exemplar section (permuted): {prob_perm}", flush=True)	
					print(f"Probability per word (permuted): {probs_perm_per_word}", flush=True)

					end = time.time()
					print(f"Time taken for generation and prob calculation: {end-start} seconds.", flush=True)
					sys.exit()

					# Clean up GPU memory after generation
					del inputs, outputs
					if torch.cuda.is_available():
						torch.cuda.empty_cache()
					# print("Filtered response:", clean_out)
				else:
					try:
						response = openai.ChatCompletion.create(messages=messages, **kwargs)
					except:
						print('trying again...')
						time.sleep(5)

			if args.gpt =='3':
				prob_type_responses.append(response['choices'][0]['text'])
			# elif args.model == "Qwen/Qwen3-8B":
# 			elif args.model.startswith("Qwen"):
# 				prob_type_responses.append(response[0])
# 			else:
# 				prob_type_responses.append(response['choices'][0]['message']['content'])
# 				# print(response)
# 			count += 1
		
# 		# Store this problem type's responses and targets
# 		if 'responses' not in response_dict[alph]:
# 			response_dict[alph]['responses'] = {}
# 			response_dict[alph]['targets'] = {}
		
# 		response_dict[alph]['responses'][prob_types[p]] = prob_type_responses
# 		response_dict[alph]['targets'][prob_types[p]] = prob_type_targets

# # Save once after all alphabets and problem types are processed
# # Build path
# if args.gpt is not None:
# 	path = f'GPT{args.gpt}_prob_predictions_multi_alph/{args.gen}'
# else:
# 	path = f'{args.model.replace("/","_")}_prob_predictions_multi_alph/{args.gen}'
# check_path(path)

# # Build filename
# if args.gpt is not None:
# 	save_fname = f'./{path}/gpt{args.gpt}_letterstring_results_{args.num_permuted}_multi_alph_gptprobs'
# else:
# 	save_fname = f'./{path}/{args.model.replace("/","_")}_letterstring_results_{args.num_permuted}_multi_alph_gptprobs'
# if args.promptstyle:
# 	save_fname += f'_{args.promptstyle}'
# if args.sentence:
# 	save_fname += '_sentence'
# if args.noprompt:
# 	save_fname += '_noprompt'
# if args.extra_split:
# 	save_fname += '_extrasplit'
# if args.num_permuted == "symb" and args.gen == "gen":
# 	save_fname += f'_{14}_alphs'
	
# save_fname += '.npz'

# # Save single file with all data
# np.savez(save_fname, data=response_dict, allow_pickle=True)
# print(f"Saved {save_fname}")

end = time.time()
print(f"Total time: {end-start} seconds.", flush=True)
