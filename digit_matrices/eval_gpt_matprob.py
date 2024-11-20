
import numpy as np
import builtins
import os
from openai import AzureOpenAI
import sys
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--id", help="The gpt version, can be 0613, 1106 or 0125 or 350613")
parser.add_argument("--user_prompt_num", help="user prompt", type=int)
parser.add_argument("--sys_prompt_num", help="sys prompt", type=int)
parser.add_argument("--prob_format", help="use webb data")
parser.add_argument("--letters", help="use letters", action="store_true")
args = parser.parse_args()

if args.prob_format not in ['digits', 'symb', 'coords']:
	print('prob_format must be one of digits, symb, or coords')

id = args.id



if args.prob_format=='digits':
	prob_format = '_digits'
	checkset = list(range(10))
elif args.prob_format=='coords':
	prob_format = '_coords'
	checkset = list(range(10))
elif args.prob_format=='symb':
	prob_format = '_symb'
	checkset =set(['%', '&', '*', '!', '(', '<', '~', '>', ':', '$']) #changed ? to (
else:
	print('prob_format must be one of digits, symb, or coords')
	sys.exit()
						

if (args.sys_prompt_num == 0):
	sys_content = 'You are a helpful assistant.'
elif args.sys_prompt_num == 1:
	sys_content = 'You are a genius at solving analogy problems.'

versions = {'0125':{'resource_name':'0125-Preview', 'deployment_name':'0125-Preview'},
            '1106':{'resource_name':'MMResearch', 'deployment_name':'gpt-4-1106-Preview'},
            '0613':{'resource_name':'0613', 'deployment_name':'0613'},
			'350613':{'resource_name':'0613', 'deployment_name':'0613'}}

if id not in versions.keys():
	print(f'id should be 0125, 1106, or 0613 or 350613')

client = AzureOpenAI(
  azure_endpoint = os.getenv(f"AZURE_OPENAI_ENDPOINT_{id}"), 
  api_key=os.getenv(f"AZURE_OPENAI_API_KEY_{id}"),  
  api_version="2024-02-01"
)

# Split word into characters
def split(word):
    return [char for char in word]

# Load all problems
if args.prob_format == 'digits':
	all_prob = np.load('./problems/all_problems.npz', allow_pickle=True)
elif args.prob_format == 'coords':
	all_prob = np.load('./problems/all_problems_coords.npz', allow_pickle=True)
else:
	all_prob = np.load('./problems/all_problems_symb.npz', allow_pickle=True)

# GPT settings
kwargs = {"temperature":0, "max_tokens":10, "stop":"\n", }


# Loop through all problem types
all_prob_types = builtins.list(all_prob['all_problems'].item().keys())
# Load data if it already exists
all_data_fname = f'./results/gpt_matprob_results_{args.prob_format}_{id}_prompt_{args.sys_prompt_num}_{args.user_prompt_num}{prob_format}.npz'
if os.path.exists(all_data_fname):
	data_exists = True
	all_data = np.load(all_data_fname, allow_pickle=True)
else:
	data_exists = False
# Create data structure for storing results
all_gen_pred = {}
all_gen_correct_pred = {}
all_MC_pred = {}
all_MC_correct_pred = {}
all_alt_MC_correct_pred = {}
for p in range(len(all_prob_types)):
	# Problem type
	prob_type = all_prob_types[p]
	# Load data
	if data_exists:
		all_gen_pred[prob_type] = all_data['all_gen_pred'].item()[prob_type]
		all_gen_correct_pred[prob_type] = all_data['all_gen_correct_pred'].item()[prob_type]
		all_MC_pred[prob_type] = all_data['all_MC_pred'].item()[prob_type]
		all_MC_correct_pred[prob_type] = all_data['all_MC_correct_pred'].item()[prob_type]
		all_alt_MC_correct_pred[prob_type] = all_data['all_alt_MC_correct_pred'].item()[prob_type]
	# Create data structure
	else:
		all_gen_pred[prob_type] = []
		all_gen_correct_pred[prob_type] = []
		all_MC_pred[prob_type] = []
		all_MC_correct_pred[prob_type] = []
		all_alt_MC_correct_pred[prob_type] = []
# Loop over all problem indices
N_prob = 50
none_count = 0
for prob_ind in range(N_prob):
	print(str(prob_ind + 1) + ' of ' + str(N_prob) + '...')
	# Loop over all problem types
	for p in range(len(all_prob_types)):
		# Problem type
		prob_type = all_prob_types[p]
		print('Problem type: ' + prob_type + '...')
		perm_invariant = all_prob['all_problems'].item()[prob_type]['perm_invariant']
		prob_type_N_prob = len(all_prob['all_problems'].item()[prob_type]['prob']) #changed this from shape[0]
		if prob_ind < prob_type_N_prob and len(all_gen_correct_pred[prob_type]) <= prob_ind:

			# Problem
			prob = all_prob['all_problems'].item()[prob_type]['prob'][prob_ind]
			size_prob =len(prob)
			answer_choices = all_prob['all_problems'].item()[prob_type]['answer_choices'][prob_ind]
			correct_ind = all_prob['all_problems'].item()[prob_type]['correct_ind'][prob_ind]
			correct_answer = answer_choices[correct_ind]

			if args.prob_format == 'coords':
				cx, cy = all_prob['all_problems'].item()[prob_type]['correct_coords'][prob_ind]

			# Generate prompt
			if args.user_prompt_num == 0:
				prompt = ''
			elif args.user_prompt_num ==1:
				prompt = "Let's try to complete the pattern:\n" #This prompt does not work Prompt 1
			elif args.user_prompt_num == 2:
				prompt = "Try to complete the pattern below. Give ONLY the answer as briefly as possible.\n"
			elif args.user_prompt_num == 3:
				prompt = "Try to fill the gap in the pattern below. Give ONLY the answer as briefly as possible.\n"
			else:
				print('You must choose prompt 1, 2, or 3')
				sys.exit()
			for r in range(size_prob):
				for c in range(size_prob):
					prompt += '['
					if args.prob_format != 'coords':
						if not (r == size_prob-1 and c == size_prob-1):
							for i in range(len(prob[r][c])):
								if prob[r][c][i] == -1:
									prompt += ' '
								else:
									prompt += str(prob[r][c][i])
								if i < len(prob[r][c]) - 1:
									prompt += ' '
							prompt += ']'
							if c < size_prob-1:
								prompt += ' '
							else:
								prompt += '\n'
					else:
						if (r == cx and c == cy):
							for i in range(len(prob[r][c])):
								prompt+= ' '
								if i < (len(prob[r][c]) - 1):
									prompt += ' '
						else:
							for i in range(len(prob[r][c])):
								if prob[r][c][i] == -1:
									prompt += ' '
								else:
									prompt += str(prob[r][c][i])
								if i < len(prob[r][c]) - 1:
									prompt += ' '
						prompt += ']'
						if c < size_prob-1:
							prompt += ' '
						else:
							prompt += '\n'
			print(prompt)
			# sys.exit()
			# Get response
			response = client.chat.completions.create(
				model= versions[id]['deployment_name'],
				messages=[
					{"role": "system", "content": sys_content},
					{"role": "user", "content": prompt},
					],
				**kwargs
			)

			response_text = response.choices[0].message.content
			if response_text is None:
				response_text = 'None'
				none_count+=1
				print(f'Nonecount is {none_count}')
			# print(response_text)
			# sys.exit()
			# Find portion of response corresponding to prediction
			prediction = response_text.lstrip('[')

			all_gen_pred[prob_type].append(prediction)
			# Get prediction set
			pred_set = []
			invalid_char = False
			closing_bracket = False
			for p in split(prediction):
				if p != ' ':
					if args.prob_format != 'symb' and p.isdigit():
							p = int(p)
					if p in checkset:
						pred_set.append(p)
					elif p == ']':
						break
					else:
						invalid_char = True
						break
			# Sort answer if problem type is permutation invariant
			if perm_invariant:
				correct_answer = np.sort(correct_answer)
				pred_set = np.sort(pred_set)
			# Determine whether prediction is correct
			correct_pred = False
			if not invalid_char and len(pred_set) == len(correct_answer):
				if np.all(pred_set == correct_answer):
					correct_pred = True
			all_gen_correct_pred[prob_type].append(correct_pred)

			# Save data
			eval_fname = f'./results/gpt_matprob_results_{args.prob_format}_{id}_prompt_{args.sys_prompt_num}_{args.user_prompt_num}{prob_format}.npz'
			np.savez(eval_fname, 
				all_gen_pred=all_gen_pred, all_gen_correct_pred=all_gen_correct_pred, all_MC_pred=all_MC_pred, all_MC_correct_pred=all_MC_correct_pred, all_alt_MC_correct_pred=all_alt_MC_correct_pred, 
				allow_pickle=True)
			
print(f'Nonecount is {none_count}')
