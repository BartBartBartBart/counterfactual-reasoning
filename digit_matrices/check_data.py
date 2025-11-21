import numpy as np 
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--prob_format', type=str, default='logit')
parser.add_argument('--gpt', type=str, default='gpt-4')
parser.add_argument('--sys_prompt', type=str, default='default')
parser.add_argument('--user_prompt', type=str, default='default')
args = parser.parse_args()

gpt_data = np.load(f'./results/gpt_matprob_results_{args.prob_format}_{args.gpt}_prompt_{args.sys_prompt}_{args.user_prompt}.npz', allow_pickle=True)

problems = np.load('./problems/all_problems_symbols.npz', allow_pickle=True)
symb_problems = problems['all_problems'].item()

problems = np.load('./problems/all_problems.npz', allow_pickle=True)
num_problems = problems['all_problems'].item()

problems = np.load('./problems/all_problems_coords.npz', allow_pickle=True)
coord_problems = problems['all_problems'].item()

if args.prob_format=='digits':
    problems = num_problems
elif args.prob_format=='symb':
    problems = symb_problems
elif args.prob_format == 'coords':
    problems = coord_problems
else: 
    print(f'prob_format is {args.prob_format}')

# - all_gen_pred
# - all_gen_correct
# - all_MC_pred
# - all_MC_correct_pred
# - all_alt_MC_correct_pred

predictions = gpt_data['all_gen_pred'].item()
correct = gpt_data['all_gen_correct_pred'].item()
# all_MC_pred = gpt_data['all_MC_pred'].item()
# all_MC_correct_pred = gpt_data['all_MC_correct_pred'].item()
# all_alt_MC_correct_pred = gpt_data['all_alt_MC_correct_pred'].item() 

for prob_type in predictions.keys():
    print(f'Prob type: {prob_type}')
    prob_ind = 0
    assert(len(predictions[prob_type])==len(correct[prob_type]))
    answers = [choices[ind] for ind, choices in zip(problems[prob_type]['correct_ind'], problems[prob_type]['answer_choices'])]
    for prob, answer_choices, correct_ind, correct_coords, pred, corr, answer in zip(problems[prob_type]['prob'], problems[prob_type]['answer_choices'], problems[prob_type]['correct_ind'], problems[prob_type]['correct_coords'], predictions[prob_type], correct[prob_type], answers[:len(predictions[prob_type])]):
        print(f'Problem {prob}')
        print('Answer choices:', answer_choices)
        print('Correct index:', correct_ind)
        print('Correct coords:', correct_coords)
        print('Prediction:', pred)
        print('Correct:', corr)
        print('Answer:', answer)    
        print('---')
        if args.prob_format=='digits':
            symb_num = 'num'
        elif args.prob_format=='symb':
            symb_num = 'symb'
        elif args.prob_format == 'letters':
            symb_num='lett'
        elif args.prob_format=='coords':
            symb_num = 'coords'
        else:
            print(f'prob_format is {args.prob_format}')
        if corr:
            corr_val = 1
        else:
            corr_val =0