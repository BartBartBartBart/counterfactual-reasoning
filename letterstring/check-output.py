# /home/bart/uva/thesis/code/robust-analogy/letterstring/Qwen_Qwen3-8B_prob_predictions_multi_alph/nogen/Qwen_Qwen3-8B_letterstring_results_1_multi_alph_gptprobs.npz
# open pickle 

import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--problem', type=str, default='succ', help='which problem to evaluate')
args = parser.parse_args()

def compute_accuracy(trues, predictions, shuffled):
    shuffled_correct = 0
    unshuffled_correct = 0

    for t, p, s in zip(trues, predictions,shuffled):
        p=p.strip(" '")
        if (t==p):
            if s:
                shuffled_correct+=1
            else:
                unshuffled_correct+=1
    shuffled_tot = sum(shuffled)
    unshuffled_tot = len(trues)-sum(shuffled)
    if shuffled_tot > 0:
        return shuffled_correct/shuffled_tot, unshuffled_correct/unshuffled_tot, shuffled_tot, unshuffled_tot
    else:
        return 0, unshuffled_correct/unshuffled_tot, shuffled_tot, unshuffled_tot

data = np.load(f'/home/bart/uva/thesis/code/counterfactual-reasoning/letterstring/Qwen_Qwen3-8B_prob_predictions_multi_alph/nogen/Qwen_Qwen3-8B_letterstring_results_1_multi_alph_gptprobs.npz', allow_pickle=True)
# data = np.load(f'/home/bart/uva/thesis/code/counterfactual-reasoning/letterstring/GPT35_prob_predictions_multi_alph/nogen/gpt35_letterstring_results_1_multi_alph_hw.npz', allow_pickle=True)
# data = np.load(f'/home/bart/uva/thesis/code/counterfactual-reasoning/letterstring/controls/gpt3_letterstring_control_1_succ.npz', allow_pickle=True)
all_prob_type_responses = data.files
for alph in all_prob_type_responses:
    print(alph)
    for subitem in data[alph].item().keys():
        print(f'  {subitem}: {type(data[alph].item()[subitem])}, length: {len(data[alph].item()[subitem])}')
        for entry in data[alph].item()[subitem][:2]:
            print(f'    {entry}')
            print(f'    Type: {type(entry)}, Length: {len(entry)}')
            for l in entry:
                print(f'      {l}')
                print(type(l), len(l))
    print("\n")

cr_trues = []
cr_preds = []
cr_shuffled = []
num_permuted = 1
data = []
# model = "Qwen/Qwen3-8B"
gpt = 3

all_responses = np.load(f'/home/bart/uva/thesis/code/counterfactual-reasoning/letterstring/Qwen_Qwen3-8B_prob_predictions_multi_alph/nogen/Qwen_Qwen3-8B_letterstring_results_1_multi_alph_gptprobs.npz', allow_pickle=True)
# all_responses = np.load(f'/home/bart/uva/thesis/code/counterfactual-reasoning/letterstring/GPT3_prob_predictions_multi_alph/nogen/gpt3_letterstring_results_1_multi_alph_hw.npz', allow_pickle=True)
# all_responses = np.load(f'/home/bart/uva/thesis/code/counterfactual-reasoning/letterstring/controls/gpt3_letterstring_control_1_succ.npz', allow_pickle=True)
all_responses = all_responses.files
for alph in all_responses:
    shuffled_letters = all_responses[alph]['shuffled_letters']
    if gpt == 3 and num_permuted not in [1, 'symb']:
        # error in saving data
        shuffled_alphabet = all_responses[alph]['shuffled_alphabet']
        control_responses = all_responses[alph]['control_responses'][:25]
    else:
        shuffled_alphabet = all_responses[alph]['shuffled_alphabet']
        control_responses = all_responses[alph]['control_responses']

    if args.problem == 'succ':
        current_cr_trues = shuffled_alphabet[1:]
        for i, t in enumerate(current_cr_trues):
            if shuffled_letters and (t in shuffled_letters or shuffled_alphabet[i] in shuffled_letters):
                cr_shuffled.append(1)
            else:
                cr_shuffled.append(0)

        cr_trues += current_cr_trues
        
    elif args.problem == 'pred':
        current_cr_trues = shuffled_alphabet[:-1]
        for i, t in enumerate(current_cr_trues):
            if shuffled_letters and (t in shuffled_letters or shuffled_alphabet[i] in shuffled_letters):
                cr_shuffled.append(1)
            else:
                cr_shuffled.append(0)
        cr_trues += current_cr_trues
    cr_preds += control_responses

shuf_acc, unshuf_acc, shuf_tot, unshuf_tot =compute_accuracy(cr_trues, cr_preds, cr_shuffled)
data.append(compute_accuracy(cr_trues, cr_preds, cr_shuffled))

# acc_dict[f'gpt_{gpt}_{num_permuted}'] = compute_accuracy(cr_trues, cr_preds, cr_shuffled)

