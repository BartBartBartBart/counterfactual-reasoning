# /home/bart/uva/thesis/code/robust-analogy/letterstring/Qwen_Qwen3-8B_prob_predictions_multi_alph/nogen/Qwen_Qwen3-8B_letterstring_results_1_multi_alph_gptprobs.npz
# open pickle 

import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--problem', type=str, default='succ', help='which problem to evaluate')
args = parser.parse_args()

data = np.load(f'/home/bart/uva/thesis/code/robust-analogy/letterstring/Qwen_Qwen3-8B_prob_predictions_multi_alph/nogen/Qwen_Qwen3-8B_letterstring_results_1_multi_alph_gptprobs.npz', allow_pickle=True)

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

# cr_trues = []
# cr_preds = []
# cr_shuffled = []
# num_permuted = 1
# model = "Qwen/Qwen3-8B"
# gpt = None

# all_responses = np.load(f'/home/bart/uva/thesis/code/robust-analogy/letterstring/Qwen_Qwen3-8B_prob_predictions_multi_alph/nogen/Qwen_Qwen3-8B_letterstring_results_1_multi_alph_gptprobs.npz', allow_pickle=True)
# all_responses = all_responses.files
# for aid, alph in enumerate(all_responses):
#     shuffled_letters = all_responses[aid]['shuffled_letters']
#     if gpt == 3 and num_permuted not in [1, 'symb']:
#         # error in saving data
#         shuffled_alphabet = all_responses[alph]['shuffled_alphabet']
#         control_responses = all_responses[alph]['control_responses'][:25]
#     else:
#         shuffled_alphabet = all_responses[aid]['shuffled_alphabet']
#         control_responses = all_responses[aid]['control_responses']

#     if args.problem == 'succ':
#         current_cr_trues = shuffled_alphabet[1:]
#         for i, t in enumerate(current_cr_trues):
#             if shuffled_letters and (t in shuffled_letters or shuffled_alphabet[i] in shuffled_letters):
#                 cr_shuffled.append(1)
#             else:
#                 cr_shuffled.append(0)

#         cr_trues += current_cr_trues
        
#     elif args.problem == 'pred':
#         current_cr_trues = shuffled_alphabet[:-1]
#         for i, t in enumerate(current_cr_trues):
#             if shuffled_letters and (t in shuffled_letters or shuffled_alphabet[i] in shuffled_letters):
#                 cr_shuffled.append(1)
#             else:
#                 cr_shuffled.append(0)
#         cr_trues += current_cr_trues
#     cr_preds += control_responses

# shuf_acc, unshuf_acc, shuf_tot, unshuf_tot =compute_accuracy(cr_trues, cr_preds, cr_shuffled)
# data.append(compute_accuracy(cr_trues, cr_preds, cr_shuffled))

# acc_dict[f'gpt_{gpt}_{num_permuted}'] = compute_accuracy(cr_trues, cr_preds, cr_shuffled)

