import numpy as np
import argparse
import pandas as pd
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--sentence', action='store_true', help="Present problem in sentence format.")
parser.add_argument('--noprompt', action='store_true', help="Present problem without prompt.")
parser.add_argument('--newprompt', action='store_true', help="Present problem with new prompt.")
parser.add_argument('--problem', help='Give a problem: succ, pred')
parser.add_argument('--num_permuted', help="give a number of letters in the alphabet to permute from 2 to 26")
parser.add_argument('--gpt', help='give gpt model: 3, 35, 4')

args = parser.parse_args()
data = []
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


def analyze_errors(trues, predictions, problem, alphabet, num_permuted):
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', \
                'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    count_standard =0
    count_other = 0
    for t, p in zip(trues, predictions):
        p=p.strip(" '")
        if t == p:
            continue
        if num_permuted == 'symb':
            count_other += 1
            continue
        if problem == 'succ':
            query = alphabet[alphabet.index(t)-1]
            # if letters.index(query) > 24:
            
            if letters.index(query) < 25 and p == letters[letters.index(query) + 1]:
                count_standard += 1
            else:
                print(t, p)
                print(alphabet)
                count_other += 1
        elif problem == 'pred':
            query = alphabet[alphabet.index(t)+1]
            # print(f'query is: {query}')
            # print(t, p)
            if p == letters[letters.index(query) - 1]:
                count_standard += 1
            else:
                print(t, p)
                print(alphabet)
                count_other += 1
            # print(f'running errors are: {count_standard, count_other}')

    return count_standard, count_other
                



table = []
error_dict = {}
acc_dict = {}
if args.problem == 'pred':
    gpts = [35, 4]
elif args.problem == 'succ':
    gpts = [3, 35, 4]
for gpt in gpts:
    for num_permuted in [2, 5, 10, 20, 'symb']:        
        print(num_permuted)
        all_responses = np.load(f'controls/gpt{gpt}_letterstring_control_{num_permuted}_{args.problem}.npz', allow_pickle=True)

        all_responses = all_responses['all_responses'].item()

        cr_trues = []
        cr_preds = []
        cr_shuffled = []

        for alph in all_responses.keys():
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

        acc_dict[f'gpt_{gpt}_{num_permuted}'] = compute_accuracy(cr_trues, cr_preds, cr_shuffled)

 
acc_df = pd.DataFrame.from_dict(acc_dict, orient='index', columns =['shuf_acc', 'unshuf_acc','shuf_tot', 'unshuf_tot'])
print(acc_df)
