import numpy as np
import argparse
import pandas as pd
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', help='analogical, hw, etc')
parser.add_argument('--problem', help='Give a problem: succ, pred')
parser.add_argument('--num_permuted', help="give a number of letters in the alphabet to permute from 2 to 26")
parser.add_argument('--model', help='give model: gpt3, gpt35, gpt4, Qwen_Qwen3-8B')
parser.add_argument('--gen', help='gen or nogen')
args = parser.parse_args()

# def compute_accuracy(trues, predictions):
#     correct = 0

#     for t, p in zip(trues, predictions):
#         p=p.strip(" '")
#         if (t==p):
#             correct+=1
#     return correct/len(trues)

acc_dict = {}

response_folder = f"{args.model}_prob_predictions_multi_alph/{args.gen}"
response_file = f"{args.model}_letterstring_results_{args.num_permuted}_multi_alph_gptprobs{"_" + args.prompt if args.prompt else ''}.npz"
print(f"Loading responses from {response_folder}/{response_file}...")
responses = np.load(f"{response_folder}/{response_file}", allow_pickle=True)["data"].item()

for alph in responses:

    print(f"Processing alphabet: {alph}")
    all_prob_type_responses = responses[alph]['responses']
    all_trues = responses[alph]['targets']
    shuffled_letters = responses[alph]['shuffled_letters']
    shuffled_alphabet = responses[alph]['shuffled_alphabet']

    accuracies = {}

    for prob_type in all_prob_type_responses.keys():
        prob_type_responses = all_prob_type_responses[prob_type]
        prob_type_trues = all_trues[prob_type]

        total = 0 
        correct = 0 

        for pred, true in zip(prob_type_responses, prob_type_trues):
            pred = pred.strip(" '").replace(" ", "").lower()
            true = ''.join(true).lower()
            print(f'Pred: {pred}, True: {true}')
            if pred == true:
                correct += 1
            total += 1

        if total > 0:
            accuracy = correct / total
            print(f"Accuracy for problem type {prob_type}: {accuracy}")
            accuracies[prob_type] = accuracy
        else:
            print(f"No predictions for problem type {prob_type}")

    acc_dict[alph] = accuracies

print(f"\n=== Summary of Accuracies ===\n")
for alph, alph_acc in acc_dict.items():
    print(f"--- Alphabet: {alph} ---")
    for prob_type, acc in alph_acc.items():
        print(f"Problem Type: {prob_type}, Accuracy: {acc}")
    print("\n")

    # Print average accuracy across all alphabets
overall_accuracies = {}
for alph, alph_acc in acc_dict.items():
    for prob_type, acc in alph_acc.items():
        if prob_type not in overall_accuracies:
            overall_accuracies[prob_type] = []
        overall_accuracies[prob_type].append(acc)
print(f"\n=== Overall Average Accuracies ===\n")
for prob_type, accs in overall_accuracies.items():
    average_acc = sum(accs) / len(accs)
    print(f"Problem Type: {prob_type}, Average Accuracy: {average_acc}")
# Save accuracies to a CSV file
df_rows = []
for alph, alph_acc in acc_dict.items():
    for prob_type, acc in alph_acc.items():
        df_rows.append({'Alphabet': alph, 'Problem_Type': prob_type, 'Accuracy': acc})
df = pd.DataFrame(df_rows)
output_csv = f"results/{args.model}_letterstring_accuracies_{args.num_permuted}_{args.prompt}.csv"
df.to_csv(output_csv, index=False)