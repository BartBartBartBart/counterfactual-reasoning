import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--prob_format', type=str, default='digits')
parser.add_argument('--model', type=str, default='Qwen_Qwen3-8B')
parser.add_argument('--sys_prompt', type=str, default='default')
parser.add_argument('--user_prompt', type=str, default='default')
parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
args = parser.parse_args()

if args.model.startswith('Qwen'):
    user_prompt = "analogical" if args.user_prompt == "4" else "minimal"
    path = f'./results/{args.model}'
    fname = f'{path}/matprob_results_{args.prob_format}_prompt_{args.sys_prompt}_{user_prompt}.npz'
else:
    path = f'./results'
    fname = f'{path}/gpt_matprob_results_{args.prob_format}_{args.model}_prompt_{args.sys_prompt}_{args.user_prompt}.npz'

data = np.load(fname, allow_pickle=True)
all_gen_correct_pred = data['all_gen_correct_pred'].item()

acc_dict = {} # keys are problem types, values are accuracies
for prob_type in all_gen_correct_pred.keys():
    correct_preds = all_gen_correct_pred[prob_type]
    accuracy = sum(correct_preds) / len(correct_preds) if len(correct_preds) > 0 else 0.0
    acc_dict[prob_type] = accuracy
print("Accuracies per problem type:")
for prob_type, accuracy in acc_dict.items():
    print(f"{prob_type}: {accuracy:.4f}")

# avg accuracy over all problem types
total_correct = 0
total_count = 0
for prob_type in all_gen_correct_pred.keys():
    correct_preds = all_gen_correct_pred[prob_type]
    total_correct += sum(correct_preds)
    total_count += len(correct_preds)
overall_accuracy = total_correct / total_count if total_count > 0 else 0.0
print("------------------------")
print(f"Overall accuracy: {overall_accuracy:.4f}")