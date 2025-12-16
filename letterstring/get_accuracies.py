import numpy as np
import argparse
import pandas as pd
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--promptstyle', help='analogical, hw, etc')
parser.add_argument('--problem', help='Give a problem: succ, pred')
parser.add_argument('--num_permuted', help="give a number of letters in the alphabet to permute from 2 to 26")
parser.add_argument('--model', help='give model: gpt3, gpt35, gpt4, Qwen_Qwen3-8B')
parser.add_argument('--gen', help='gen or nogen')
parser.add_argument('--extra_split', action='store_true', help='whether to include extra split for 3gen problems')
parser.add_argument('--gen_avg', action='store_true', help='calculate avg across gen')
parser.add_argument('--verbose', action='store_true', help='whether to print detailed logs')
parser.add_argument('--symb_avg', action='store_true', help='calculate accuracies for symbolic alphabet')
args = parser.parse_args()

if args.gen_avg and (args.model == None or args.promptstyle == None):
    print("When using --gen_avg, please also provide --model and --promptstyle")
    sys.exit(1)

# def compute_accuracy(trues, predictions):
#     correct = 0

#     for t, p in zip(trues, predictions):
#         p=p.strip(" '")
#         if (t==p):
#             correct+=1
#     return correct/len(trues)

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

if not args.gen_avg and not args.symb_avg:
    acc_dict = {}

    response_folder = f"{args.model}_prob_predictions_multi_alph/{args.gen}"
    response_file = f"{args.model}_letterstring_results_{args.num_permuted}_multi_alph_gptprobs{"_" + args.promptstyle if args.promptstyle else ''}.npz"
    print(f"Loading responses from {response_folder}/{response_file}...")
    responses = np.load(f"{response_folder}/{response_file}", allow_pickle=True)["data"].item()

    if args.extra_split:
        # Also load the extra split and merge
        extra_response_file = f"{args.model}_letterstring_results_{args.num_permuted}_multi_alph_gptprobs_{args.promptstyle}_extrasplit.npz"
        # This file contains only 3gensplit7, add it to the main responses
        print(f"Loading extra split responses from {response_folder}/{extra_response_file}...")
        extra_responses = np.load(f"{response_folder}/{extra_response_file}", allow_pickle=True)["data"].item()
        for alph in extra_responses:
            if alph in responses:
                responses[alph]['responses']['3gensplit7'] = extra_responses[alph]['responses']['3gensplit7']
                responses[alph]['targets']['3gensplit7'] = extra_responses[alph]['targets']['3gensplit7']
            else:
                responses[alph] = extra_responses[alph]

    partly_correct = {}

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
                if type(true[0]) == np.int64:
                    true = [str(x) for x in true]
                true = ''.join(true).lower()
                if args.verbose:
                    print(f'Pred: {pred}, True: {true}')
                if pred == true:
                    correct += 1
                elif true in pred:
                    decision = check_partly_correct(true, pred)
                    partly_correct["count"] = partly_correct.get("count", 0) + 1
                    partly_correct["pred"] = partly_correct.get("pred", []) + [pred]
                    partly_correct["true"] = partly_correct.get("true", []) + [true]   
                    partly_correct["decision"] = partly_correct.get("decision", []) + [decision]   
                    if decision:
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
            if prob_type.startswith('3gen'):
                if '3gen' in overall_accuracies:
                    overall_accuracies['3gen'].append(acc)
                else:
                    overall_accuracies['3gen'] = [acc]
            elif prob_type.startswith('2gen'):
                if '2gen' in overall_accuracies:
                    overall_accuracies['2gen'].append(acc)
                else:
                    overall_accuracies['2gen'] = [acc]
            elif prob_type in overall_accuracies:
                overall_accuracies[prob_type].append(acc)
            else:
                overall_accuracies[prob_type] = [acc]

    print(f"\n=== Overall Average Accuracies ===\n")
    for prob_type, accs in overall_accuracies.items():
        average_acc = sum(accs) / len(accs)
        print(f"Problem Type: {prob_type}, Average Accuracy: {average_acc}")

    # Save accuracies to a CSV file
    # df_rows = []
    # for alph, alph_acc in acc_dict.items():
    #     for prob_type, acc in alph_acc.items():
    #         df_rows.append({'Alphabet': alph, 'Problem_Type': prob_type, 'Accuracy': acc})
    # df = pd.DataFrame(df_rows)
    # output_csv = f"results/{args.model}_letterstring_accuracies_{args.num_permuted}_{args.prompt}.csv"
    # df.to_csv(output_csv, index=False)

    print(f"\n=== Partly Correct Predictions ===")
    if args.verbose:
        for t, p, d in zip(partly_correct.get('true', []), partly_correct.get('pred', []), partly_correct.get('decision', [])):
            print(f"True: {t}, Predicted: {p}, Decision: {d}")
    print(f"\nPartly correct counts: {partly_correct.get('count', 0)}")

elif args.gen_avg:
    # calculate average across gen (0-gen, 1-gen, 2-gen, 3-gen)
    # table looks like:
    # model | promptstyle | num_permuted | 0gen | 1gen | 2gen | 3gen
    gen_accs = {'0gen': [], '1gen': [], '2gen': [], '3gen': []}

    zero_gen_prob_names = ['succ', 'pred', 'add_letter', 'remove_redundant', 'fix_alphabet', 'sort', 'attn']
    one_gen_prob_names = ['larger_int', 'longer_targ', 'group', 'interleaved', 'letter2num', 'reverse']
    two_gen_prob_names = ['2gen_split1', '2gen_split2', '2gen_split3', '2gen_split4', '2gen_split5', '2gen_split6', '2gensplit7']
    three_gen_prob_names = ['3gen_split1', '3gen_split2', '3gen_split3', '3gen_split4', '3gen_split5', '3gen_split6', '3gensplit7']

    for gen in ['gen', 'nogen']:

        for num_permuted in [1,2,5,10,20]:
            response_folder = f"{args.model}_prob_predictions_multi_alph/{gen}"
            response_file = f"{args.model}_letterstring_results_{num_permuted}_multi_alph_gptprobs{"_" + args.promptstyle if args.promptstyle else ''}.npz"
            print(f"Loading responses from {response_folder}/{response_file}...")
            responses = np.load(f"{response_folder}/{response_file}", allow_pickle=True)["data"].item()
            all_gen_accuracies = {'0gen': [], '1gen': [], '2gen': [], '3gen': []}

            for alph in responses:
                all_prob_type_responses = responses[alph]['responses']
                all_trues = responses[alph]['targets']

                for prob_type in all_prob_type_responses.keys():
                    prob_type_responses = all_prob_type_responses[prob_type]
                    prob_type_trues = all_trues[prob_type]

                    total = 0 
                    correct = 0 

                    for pred, true in zip(prob_type_responses, prob_type_trues):
                        pred = pred.strip(" '").replace(" ", "").lower()
                        if type(true[0]) == np.int64:
                            true = [str(x) for x in true]
                        true = ''.join(true).lower()
                        if pred == true:
                            correct += 1
                        elif true in pred:
                            decision = check_partly_correct(true, pred)
                            if decision:
                                correct += 1
                        total += 1

                    if total > 0:
                        accuracy = correct / total
                        if prob_type in zero_gen_prob_names:
                            all_gen_accuracies['0gen'].append(accuracy)
                        elif prob_type in one_gen_prob_names:
                            all_gen_accuracies['1gen'].append(accuracy)
                        elif prob_type in two_gen_prob_names:
                            all_gen_accuracies['2gen'].append(accuracy)
                        elif prob_type in three_gen_prob_names:
                            all_gen_accuracies['3gen'].append(accuracy)

            for gen_key in gen_accs.keys():
                if len(all_gen_accuracies[gen_key]) > 0:
                    avg_acc = sum(all_gen_accuracies[gen_key]) / len(all_gen_accuracies[gen_key])
                    gen_accs[gen_key].append(avg_acc)
    
    print(f"\n=== Average Accuracies across generations ===\n")
    for num_permuted in [1,2,5,10,20]:
        print(f"Model: {args.model}, Prompt Style: {args.promptstyle}, Num Permuted: {num_permuted}")
        for gen_key in gen_accs.keys():
            avg_acc = gen_accs[gen_key][[1,2,5,10,20].index(num_permuted)]
            print(f"Generation: {gen_key}, Average Accuracy: {avg_acc}")

elif args.symb_avg:

    # the symbolic alphabet only has 2 problem types: succ and pred
    # only for 0 and 1 gen
    acc_dict = {}
    for gen in ['gen', 'nogen']:
        response_folder = f"{args.model}_prob_predictions_multi_alph/{gen}"
        response_file = f"{args.model}_letterstring_results_symb_multi_alph_gptprobs{"_" + args.promptstyle if args.promptstyle else ''}.npz"
        print(f"Loading responses from {response_folder}/{response_file}...")
        responses = np.load(f"{response_folder}/{response_file}", allow_pickle=True)["data"].item()

        accuracies = {}

        for alph in responses:
            print(f"Processing alphabet: {alph}")
            all_prob_type_responses = responses[alph]['responses']
            all_trues = responses[alph]['targets']

            alph_accuracies = {}

            for prob_type in all_prob_type_responses.keys():
                prob_type_responses = all_prob_type_responses[prob_type]
                prob_type_trues = all_trues[prob_type]

                total = 0
                correct = 0

                for pred, true in zip(prob_type_responses, prob_type_trues):
                    pred = pred.strip(" '").replace(" ", "").lower()
                    if type(true[0]) == np.int64:
                        true = [str(x) for x in true]
                    true = ''.join(true).lower()
                    if args.verbose:
                        print(f'Pred: {pred}, True: {true}')
                    if pred == true:
                        correct += 1
                    elif true in pred:
                        decision = check_partly_correct(true, pred)
                        if decision:
                            correct += 1
                    total += 1

                if total > 0:
                    accuracy = correct / total
                    print(f"Accuracy for problem type {prob_type}: {accuracy}")
                    alph_accuracies[prob_type] = accuracy
                else:
                    print(f"No predictions for problem type {prob_type}")

            # compute acc across alphabets
            for prob_type, acc in alph_accuracies.items():
                if prob_type in accuracies:
                    accuracies[prob_type].append(acc)
                else:
                    accuracies[prob_type] = [acc]
        # Save per gen accuracies
        acc_dict[gen] = accuracies  


    print(f"\n=== Summary of Accuracies for Symbolic Alphabet ===\n")
    for gen in acc_dict:
        print(f"--- Generation: {gen} ---")
        gen_accuracies = acc_dict[gen]
        for alph, acc in gen_accuracies.items():
            print(f"Alphabet: {alph}, Accuracy: {acc}")
        print("\n")        