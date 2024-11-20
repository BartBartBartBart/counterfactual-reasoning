# building npz of problems just by replacing numbers with symbols
import numpy as np

all_prob = np.load(f'./problems/all_problems.npz', allow_pickle=True)

all_problems = all_prob['all_problems'].item()
all_prob_types = all_problems.keys()

symbs = ['%', '&', '*', '!', '(', '<', '~', '>', ':', '$'] #NB: This replaces ? with (
all_prob_types = list(all_prob['all_problems'].item().keys())
print(all_prob_types)
all_problems = all_prob['all_problems'].item()
print(list(all_problems['row_constant'].keys()))
all_symb_problems_np={}
for prob_type in all_prob_types:
    symb_probs = []
    for prob in all_problems[prob_type]['prob']:
        symb_prob = []
        for row in prob:
            symb_row = []
            for column in row:
                symb_col = [symbs[i] for i in column]
                symb_row.append(symb_col)
            symb_prob.append(symb_row)
        symb_probs.append(symb_prob)

    all_symb_choices = []
    for choices in all_problems[prob_type]['answer_choices']:
        symb_choices = []
        for choice in choices:
            symb_choice = [symbs[i] for i in choice]
            symb_choices.append(symb_choice)
        all_symb_choices.append(symb_choices)
    all_symb_problems_np[prob_type]={'prob':symb_probs,\
                                     'answer_choices':all_symb_choices,\
                                     'correct_ind':all_problems[prob_type]['correct_ind'],\
                                     'perm_invariant':all_problems[prob_type]['perm_invariant']}


np.savez('./all_problems_symb.npz', all_problems=all_symb_problems_np)
