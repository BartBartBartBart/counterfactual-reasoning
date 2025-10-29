from signal import pause
import numpy as np
from copy import deepcopy
import random
import json
import argparse
import sys
import os

random.seed(1729)
np.random.seed(1729)

def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


## Problems generated using one of the following 6 transformations:
#
# Successorship
# [a b c d] [a b c e]
# [i j k l] [i j k m]
#
# Predecessorship
# [b c d e] [a c d e]
# [j k l m] [i k l m]
#
# Add letter to sequence
# [a b c d] [a b c d e]
# [i j k l] [i j k l m]
#
# Remove redundant character
# [a b b c d e] [a b c d e]
# [i j k l l m] [i j k l m]
#
# Fix alphabetic sequence
# [a w c d e] [a b c d e]
# [i j k p m] [i j k l m]
#
# Sort characters
# [a b e d c] [a b c d e]
# [i k j l m] [i j k l m]
#
#
## and between 0 and 3 generalizations, out of the following 6:
#
# Larger interval
# [a b c d] [a b c e]
# [i k m o] [i k m q]
#
# Longer target
# [a b c d] [a b c e]
# [i j k l m n o p] [i j k l m n o q]
#
# Grouping
# [a b c d] [a b c e]
# [i i j j k k l l] [i i j j k k m m]
#
# Interleaved X's
# [a b c d] [a b c e]
# [i x j x k x l x] [i x j x k x m x] 
#
# Letter-to-number
# [a b c d] [a b c e]
# [1 2 3 4] [1 2 3 5]
#
# Reversal
# [a b c d] [a b c e]
# [m l k j] [m l k i]
#
##

parser = argparse.ArgumentParser()
parser.add_argument('--num_permuted', help="give a number of letters in the alphabet to permute from 2 to 26")
args = parser.parse_args()
if args.num_permuted != 'greek' and args.num_permuted != 'symb':
    args.num_permuted = int(args.num_permuted)

# Generate derangement
def k_derange(k, letters):
    if k == 1:
        return None, None, letters
    
    to_shuffle = sorted(random.sample(range(len(letters)), k=k))
    shuffled = random.sample(to_shuffle, k=len(to_shuffle))
    not_derangement = sum([i == shuffled[to_shuffle.index(i)] for i in to_shuffle])

    while not_derangement:
        shuffled = random.sample(to_shuffle, k=len(to_shuffle))
        not_derangement = sum([i == shuffled[to_shuffle.index(i)] for i in to_shuffle])
    shuffled_alphabet = [letters[i] if i not in to_shuffle else letters[shuffled[to_shuffle.index(i)]] for i in range(len(letters))]
    return to_shuffle, [letters[i] for i in shuffled], shuffled_alphabet

# Successor transformation
def apply_succ(prob_letters):
    return [prob_letters[:-1], prob_letters[:-2] + [prob_letters[-1]]]

# Predecessor transformation
def apply_pred(prob_letters):
    return [prob_letters[1:], [prob_letters[0]] + prob_letters[2:]]

# Add letter to sequence
def apply_add_letter(prob_letters):
    return [prob_letters[:-1], prob_letters]

# Remove redundant letter
def apply_remove_redundant(prob_letters):
    redundant_loc = np.arange(len(prob_letters))
    np.random.shuffle(redundant_loc)
    redundant_loc = redundant_loc[0]
    prob_redundant = deepcopy(prob_letters)
    prob_redundant.insert(redundant_loc, prob_letters[redundant_loc])
    return [prob_redundant, prob_letters]

# Remove out-of-place character
def apply_fix_alphabet(prob_letters):
    remaining_letters = np.array(deepcopy(letters))
    remaining_letters = remaining_letters[np.all(np.expand_dims(np.array(remaining_letters),1) != np.expand_dims(np.array(prob_letters),0), 1)]
    np.random.shuffle(remaining_letters)
    insert_letter = remaining_letters[0]
    insert_loc = np.arange(len(prob_letters))
    np.random.shuffle(insert_loc)
    insert_loc = insert_loc[0]
    prob_letters_insert = deepcopy(prob_letters)
    prob_letters_insert[insert_loc] = insert_letter
    return [prob_letters_insert, prob_letters]

# Sort letters
def apply_sort(prob_letters):
    swap_loc = np.arange(len(prob_letters))
    np.random.shuffle(swap_loc)
    i_loc = swap_loc[0]
    j_loc = swap_loc[1]
    i_letter = prob_letters[i_loc]
    j_letter = prob_letters[j_loc]
    prob_swapped = deepcopy(prob_letters)
    prob_swapped[i_loc] = j_letter
    prob_swapped[j_loc] = i_letter
    return [prob_swapped, prob_letters]

# Attention check problem
def apply_attn(prob_letters):
    # same as apply_succ
    return [prob_letters[:-1], prob_letters[:-2] + [prob_letters[-1]]]

# Method for generating subset of problems
def gen_prob_subset(N_generalize=0, N_prob=10, standard_len=5, longer_targ_len=9, larger_int_size=2,
                    trans_allowed=['succ', 'pred', 'add_letter', 'remove_redundant', 'fix_alphabet', 'sort', 'attn'],
                    gen_allowed=['larger_int', 'longer_targ', 'group', 'interleaved', 'letter2num', 'reverse', 'realworld'],
                    shuffled_indices=None, symb_gen=False):
    # Initialize storage for problems
    all_prob = []
    all_trans = []
    all_gen = []
    all_src_letters = []
    all_tgt_letters = []
    count_ltli = 0
    while len(all_prob) < N_prob:
        # Sample source letters
        # Randomly specify whether source letter indices need to include shuffled indices
        if shuffled_indices is not None:
            src_shuffled = False
            if np.random.rand() > 0.5:
                src_shuffled = True
                offset = np.random.randint(0, standard_len)
                print(f'src offset is {offset}')
                index_choice = np.random.choice(shuffled_indices)
                print(f'src index choice is {index_choice}')
                src_start = min(max(0, index_choice - offset), (len(letters)-standard_len)) #makes sure we don't go off either end of the array
            else:        
                src_start = np.floor(np.random.rand() * (len(letters)-(standard_len-1))).astype(int)
        else:
            src_start = np.floor(np.random.rand() * (len(letters)-(standard_len-1))).astype(int)
        src_letters = letters[src_start:src_start+standard_len]
        # Sample generalizations
        random.shuffle(gen_allowed)
        generalize = gen_allowed[:N_generalize]
        print(generalize)
        # Sample target letters
        if 'realworld' in generalize:
            random.shuffle(realworld_linear)
            tgt_letters = realworld_linear[0]
        else:
            if 'longer_targ' in generalize and 'larger_int' in generalize:
                tgt_span = ((longer_targ_len - 1) * larger_int_size) #gatepost problem
                # print(f'tgt_span is: {tgt_span}')
                count_ltli += 1
                # For symbols, we can only generate a limited number of longer targ, larger int (ltli)
                if tgt_span <= len(letters)+count_ltli:
                    continue
                src_duplicate = True
                while src_duplicate:
                    if shuffled_indices is not None:
                        if not src_shuffled:
                            # need to make sure target includes shuffled letters
                            offset = np.random.randint(0, longer_targ_len)*larger_int_size
                            print(f'longer targ and larger int offset is {offset}')
                            index_choice = np.random.choice(shuffled_indices)
                            print(f'longer targ and larger int index choice is {index_choice}')
                            print(index_choice%offset)
                            print(index_choice - offset)
                            if index_choice - offset < index_choice%offset:
                                tgt_start = index_choice%offset
                            elif index_choice - offset > len(letters)-tgt_span: # if there is not enough room 
                                    tmp_tgt_start = (len(letters) - tgt_span)
                                    rem = (index_choice - tmp_tgt_start)%larger_int_size
                                    tgt_start = (tmp_tgt_start - (larger_int_size -rem))
                            else:
                                tgt_start = index_choice - offset
                            # print(f'tgt_start is {tgt_start}')
                        else:
                            tgt_start = np.floor(np.random.rand() * (len(letters)-(tgt_span-1))).astype(int)
                    else:
                        tgt_start = np.floor(np.random.rand() * (len(letters)-(tgt_span-1))).astype(int)
                    if src_start != tgt_start:
                        src_duplicate = False
                tgt_letters = letters[tgt_start:tgt_start+tgt_span+larger_int_size][::larger_int_size] #gatepost problem again
            elif 'longer_targ' in generalize and 'larger_int' not in generalize:
                src_duplicate = True
                while src_duplicate:
                    if shuffled_indices is not None:
                        if not src_shuffled:
                            offset = np.random.randint(0, longer_targ_len)
                            # print(f'longer targ offset is {offset}')
                            index_choice = np.random.choice(shuffled_indices)
                            # print(f'longer targ index choice is {index_choice}')
                            if index_choice - offset < 0:
                                tgt_start = 0
                            elif index_choice - offset > len(letters) - longer_targ_len:
                                tgt_start = len(letters) - longer_targ_len
                            else:
                                tgt_start = index_choice-offset
                        else:
                            tgt_start = np.floor(np.random.rand() * (len(letters)-(longer_targ_len-1))).astype(int)
                    else:
                        tgt_start = np.floor(np.random.rand() * (len(letters)-(longer_targ_len-1))).astype(int)
                    if src_start != tgt_start:
                        src_duplicate = False
                tgt_letters = letters[tgt_start:tgt_start+longer_targ_len]
            elif 'longer_targ' not in generalize and 'larger_int' in generalize:
                tgt_span = ((standard_len-1) * larger_int_size) #gatepost problem
                src_duplicate = True
                while src_duplicate:
                    if shuffled_indices is not None:
                        if not src_shuffled:
                            offset = np.random.randint(0, standard_len)*larger_int_size
                            # print(f'larger int offset is {offset}')
                            index_choice = np.random.choice(shuffled_indices)
                            # print(f'larger int index choice is {index_choice}')
                            # print(f'tgt_span is:{tgt_span}')
                            # print(index_choice%offset)
                            # print(index_choice - offset)
                            # index_choice%offset is the equivalent of 0: we cannot go below it
                            if index_choice - offset < index_choice%offset:
                                tgt_start = index_choice%offset
                            elif index_choice - offset > len(letters)-tgt_span:
                                tmp_tgt_start = (len(letters) - tgt_span)
                                rem = (index_choice - tmp_tgt_start)%larger_int_size
                                tgt_start = (tmp_tgt_start - (larger_int_size -rem))
                            else:
                                tgt_start = index_choice - offset

                            # print(f'tgt_start is {tgt_start}')
                        else:
                            tgt_start = np.floor(np.random.rand() * (len(letters)-(tgt_span-1))).astype(int)
                    else:
                        tgt_start = np.floor(np.random.rand() * (len(letters)-(tgt_span-1))).astype(int)
                    if src_start != tgt_start:
                        src_duplicate = False
                tgt_letters = letters[tgt_start:tgt_start+tgt_span+larger_int_size][::larger_int_size] #gatepost problem in Python
            elif 'longer_targ' not in generalize and 'larger_int' not in generalize:
                src_duplicate = True
                while src_duplicate:
                    if shuffled_indices is not None:
                        if not src_shuffled:
                            offset = np.random.randint(0, standard_len)
                            # print(f'offset is {offset}')
                            index_choice = np.random.choice(shuffled_indices)
                            # print(f'index choice is {index_choice}')
                            if index_choice - offset < 0:
                                tgt_start = 0
                            elif index_choice - offset > len(letters) - standard_len:
                                tgt_start = len(letters) - standard_len
                            else:
                                tgt_start = index_choice-offset
                            # print(f'tgt_start is {tgt_start}')
                        else:
                            tgt_start = np.floor(np.random.rand() * (len(letters)-(standard_len-1))).astype(int)
                    else:
                        tgt_start = np.floor(np.random.rand() * (len(letters)-(standard_len-1))).astype(int)
                    if src_start != tgt_start:
                        src_duplicate = False
                tgt_letters = letters[tgt_start:tgt_start+standard_len]
                # print(tgt_letters)
        # Reverse target letters
        if 'reverse' in generalize:
            tgt_letters.reverse()
        # Sample transformation
        random.shuffle(trans_allowed)
        trans = trans_allowed[0]
        # Apply transformation
        if trans == 'succ':
            src = apply_succ(src_letters)
            tgt = apply_succ(tgt_letters)
        elif trans == 'pred':
            src = apply_pred(src_letters)
            tgt = apply_pred(tgt_letters)
        elif trans == 'add_letter':
            src = apply_add_letter(src_letters)
            tgt = apply_add_letter(tgt_letters)
        elif trans == 'remove_redundant':
            src = apply_remove_redundant(src_letters)
            tgt = apply_remove_redundant(tgt_letters)
        elif trans == 'fix_alphabet':
            src = apply_fix_alphabet(src_letters)
            tgt = apply_fix_alphabet(tgt_letters)
        elif trans == 'sort':
            src = apply_sort(src_letters)
            tgt = apply_sort(tgt_letters)
        elif trans == 'attn':
            src = apply_attn(src_letters)
            tgt = apply_attn(tgt_letters)
        # Generalization from letters to numbers
        print(f'src is:{src}')
        print(f'tgt is:{tgt}')
        if 'letter2num' in generalize:
            new_tgt = []
            for i in range(len(tgt)):
                new_tgt_i = []
                for j in range(len(tgt[i])):
                    new_tgt_i.append(numbers[np.where(np.array(letters) == tgt[i][j])[0][0]])
                new_tgt.append(new_tgt_i)
            tgt = new_tgt
        # Interleaved X's (or 0's, if target composed of numbers)
        if 'interleaved' in generalize:
            new_tgt = []
            for i in range(len(tgt)):
                new_tgt_i = []
                for j in range(len(tgt[i])):
                    new_tgt_i.append(tgt[i][j])
                    if 'letter2num' in generalize:
                        new_tgt_i.append('0')
                    else:
                        new_tgt_i.append('x')
                new_tgt.append(new_tgt_i)
            tgt = new_tgt
        # Grouping
        if 'group' in generalize:
            new_tgt = []
            for i in range(len(tgt)):
                new_tgt_i = []
                for j in range(len(tgt[i])):
                    new_tgt_i.append(tgt[i][j])
                    new_tgt_i.append(tgt[i][j])
                new_tgt.append(new_tgt_i)
            tgt = new_tgt
        # Check that problem doesn't already exist
        prob = [src, tgt]
        duplicate = False
        for p_prev in range(len(all_prob)):
            if np.array(prob, dtype="object").shape == np.array(all_prob[p_prev], dtype="object").shape:
                if np.all(np.array(prob, dtype="object") == np.array(all_prob[p_prev], dtype="object")):
                    duplicate = True
        
        # check valid
        if shuffled_indices is not None:
            if len(set([letters[i] for i in shuffled_indices]).intersection(tgt_letters)) == 0 and len(set([letters[i] for i in shuffled_indices]).intersection(src_letters)) == 0:
                print(letters)
                print([letters[i] for i in shuffled_indices], src_letters, tgt_letters)
                print('problem invalid - exiting')
                sys.exit()
        # Add to problem subset
        if not duplicate:
            all_prob.append(prob)
            all_trans.append(trans)
            all_gen.append(generalize)
            all_src_letters.append(src_letters)
            all_tgt_letters.append(tgt_letters)
        
    return {'prob': all_prob, 'trans': all_trans, 'gen': all_gen, 'src_letters': all_src_letters, 'tgt_letters': all_tgt_letters}

# Add problems to json and numpy file
def save_prob(all_prob, prob_type_name, all_prob_js):
    # Convert to strings and save as json
    all_data = []
    for p in range(len(all_prob['prob'])):
        # A
        prompt = '['
        for i in range(len(all_prob['prob'][p][0][0])):
            prompt += str(all_prob['prob'][p][0][0][i])
            if i < len(all_prob['prob'][p][0][0]) - 1:
                prompt += ' '
        prompt += '] &nbsp ['
        # B
        for i in range(len(all_prob['prob'][p][0][1])):
            prompt += str(all_prob['prob'][p][0][1][i])
            if i < len(all_prob['prob'][p][0][1]) - 1:
                prompt += ' '
        prompt += ']<br>['
        # C
        for i in range(len(all_prob['prob'][p][1][0])):
            prompt += str(all_prob['prob'][p][1][0][i])
            if i < len(all_prob['prob'][p][1][0]) - 1:
                prompt += ' '
        prompt += '] &nbsp [&nbsp &nbsp]'
        # Add to dataset
        all_data.append({'prompt': prompt, 'prob_ind': p})
    # Add to javascript data
    all_prob_js[prob_type_name] = all_data
    return all_prob_js

# Split subset
def split_subset(all_prob, N_split):
    all_prob_split = []
    N_subset = int(len(all_prob['prob']) / N_split)
    for s in range(N_split):
        subset = {}
        for key in all_prob.keys():
            subset[key] = []
        for p in range(N_subset*s,N_subset*(s+1)):
            for key in all_prob.keys():
                subset[key].append(all_prob[key][p])
        all_prob_split.append(subset)
    return all_prob_split



# Linearly ordered real-world concepts - we do not use these
realworld_linear = [['cold', 'cool', 'warm', 'hot'],
                    ['love', 'like', 'dislike', 'hate'],
                    ['jack', 'queen', 'king', 'ace'],
                    ['penny', 'nickel', 'dime', 'quarter'],
                    ['second', 'minute', 'hour', 'day']]

num_alph = 7

all_prob_js = {}
all_prob_np = {}

human_prob = np.load(f'./problems/human/all_prob_{args.num_permuted}_7_human.npz', allow_pickle=True)['all_prob']

if args.num_permuted != 'greek' and args.num_permuted != 'symb':
    for i in range(num_alph):
        # generate alphabet
        #Alphabet
        letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']   
        # Get shuffled letters from the human problems
        shuffled_letters = human_prob.item()[f'alph_{i}']['shuffled_letters']
        shuffled_alphabet = human_prob.item()[f'alph_{i}']['shuffled_alphabet']
        if shuffled_letters is not None:
            shuffled_indices = [letters.index(l) for l in shuffled_letters]
        else:
            shuffled_indices = None

        # originally, we used the below:
        # shuffled_indices, shuffled_letters, shuffled_alphabet = k_derange(k=args.num_permuted, letters=letters)
        letters = deepcopy(shuffled_alphabet)
        print(shuffled_letters)
        N_letters = len(letters)
        # Numbers
        numbers = np.arange(N_letters) + 1

        # generate 10 problems using alphabet for each problem type
        N_prob = 10
        # Generate all basic analogies (zero generalizations)
        all_succ = gen_prob_subset(N_prob=N_prob, trans_allowed=['succ'], shuffled_indices=shuffled_indices)
        all_pred = gen_prob_subset(N_prob=N_prob, trans_allowed=['pred'], shuffled_indices=shuffled_indices)
        all_add_letter = gen_prob_subset(N_prob=N_prob, trans_allowed=['add_letter'], shuffled_indices=shuffled_indices)
        all_remove_redundant = gen_prob_subset(N_prob=N_prob, trans_allowed=['remove_redundant'], shuffled_indices=shuffled_indices)
        all_fix_alphabet = gen_prob_subset(N_prob=N_prob, trans_allowed=['fix_alphabet'], shuffled_indices=shuffled_indices)
        all_sort = gen_prob_subset(N_prob=N_prob, trans_allowed=['sort'], shuffled_indices=shuffled_indices)
        all_attn = gen_prob_subset(N_prob=N_prob, trans_allowed=['attn'], shuffled_indices=shuffled_indices)

        # Generate all problems with one generalization (randomly sample transformations)
        all_larger_int = gen_prob_subset(N_prob=N_prob, N_generalize=1, gen_allowed=['larger_int'], shuffled_indices=shuffled_indices)
        all_longer_targ = gen_prob_subset(N_prob=N_prob, N_generalize=1, gen_allowed=['longer_targ'], shuffled_indices=shuffled_indices)
        all_group = gen_prob_subset(N_prob=N_prob, N_generalize=1, gen_allowed=['group'], shuffled_indices=shuffled_indices)
        all_interleaved = gen_prob_subset(N_prob=N_prob, N_generalize=1, gen_allowed=['interleaved'], shuffled_indices=shuffled_indices)
        all_letter2num = gen_prob_subset(N_prob=N_prob, N_generalize=1, gen_allowed=['letter2num'], shuffled_indices=shuffled_indices)
        all_reverse = gen_prob_subset(N_prob=N_prob, N_generalize=1, gen_allowed=['reverse'], shuffled_indices=shuffled_indices)

        # Generate all problems with 2 and 3 generalizations
        all_2gen = gen_prob_subset(N_generalize=2, N_prob=N_prob*7, gen_allowed=['larger_int', 'longer_targ', 'group', 'interleaved', 'letter2num', 'reverse'], shuffled_indices=shuffled_indices)
        all_2gen_split = split_subset(all_2gen, 7)
        all_3gen = gen_prob_subset(N_generalize=3, N_prob=N_prob*7, gen_allowed=['larger_int', 'longer_targ', 'group', 'interleaved', 'letter2num', 'reverse'], shuffled_indices=shuffled_indices)
        all_3gen_split = split_subset(all_3gen, 7)


        # # Generate problems involving generalization to real-world concepts - we do not do this
        # all_realworld_succ = gen_prob_subset(standard_len=4, N_generalize=1, trans_allowed=['succ'], gen_allowed=['realworld'])
        # all_realworld_pred = gen_prob_subset(standard_len=4, N_generalize=1, trans_allowed=['pred'], gen_allowed=['realworld'])
        # all_realworld_add_letter = gen_prob_subset(standard_len=4, N_generalize=1, trans_allowed=['add_letter'], gen_allowed=['realworld'])
        # all_realworld_sort = gen_prob_subset(standard_len=4, N_generalize=1, trans_allowed=['sort'], gen_allowed=['realworld'])

        # Combine problems
        zero_gen_prob_types = [all_succ, all_pred, all_add_letter, all_remove_redundant, all_fix_alphabet, all_sort, all_attn]
        one_gen_prob_types = [all_larger_int, all_longer_targ, all_group, all_interleaved, all_letter2num, all_reverse]
        two_gen_prob_types = all_2gen_split
        three_gen_prob_types = all_3gen_split
        all_prob_types = zero_gen_prob_types + one_gen_prob_types + two_gen_prob_types + three_gen_prob_types
        # all_prob_types = [all_larger_int, all_longer_targ, all_group, all_interleaved, all_letter2num, all_reverse,
                        # all_2gen_split[0], all_2gen_split[1], all_2gen_split[2], all_2gen_split[3], all_2gen_split[4], all_2gen_split[5], all_2gen_split[6], 
                        # all_3gen_split[0], all_3gen_split[1], all_3gen_split[2], all_3gen_split[3], all_3gen_split[4], all_3gen_split[5], all_3gen_split[6]]
        # for the human problems:
        # [all_succ, all_pred, all_add_letter, all_remove_redundant, all_fix_alphabet, all_sort, all_attn]
        
        zero_gen_prob_names = ['succ', 'pred', 'add_letter', 'remove_redundant', 'fix_alphabet', 'sort', 'attn']
        one_gen_prob_names = ['larger_int', 'longer_targ', 'group', 'interleaved', 'letter2num', 'reverse']
        two_gen_prob_names = ['2gen_split1', '2gen_split2', '2gen_split3', '2gen_split4', '2gen_split5', '2gen_split6', '2gensplit7']
        three_gen_prob_names = ['3gen_split1', '3gen_split2', '3gen_split3', '3gen_split4', '3gen_split5', '3gen_split6', '3gensplit7']
        all_prob_type_names = zero_gen_prob_names + one_gen_prob_names + two_gen_prob_names + three_gen_prob_names
        # all_prob_type_names = ['larger_int', 'longer_targ', 'group', 'interleaved', 'letter2num', 'reverse',
                            # '2gen_split1', '2gen_split2', '2gen_split3', '2gen_split4', '2gen_split5', '2gen_split6', '2gensplit7',
                            # '3gen_split1', '3gen_split2', '3gen_split3', '3gen_split4', '3gen_split5', '3gen_split6', '3gensplit7']
        
        # human
        # ['succ', 'pred', 'add_letter', 'remove_redundant', 'fix_alphabet', 'sort', 'attn']

        # Set up js and np vars
        all_prob_js[f'alph_{i}'] = {}
        all_prob_np[f'alph_{i}'] = {}
        all_prob_js[f'alph_{i}']['shuffled_letters'] = shuffled_letters
        all_prob_np[f'alph_{i}']['shuffled_letters'] = shuffled_letters
        all_prob_js[f'alph_{i}']['shuffled_alphabet'] = letters
        all_prob_np[f'alph_{i}']['shuffled_alphabet'] = letters

        for p in range(len(all_prob_types)):
            all_prob_js[f'alph_{i}'] = save_prob(all_prob_types[p], all_prob_type_names[p], all_prob_js[f'alph_{i}'])
            all_prob_np[f'alph_{i}'][all_prob_type_names[p]] = all_prob_types[p]

elif args.num_permuted == 'symb':
    num_alph = 14
    
    symb_prob_type_names = ['succ', 'pred']
    # letters = ['>', '*', '+','<', '!','@','$',')','&','=']
    letters = ['>', '*', '+','<', '!','@','$',')','&','=', ":", "-", "(", "%", "~"]
    letter_dict = {0: [">", "$", ")", "*", "@", "=", "&", "<", "+", "!", ":", "-", "(", "%", "~"],\
                   1: ["&", ">", "$", "!", "+", ")", "<", "@", "=", "*", "~", "%", "(", "-", ":"]}
    for i in range(2, num_alph):
        letter_dict[i] = random.sample(letters, len(letters))

    symb_prob_types = []
    for i in range(num_alph):
        letters = letter_dict[i]
        print(letters)
        N_letters = len(letters)
        numbers = np.arange(N_letters) + 1
        # generate 10 problems using alphabet for each problem type
        # We cannot generate more than this because we run out of unique problems. A larger number
        # of symbols would allow us to do more.
        N_prob = 10
        shuffled_indices = None
        shuffled_letters = letters
        # Generate basic analogies for succ and pred (zero generalizations)
        symb_succ = gen_prob_subset(N_prob=N_prob, trans_allowed=['succ'], shuffled_indices=shuffled_indices)
        symb_pred = gen_prob_subset(N_prob=N_prob, trans_allowed=['pred'], shuffled_indices=shuffled_indices)

        symb_prob_types = [symb_succ, symb_pred]

        # Generate all problems with one generalization (randomly sample transformations)
        symb_larger_int = gen_prob_subset(N_prob=N_prob, N_generalize=1, gen_allowed=['larger_int'], \
                                          trans_allowed=['succ', 'pred'], shuffled_indices=shuffled_indices)
        symb_longer_targ = gen_prob_subset(N_prob=N_prob, N_generalize=1, gen_allowed=['longer_targ'], \
                                          trans_allowed=['succ', 'pred'], shuffled_indices=shuffled_indices)
        symb_group = gen_prob_subset(N_prob=N_prob, N_generalize=1, gen_allowed=['group'], \
                                          trans_allowed=['succ', 'pred'], shuffled_indices=shuffled_indices)
        symb_interleaved = gen_prob_subset(N_prob=N_prob, N_generalize=1, gen_allowed=['interleaved'], \
                                          trans_allowed=['succ', 'pred'], shuffled_indices=shuffled_indices)
        symb_letter2num = gen_prob_subset(N_prob=N_prob, N_generalize=1, gen_allowed=['letter2num'], \
                                          trans_allowed=['succ', 'pred'], shuffled_indices=shuffled_indices)
        symb_reverse = gen_prob_subset(N_prob=N_prob, N_generalize=1, gen_allowed=['reverse'], \
                                          trans_allowed=['succ', 'pred'],shuffled_indices=shuffled_indices)

        # Generate all problems with 2 and 3 generalizations
        symb_2gen = gen_prob_subset(N_generalize=2, N_prob=N_prob*7, \
                                   gen_allowed=['larger_int', 'longer_targ', 'group', 'interleaved', 'letter2num', 'reverse'], \
                                          trans_allowed=['succ', 'pred'], shuffled_indices=shuffled_indices)
        symb_2gen_split = split_subset(symb_2gen, 7)
        symb_3gen = gen_prob_subset(N_generalize=3, N_prob=N_prob*7, \
                                   gen_allowed=['larger_int', 'longer_targ', 'group', 'interleaved', 'letter2num', 'reverse'], \
                                   trans_allowed=['succ', 'pred'], shuffled_indices=shuffled_indices)
        symb_3gen_split = split_subset(symb_3gen, 7)

        # symb_prob_types = [symb_succ, symb_pred]
        symb_prob_types = [symb_larger_int, symb_longer_targ, symb_group, symb_interleaved, symb_letter2num, symb_reverse,
                            symb_2gen_split[0], symb_2gen_split[1], symb_2gen_split[2], symb_2gen_split[3], symb_2gen_split[4], symb_2gen_split[5],
                            symb_3gen_split[0], symb_3gen_split[1], symb_3gen_split[2], symb_3gen_split[3], symb_3gen_split[4], symb_3gen_split[5]]
        # symb_prob_type_names = ['succ', 'pred', 'add_letter', 'remove_redundant', 'fix_alphabet', 'sort', 'attn']
        symb_prob_type_names = ['larger_int', 'longer_targ', 'group', 'interleaved', 'letter2num', 'reverse',
                            '2gen_split1', '2gen_split2', '2gen_split3', '2gen_split4', '2gen_split5', '2gen_split6',
                            '3gen_split1', '3gen_split2', '3gen_split3', '3gen_split4', '3gen_split5', '3gen_split6']
 
    

        all_prob_js[f'symb_{i}'] = {}
        all_prob_np[f'symb_{i}'] = {}
        all_prob_js[f'symb_{i}']['shuffled_letters'] = shuffled_letters
        all_prob_np[f'symb_{i}']['shuffled_letters'] = shuffled_letters
        print(f'Saving with {letters}')
        all_prob_js[f'symb_{i}']['shuffled_alphabet'] = letters
        all_prob_np[f'symb_{i}']['shuffled_alphabet'] = letters
        

        for p in range(len(symb_prob_types)):
            all_prob_js[f'symb_{i}'] = save_prob(symb_prob_types[p], symb_prob_type_names[p], all_prob_js[f'symb_{i}'])
            all_prob_np[f'symb_{i}'][symb_prob_type_names[p]] = symb_prob_types[p]

save_path = './problems/gpt_human_alphs/'
check_path(save_path)

# Write numpy file
np.savez(save_path+f'all_prob_{args.num_permuted}_{num_alph}_gen.npz', all_prob=all_prob_np)
# Convert to json strings
all_prob_json_string = json.dumps(all_prob_js, indent=2)
# Write to js script
js_fname = save_path+f'all_prob_{args.num_permuted}_{num_alph}_gen.js'
js_fid = open(js_fname, 'w')
js_fid.write('var all_problems = ' + all_prob_json_string)
js_fid.close()

json_fname = save_path+f'all_prob_{args.num_permuted}_{num_alph}_gen.json'
json_fid = open(json_fname, 'w')
json_fid.write(all_prob_json_string)
json_fid.close()











