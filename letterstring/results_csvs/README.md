# Human and GPT data

This directory contains .csv files with data from human and gpt experiments.

- `human_nogen.csv` contains data from human zero-generalization experiments
- `human_gen.csv` contains data from human generalization experiments
- `annotated_data.csv/xlsx` contains data from zero-generalization experiments for humans and GPT models annotated for error analysis
- `gpt_human_data.csv` contains data from all experiments.

The structure of `gpt_human_data.csv` is as follows:

- subj_id: numerical id for individual human subjects ranging from 1 to 311, and ids gpt_3, gpt_35, and gpt_4 for GPT 3, 3.5, and 4 respectively
- model: takes values 3, 35, 4, or human for for GPT 3, 3.5, and 4 or human participants respectively
- promptstyle: human participants all have promptstyle human. GPT models can take promptstyle human-like, hw, minimal, webb. We only report hw in the paper as this was the best.
- nperms: takes np_1, np_2, np_5, np_10, np_20, np_symb for alphabets with 0, 2, 5, 10, 20 letters permuted and symbolic alphabets respectively.
- alph: a key to the specific alphabet used.
- prob_type: the problem type.
- prob_ind: an index to the problem used within the problem type, for that alphabet, for that number of permutations
- response: the response given
- source1, source2, target1, correct_answer: the actual letterstring problem and the correct answer
- given_answer: the response string parsed into a list
- correct: True/False, whether the answer was correct or not
- total: just a column of 1 to help calculate total.
- num_gen: number of generalizations, ranges from 0 to 3

Subjects are individuated by subj_id.
Items are individuated by the combination of values taken in nperms, alph, prob_type, prob_ind. For example, the values (np_5, alph_3, pred, 2) specfies a problem with an alphabet with 5 letters permuted, the specific such alphabet is called alph_3, the problem type is pred, and the specific problem is indexed by 2.

