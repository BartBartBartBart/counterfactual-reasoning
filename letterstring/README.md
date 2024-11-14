# Letterstring analogies

## Generating problems
This directory contains code to generate letterstring analogy problems with permuted alphabets, in `gen_problems_by_alph.py`.

The problems used in the paper are all available in `problems`

## Testing GPT
GPT can be tested on the problems by running `eval_GPT_letterstring.py` with command line arguments `--promptstyle` to choose a promptstyle -- best results obtained with `hw` for Hodel and West's prompt. `--num_permuted` allows you to to choose a number of letters permuted.  Choices are: 1, 2, 5, 10, 20 or symb. `--gpt` allows you to choose a gpt model. Now that GPT-3 is deprecated, the choices are `35` for 3.5 or `4` for 4. `--gen` allows you to choose between generalized problems (gen) or non generalized problems (nogen)

#### Example usage. 
To evaluate GPT 4 on problems with 10 letters permuted, not generalized, with the prompt from Hodel and West 2024, you would call

`python eval_GPT_letterstring.py --gpt 4 --num_permuted 10 --gen nogen --promptstyle hw`

## Evaluating GPT on counterfactual comprehesion test
GPT can be tested on the CCC by running `eval_GPT_letterstring_control.py` with command line arguments `--num_permuted` allowing you to to choose a number of letters permuted.  Choices are: 1, 2, 5, 10, 20 or symb. `--gpt` allows you to choose a gpt model. Now that GPT-3 is deprecated, the choices are `35` for 3.5 or `4` for 4. `--problem` allows you to choose between successor (`succ`) or predecessor (`pred`) CCC tests.

#### Example usage. 
To evaluate GPT 3.5 on the CCC with 20 letters permuted and the successor problem, you would call

`python eval_GPT_letterstring_control.py --gpt 35 --num_permuted 20 --problem pred`

## Results
Results are stored in `GPT{X}_prob_predictions_multi_alph` directories as `.npz` files. Results are processed and saved as csv in `results_csvs`.

## Human data
Human data is available in `results_csvs` as `human_gen.csv`, `human_nogen.csv`, and in `gpt_human_data.csv`

## Data analysis and plotting
A notebook in `plotting` gives code to generate all plots in the paper.



