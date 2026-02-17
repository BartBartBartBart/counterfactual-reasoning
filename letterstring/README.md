# Letterstring analogies

## Generating problems
This directory contains code to generate letterstring analogy problems with permuted alphabets, in `gen_problems_by_alph.py`.

The problems used in the paper are all available in `problems`

## Testing GPT
GPT can be tested on the problems by running `eval_GPT_letterstring.py` with command line arguments `--promptstyle` to choose a promptstyle -- best results obtained with `hw` for Hodel and West's prompt. `--num_permuted` allows you to to choose a number of letters permuted.  Choices are: 1, 2, 5, 10, 20 or symb. `--gpt` allows you to choose a gpt model. Now that GPT-3 is deprecated, the choices are `35` for 3.5 or `4` for 4. `--gen` allows you to choose between generalized problems (gen) or non generalized problems (nogen)

#### Example usage. 
To evaluate GPT 4 on problems with 10 letters permuted, not generalized, with the prompt from Hodel and West 2024, you would call

`python eval_GPT_letterstring.py --gpt 4 --num_permuted 10 --gen nogen --promptstyle hw`

## Testing Qwen3 models
`eval_GPT_letterstring.py` has been extended to work with Qwen3 models. Instead of using `--gpt`, specify the model with `model`. Additionally, analogical prompting has been included in the code, which can be set by specifying "analogical" at the `num_permuted` flag. 

### Example usage. 
To run with Qwen3-8B, gen problems (1-, 2- and 3-gen), no permutations and analogical prompting, use: 

`python eval_GPT_letterstring.py --gen gen --model "Qwen/Qwen3-14B" --num_permuted 1 --promptstyle analogical`

## Comparative analysis of probabilities
To obtain information on the average probabilities of final answers, exemplars and the ratios of p(correct answer|previous output)/p(given answer|previous output), use `inspect_probs.py`. Works only for Qwen3 models. 

### Example usage.
To run with Qwen3-8B and gen problems, use:

`python inspect_probs.py --model Qwen/Qwen3-8B --gen gen`

## Calculate average accuracies
To gather accuracies from model outputs, use `get_accuracies.py`. It contains the following flags:
- `--promptstyle`: the promptstyle to calculate accuracies for (str)
- `--num_permuted`: the number of permutations (int)
- `--model`: the Qwen3 model (str)
- `--gen`: gen or nogen (str)
- `--gen_avg`: set this to calculate average across gens (bool)
- `--symb_avg`: set this to calculate average across symb alphs (bool)
- `--verbose`: set verbosity (0 or 1)

### Example usage. 
To run with Qwen3-8B, analogical prompting and obtain average accuracies across generalizations, use: 

`python get_accuracies.py --model Qwen_Qwen3-14B --promptstyle analogical --gen_avg`

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



