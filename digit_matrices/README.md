# Digit matrix analogies

## Generating problems
This directory contains code to generate digit matrix analogy problems with symbol alphabets or alternative blanks, in `gen_problems_coords.py`. The original problems used by Webb, Holyoak, Lu (2023) can be converted to a symbol alphabet using `convert_to_symb.py`.

The problems used in the paper are all available in `problems`.

## Testing GPT
GPT can be tested on the problems by running `eval_GPT_matprob.py` with command line arguments `--id` to choose GPT model, `--user_prompt_num` and `--sys_prompt_num` for alternative user and system prompts -- best results obtained with 2 and 1 respectively `--prob_format` specifies the problem format out of 'digits', 'symb', or 'coords'. GPT model setup needs to be defined by the user.

#### Example usage. 
To evaluate GPT 4 (0613) on symbol problems with system prompt 1 and user prompt 2, you would call

`python eval_GPT_digitmat.py --gpt 0613 --sys_prompt_num 1 --user_prompt_num 2 --prob_format symb`


## Results
Results are stored in `results` directory as `.npz` files. Results are processed and saved as csv in `all_data.csv`.

## Human data
Human data is available in `all_data.csv` for numeric subj_id

## Data analysis and plotting
A notebook in `plotting` gives code to generate all plots in the paper.



