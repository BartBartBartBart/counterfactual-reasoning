# Story analogies
The files `stories_orig.txt` and `stories_new.txt` contain respectively the original and new stories.

The directory `gpt_experiment` contains files containing the stories in json format, and code to elicit responses from GPT models.

`gpt_results` contains GPT responses in the story task.

`human_data` contains human responses to the story task and a notebook to calculate accuracies.

# Get model responses
To obtain responses from the model, use `gpt_experiment/stories_gpt.py`. The code has been adapted to support Qwen3 models and analogical prompting. See info on the flags and usage below: 
- `--model`: model version to use: 0125, 1106, 0613, Qwen_Qwen3-8B or Qwen_Qwen3-14B
- `--version`: new or orig
- `--promptstyle`: default, concise, detailed or analogical
- `--ordering`: ab (correct-first), ba or random
- `--verbose`: set verbosity (bool)

## Example usage
To run with Qwen3-8B, analogical prompting, paraphrased stories, random ordering and verbosity, use 

`python stories_gpt.py --model Qwen/Qwen3-8B --promptstyle analogical --version new --ordering random --verbose`


