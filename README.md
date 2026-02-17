# Analogical Reasoning in LLMs
Code associated with the thesis titled: "Does Analogical Prompting Enable True Analogical Reasoning in LLMs?" by Bart den Boef, written as part of the MSc Artificial Intelligence at the University of Amsterdam. Code is based on the code from the paper "Evaluating the Robustness of Analogical Reasoning in Large Language Models" [https://arxiv.org/abs/2411.14215]. 

The codebase includes code and data for each experiment, with each task having its own directory. Each directory has its own README with details on how to run the code. All code was run on a Nvidia A100 from Snellius, The `snellius` directory contains the jobfiles used and output files generated. 

Code for letterstring and digit matrix experiments is modified from [https://github.com/taylorwwebb/emergent_analogies_LLM/tree/main]

## Prerequisites
Prerequisites are listed below. Alternatively, use `environment.yml`.

- Python 3
- [OpenAI Python Library](https://github.com/openai/openai-python)
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [statsmodels](https://www.statsmodels.org/stable/index.html)
- [Matplotlib](https://matplotlib.org/)
- [pandas](https://pandas.pydata.org/)
- [transformers](https://pypi.org/project/transformers/)
