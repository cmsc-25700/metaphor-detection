# Metaphor Detection
NLP Group Research Project to replicate <a href="https://arxiv.org/pdf/1808.09653.pdf" target="_blank">Neural Metaphor Detection in Context</a>
### Project Overview
**Language Requirements:**
Python 3.9

**Required Libraries:**
For requirements related to reproducing study results, see [gao-g-requirements.txt](gao-g-requirements.txt)

### Project Installation
1. In the project root run `bash install.sh` to verify python installation and create pip environment.
2. Activate the virtual environment by running `source gao-env/bin/activate`.
3. Download the data and unzip it into the resources directory: <a href="https://drive.google.com/file/d/18tBHegty7sWreqj9Fp8zETKloeOpsGHO/view?usp=sharing" target="_blank">Neural Metaphor Detection in Context Data (zip)</a>

-- To run steps 4 and 5 in Colab, see [notebooks/Download_large_data.ipynb](Download_large_data.ipynb) --
4. (Optional) GloVe: download pretrained vectors, glove.840B.300d.zip from [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/) and unzip it under resources/glove directory and change file name to `glove840B300d.txt`.
5. (Optiona) ELMo vectors: available upon request