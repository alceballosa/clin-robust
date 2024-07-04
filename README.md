# Robustness of Clinical and non-Clinical Instruction Finetuned Models

Repository for the paper titled "Open (Clinical) LLMs are Sensitive to Instruction Phrasings", to appear in the BioNLP 2024 workshop as part of ACL 2024.

## Create the environment

```bash
conda env create -f _env/env.yml
conda activate robust
pip install -r requirements_torch.txt 
pip install -r requirements_base.txt
```

## Downloading the data

Download the data from https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/ and place it under the following path (relative to the repository's root):

```bash
./datasets/n2c2_raw/
./datasets/n2c2_raw/2006
...
./datasets/n2c2_raw/2018
```

## Preprocessing the data:

To preprocess the data, ensure you've created the above folders and then run all the notebooks present under  `preprocessing_notebooks`. 

## Running inference

Having preprocessed the data, you can run inference as follows:

```bash 
python inference_auto.py --annotator="Annotator_9" --model="clinical-camel-7b" --root_dir="./results" # clf tasks
python inference_auto.py --annotator="Annotator_9" --model="clinical-camel-7b" --root_dir="./results"  # ie tasks
```

For ease of use, we provide a SLURM submit file that enables running the entire study by queueing up to N simultaneous jobs. A non-SLURM script can be easily derived from it if required.

Be aware that, under the provided configuration (bfloat16, batch size 1) the pipeline will require at least one A6000, A100 80 GB or H100 GPU. We cannot guarantee reproducibility of the results using quantization to fit lower capacity GPUs, although they should be fairly similar. Moreover, the pipeline has only been tested with up to batch size 1 and is not likely to work with higher values due to how we run evaluation in the binary classification case.

The following models are supported (should be referenced using the strings below). The actual model versions are the ones described in the paper (e.g., we're using Mistral 7b instruct 0.2, not the base version):

```bash
"Llama-2-7b-chat-hf",
"Llama-2-13b-chat",
"mistral-7b",
"asclepius",
"clinical-camel-7b",
"alpaca-7b",
"medalpaca-7b",
```

## Postprocessing and results

To post-process the outputs, run the following commands from the root of the repository:

```bash
mkdir postprocessing/processed_csvs
python ./postprocessing/eval_script_extract.py

```
After that, you can access the notebooks and run them to produce the relevant figures.

For ease of reproducibility, we include pre-computed results as csv files under `postprocessing/processed_csvs`. 

After running each notebook, the relevant figures will be stored under `postprocessing/plots/`.

## Contact

If required, contact authors Monica Munnangi or Alberto Ceballos Arroyo via the email addresses listed in the paper or open a GitHub issue.
