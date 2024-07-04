import os
from pathlib import Path

import pandas as pd

ROOT_PATH = "./results/"
INSTR_CSV = "./instructions/instructions_from_experts.csv"


SBATCH_SUBMIT_DIR = Path("./sbatch/submit_files")
SBATCH_LOGS_DIR = Path("./sbatch/logs")

SBATCH_SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
SBATCH_LOGS_DIR.mkdir(parents=True, exist_ok=True)


server = "[SERVER_NAME]"
max_job_at_once = 4


def main():
    models = [
        "Llama-2-7b-chat-hf",
        "asclepius",
        "clinical-camel-7b",
        "mistral-7b",
        "alpaca-7b",
        "medalpaca-7b",
        "Llama-2-13b-chat",
    ]

    df = pd.read_csv(INSTR_CSV)

    annotators = []
    for i in df.index:
        row = df.loc[i]
        annotator = row[0]
        if annotator not in annotators:
            annotators.append(annotator)

    cur_idx = 1

    while os.path.exists(SBATCH_SUBMIT_DIR / f"slurm_{cur_idx}.sh"):
        cur_idx += 1

    all_sbatches_files = []

    for model in models:
        for annotator in annotators:
            slurm_path = SBATCH_SUBMIT_DIR / f"slurm_{cur_idx}.sh"

            with open(slurm_path, "w", encoding="utf-8") as f:
                f.write("#!/bin/bash\n")

                if max_job_at_once is None:
                    f.write(f"#SBATCH --job-name=slurm_{cur_idx}\n")
                else:
                    if server == "jiang":
                        f.write(
                            f"#SBATCH --job-name=deform_{(cur_idx) % max_job_at_once}\n"
                        )
                    else:
                        f.write(
                            f"#SBATCH --job-name=robust_{(cur_idx) % max_job_at_once}\n"
                        )

                f.write(
                    f"#SBATCH --output={str(SBATCH_LOGS_DIR)}/slurm_{cur_idx}.out\n"
                )

                f.write(f"#SBATCH --partition={server}\n")
                f.write("#SBATCH --nodes=1\n")
                f.write("#SBATCH --gres=gpu:a100:1\n")
                f.write("#SBATCH --ntasks-per-node=1\n")
                f.write("#SBATCH --mem=64GB\n")
                f.write("#SBATCH --time=23:59:00\n")
                f.write("#SBATCH --exclude=d1026\n")

                if max_job_at_once is not None:
                    f.write(f"#SBATCH --dependency=singleton\n")

                f.write("source activate robust\n")
                f.write("\n")
                f.write(
                    f'python inference_auto_ie.py --annotator "{annotator}" --model "{model}" --root_dir="{ROOT_PATH}" \n'
                )
            f.close()
            all_sbatches_files.append(
                os.path.join(SBATCH_SUBMIT_DIR, f"slurm_{0 + cur_idx}.sh")
            )
            cur_idx += 1

    for s in all_sbatches_files:
        os.system(f"sbatch {s}")


if __name__ == "__main__":
    main()
