import argparse
import os
import optuna
import sys
import re
import ruamel
from ruamel.yaml.comments import CommentedMap


def generate_sbatch(
    input_data: str,
    yaml_list_file: str,
    n_files: int,
    working_dir: str,
    apptainer_image: str,
    n_jobs_in_parallel: int = 1,
):
    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name=rna_evolve
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=23G
#SBATCH --gres-flags=enforce-binding
#SBATCH --array=0-{n_files - 1}%{n_jobs_in_parallel}
#SBATCH --output=rna_evolution_%j.log

# This sbatch processes always the same input.

# Activate environment that has dress installed
#source activate genSplicing_experiments

# Load individual configuration files and run dress on each of them using JobArrays
readarray yaml_files < $(readlink -f "{yaml_list_file}")
yaml=${{yaml_files[$SLURM_ARRAY_TASK_ID]}}

CMD="srun dress generate --config $yaml {input_data}"
echo $CMD
$CMD

#conda deactivate
"""
    return sbatch_script


def submit_job(
    sbatch: str, outdir: str, yaml_list_fn: str, after_id: int | None = None
):
    """
    Submit job in the server.
    """
    match = re.search(r"_\d+\.", yaml_list_fn)
    if match:
        counter = match.group(0).strip("_").strip(".")
        sbatch_name = f"run_evolutions_{counter}.sbatch"
    else:
        sbatch_name = "run_evolutions.sbatch"

    sbatch_script = os.path.join(outdir, sbatch_name)

    with open(sbatch_script, "w") as f:
        f.write(sbatch)

    base_cmd = "sbatch"
    if after_id is not None:
        base_cmd += f" --dependency=afterok:{after_id}"

    os.system(f"{base_cmd} {sbatch_script} {yaml_list_fn}")


def get_all_keys(data):
    """
    Get all keys from the nested yaml file.
    """
    keys = []
    if isinstance(data, CommentedMap):
        for key in data.keys():
            keys.append(key)
            if isinstance(data[key], CommentedMap):
                keys.extend(get_all_keys(data[key]))
    return keys


def generate_yaml_list(
    default_yaml: str,
    outdir: str,
    n_seeds_per_combination: int,
    combinations: dict = {"default": {}},
):
    """
    Generate yaml files with the combinations of parameters to test and
    return a list with the names of the yaml files.
    """

    yaml_list = []
    for _run_id, parameters in combinations.items():
        yaml = ruamel.yaml.YAML()
        with open(default_yaml) as f:
            settings = yaml.load(f)

        all_keys = get_all_keys(settings)

        for seed in range(n_seeds_per_combination):
            parameters["outdir"] = os.path.join(outdir, _run_id)
            parameters["seed"] = seed
            run_id = f"{_run_id}_seed{seed}"

            if len(parameters) > 0:
                for arg_name, new_value in parameters.items():
                    assert (
                        arg_name in all_keys
                    ), f"Parameter {arg_name} not found in yaml file. Change combinations of parameters to test."
                    settings = update_argument_value(settings, arg_name, new_value)

            outfile = os.path.join(outdir, run_id + ".yaml")
            with open(outfile, "w") as f:
                yaml.dump(settings, f)

            yaml_list.append(outfile)

    return yaml_list


def write_yaml_list(
    yaml_list: list, outdir: str, filename: str = "list_yaml_configs.txt"
):
    basename, extension = os.path.splitext(filename)
    fn = os.path.join(outdir, filename)

    increment = 2
    while os.path.exists(fn):
        if increment == 0:
            fn = os.path.join(outdir, filename)
        else:
            new_fn = f"{basename}_{increment}{extension}"
            fn = os.path.join(outdir, f"{new_fn}")
        increment += 1

    with open(fn, "w") as f:
        for yml in yaml_list:
            f.write(f"{yml}\n")

    return fn


def update_argument_value(data, arg_name, new_value):
    """
    Update value of a given argument in the nested yaml file.
    """
    if isinstance(data, CommentedMap):
        for key, value in data.items():
            if key == arg_name:
                data[key] = new_value
            elif isinstance(value, CommentedMap):
                update_argument_value(value, arg_name, new_value)
    return data


def get_params(trial):
    # Random search
    if "operators_weight_0" not in trial.params:
        trial.params["operators_weight"] = [0]
        trial.params["elitism_weight"] = [0]
        trial.params["novelty_weight"] = [1]
    else:
        # Conversion between optuna (each list argument is a different argument) and yaml
        trial.params["operators_weight"] = [trial.params["operators_weight_0"]]
        trial.params["elitism_weight"] = [trial.params["elitism_weight_0"]]
        trial.params["novelty_weight"] = [trial.params["novelty_weight_0"]]
        del trial.params["operators_weight_0"]
        del trial.params["elitism_weight_0"]
        del trial.params["novelty_weight_0"]
    return [trial.number, trial.params]


def check_best_trials(file):
    name = os.path.basename(file).replace(".db", "")
    is_random = "_random" if "random" in name else ""
    fitness_function = name.split("tpe_")[1]
    study = optuna.load_study(
        study_name=name,
        storage=f"sqlite:///{file}",
    )

    best_params = {}
    # if len(study.best_trials) > 5:
    #     sorted_objects = sorted(
    #         study.best_trials, key=lambda obj: obj.values[0], reverse=True
    #     )[:5]

    # else:
    best_trial_ids = (
        study.trials_dataframe()
        .drop_duplicates(subset="value", keep="first")
        .sort_values("value", ascending=False)
        .head(5)
        .index
    )
    # best_trial_ids = (
    #     study.trials_dataframe().sort_values("value", ascending=False).head(5).index
    # )
    sorted_objects = [trial for trial in study.trials if trial.number in best_trial_ids]

    for i, _trial in enumerate(sorted_objects):
        i += 1
        _params = get_params(_trial)
        _params[1]["fitness_function"] = fitness_function
        best_params[f"{fitness_function}{is_random}_{i}_trial{_params[0]}"] = _params[1]

    return best_params


def main():
    parser = argparse.ArgumentParser(
        description="Extract the paramater configuration of the best optuna trials and submit slurm jobs to test each parameter scheme across 5 different seeds."
    )
    parser.add_argument(
        dest="optuna_results",
        help="File listing (one per line) the results (sqlite file) of a single optuna run",
    )

    parser.add_argument(
        dest="input", help="Input exon used for hyperparameter optimization."
    )

    parser.add_argument(
        "--default_yaml",
        metavar="",
        help="Default yaml file to change hyperparameters to test. Default: generate.yaml",
        default="generate.yaml",
    )

    parser.add_argument(
        "--outdir",
        default="1_bestTrialsMultipleSeeds",
        help="Output directory where all results will be stored. Default: 1_bestTrialsMultipleSeeds",
    )

    parser.add_argument(
        "--apptainer_image",
        metavar="",
        help="Path to apptainer image with all dependencies. Default: /home/pbarbosa/apptainer/images/pbarbosa_spliceai_0.0.2.sif",
        default="/home/pbarbosa/apptainer/images/pbarbosa_spliceai_0.0.2.sif",
    )

    parser.add_argument(
        "--working_dir",
        metavar="",
        help="Path to the repository of the generativeSplicing project. Default: /home/pbarbosa/git_repos/dress",
        default="/home/pbarbosa/git_repos/dress",
    )

    args = parser.parse_args()

    filelist = open(args.optuna_results, "r")
    combinations = {}
    common_to_all = {
        "stopping_criterium": ["archive_size", "time"],
        "stop_at_value": [5000, 5],
        "track_full_archive": False,
        "prune_archive_individuals": True,
    }
    for hyperopt in filelist:
        combinations.update(check_best_trials(hyperopt.rstrip()))

    for k, v in combinations.items():
        combinations[k] = {**v, **common_to_all}

    os.makedirs(args.outdir, exist_ok=True)
    yaml_list = generate_yaml_list(
        args.default_yaml,
        args.outdir,
        n_seeds_per_combination=5,
        combinations=combinations,
    )
    yaml_list_fname = write_yaml_list(yaml_list, args.outdir)
    sbatch = generate_sbatch(
        input_data=args.input,
        yaml_list_file=yaml_list_fname,
        n_files=len(yaml_list),
        working_dir=args.working_dir,
        apptainer_image=args.apptainer_image,
        n_jobs_in_parallel=1,
    )

    submit_job(sbatch, args.outdir, yaml_list_fname)


if __name__ == "__main__":
    main()
