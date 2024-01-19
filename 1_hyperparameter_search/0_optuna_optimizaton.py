import argparse
from functools import partial
import time
import numpy as np
import ruamel.yaml

import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from datasetgeneration.archive import Archive
from datasetgeneration.evolution import (
    do_evolution,
    get_score_of_input_sequence,
)
from datasetgeneration.json_schema import flatten_dict
from datasetgeneration.preprocessing.gtf_cache import preprocessing

from datasetgeneration.preprocessing.utils import tabular_file_to_genomics_df


def print_best_callback(study, trial):
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")


def get_yaml(
    default_yaml: str,
) -> dict:
    yaml = ruamel.yaml.YAML()
    with open(default_yaml) as f:
        settings = yaml.load(f)

    return settings


def _constraints(trial):
    return trial.user_attrs["constraint"]


def _objetive(trial, _input, fit_fun, is_random, params):
    # define parameters
    params["disable_tracking"] = True
    params["stopping_criterium"] = ["time", "archive_size"]
    params["stop_at_value"] = [10, 5000]
    params["stop_when_all"] = False
    params["prune_archive_individuals"] = False

    # fitness and archive diversity
    params["fitness_function"] = fit_fun
    # params["fitness_function"] = trial.suggest_categorical(
    #     "fitness_function", choices=["bin_filler", "increase_archive_diversity"]
    # )

    # grammar parameters
    params["max_diff_units"] = trial.suggest_int("max_diff_units", 1, 6)
    params["max_insertion_size"] = trial.suggest_int("max_insertion_size", 1, 5)
    params["max_deletion_size"] = trial.suggest_int("max_deletion_size", 1, 5)
    params["snv_weight"] = trial.suggest_float("snv_weight", 0, 1, step=0.05)
    params["insertion_weight"] = trial.suggest_float(
        "insertion_weight", 0, 1, step=0.05
    )
    params["deletion_weight"] = trial.suggest_float("deletion_weight", 0, 1, step=0.05)

    # Evolutionary alg
    params["population_size"] = trial.suggest_int(
        "population_size", 100, 2000, step=200
    )

    # Dynamic operator weights
    # number_of_updates = trial.suggest_categorical(
    #     "n_updates", [None, int(1), int(2), int(3), int(4), int(5)]
    # )

    if is_random:
        selection_method = "tournament"
        custom_mutation_operator = False
        custom_mutation_operator_weight = 0
        mutation_probability = 0
        crossover_probability = 0
        operators_weight = [0]
        elitism_weight = [0]
        novelty_weight = [1]
    else:
        selection_method = trial.suggest_categorical(
            "selection_method", choices=["tournament", "lexicase"]
        )
        custom_mutation_operator = trial.suggest_categorical(
            "custom_mutation_operator", choices=[True, False]
        )
        custom_mutation_operator_weight = trial.suggest_float(
            "custom_mutation_operator_weight", 0, 1, step=0.1
        )
        mutation_probability = trial.suggest_float(
            "mutation_probability", 0.2, 1, step=0.1
        )
        crossover_probability = trial.suggest_float(
            "crossover_probability", 0.05, 0.5, step=0.05
        )
        operators_weight = [trial.suggest_float(f"operators_weight_0", 0, 1, step=0.1)]
        elitism_weight = [trial.suggest_float(f"elitism_weight_0", 0, 1, step=0.1)]
        novelty_weight = [trial.suggest_float(f"novelty_weight_0", 0, 1, step=0.1)]

    # if number_of_updates is None:
    params["update_weights_at_generation"] = None

    # else:
    #     update_at_gens = []
    #     possible_generations = [int(x) for x in list(np.arange(2, 30))]

    #     for i in range(number_of_updates):
    #         _gen = trial.suggest_categorical(
    #             name=f"update_weights_at_generation_{i}",
    #             choices=possible_generations,
    #         )

    #         update_at_gens.append(_gen)

    #         operators_weight.append(
    #             trial.suggest_float(f"operators_weight_{i + 1}", 0, 1, step=0.2)
    #         )
    #         elitism_weight.append(
    #             trial.suggest_float(f"elitism_weight_{i + 1}", 0, 1, step=0.2)
    #         )
    #         novelty_weight.append(
    #             trial.suggest_float(f"novelty_weight_{i + 1}", 0, 1, step=0.2)
    #         )

    #     params["update_weights_at_generation"] = sorted(update_at_gens)
    params["selection_method"] = selection_method
    params["custom_mutation_operator"] = custom_mutation_operator
    params["custom_mutation_operator_weight"] = custom_mutation_operator_weight
    params["mutation_probability"] = mutation_probability
    params["crossover_probability"] = crossover_probability
    params["operators_weight"] = operators_weight
    params["elitism_weight"] = elitism_weight
    params["novelty_weight"] = novelty_weight

    # Constraints exceeding 1
    c0 = (params["mutation_probability"] + params["crossover_probability"]) - 1
    c1 = (
        params["snv_weight"] + params["insertion_weight"] + params["deletion_weight"]
    ) - 1
    c2 = [
        (elitism_weight[i] + operators_weight[i] + novelty_weight[i]) - 1
        for i in range(len(operators_weight))
    ]

    # Constraints with very low sums
    c3 = 0.5 - (
        params["snv_weight"] + params["insertion_weight"] + params["deletion_weight"]
    )

    c4 = [
        0.5 - (elitism_weight[i] + operators_weight[i] + novelty_weight[i])
        for i in range(len(operators_weight))
    ]
    trial.set_user_attr(
        "constraint",
        ((c0, c1, c3) + tuple(c2) + tuple(c4)),
    )

    start_time = time.time()
    try:
        archive = do_evolution(_input, **params)
    except Exception as e:
        raise optuna.TrialPruned()

    end_time = time.time()
    return archive, end_time - start_time


class Sampler(object):
    def __init__(self, study_name, input, args) -> None:
        self.study_name = study_name
        self.storage_name = f"sqlite:///{study_name}.db"
        self.input = input
        self.args = args

    def eval_solution(self, archive: Archive):
        
        qual = archive.quality

        print(
            f"Archive quality: {qual}\n"
        )
        return qual

class NSGAIISampler(Sampler):
    def __init__(
        self, study_name, n_trials, _input, fit_func, is_random, _args
    ) -> None:
        super().__init__(study_name, _input, _args)
        self.sampler = optuna.samplers.NSGAIISampler(
            seed=11, constraints_func=_constraints
        )
        study = optuna.create_study(
            directions=["maximize", "minimize"],
            study_name=self.study_name,
            storage=self.storage_name,
            sampler=self.sampler,
            load_if_exists=True,
        )
        study.optimize(
            partial(
                self.objective,
                input=_input,
                fit_fun=fit_func,
                is_random=is_random,
                params=_args,
            ),
            callbacks=[
                MaxTrialsCallback(n_trials, states=(TrialState.COMPLETE,)),
                print_best_callback,
            ],
            show_progress_bar=True,
        )

        self.study = study

    def objective(self, trial, input=None, fit_fun=None, is_random=None, params=None):
        archive, runtime = _objetive(
            trial=trial,
            _input=input,
            fit_fun=fit_fun,
            is_random=is_random,
            params=params,
        )

        if isinstance(archive, float):
            return None
        else:
            return self.eval_solution(archive), runtime


class TPESampler(Sampler):
    def __init__(self, study_name, n_trials, _input, fit_func, is_random, _args):
        super().__init__(study_name, _input, _args)
        self.sampler = optuna.samplers.TPESampler(
            seed=11, constraints_func=_constraints
        )

        study = optuna.create_study(
            direction="maximize",
            study_name=self.study_name,
            storage=self.storage_name,
            sampler=self.sampler,
            load_if_exists=True,
        )
        study.optimize(
            partial(
                self.objective,
                input=_input,
                fit_fun=fit_func,
                is_random=is_random,
                params=_args,
            ),
            callbacks=[
                MaxTrialsCallback(n_trials, states=(TrialState.COMPLETE,)),
                print_best_callback,
            ],
            show_progress_bar=True,
        )

        self.study = study

    def objective(self, trial, input=None, fit_fun=None, is_random=None, params=None):
        archive, _ = _objetive(
            trial=trial,
            _input=input,
            fit_fun=fit_fun,
            is_random=is_random,
            params=params,
        )
        if isinstance(archive, float):
            return None
        else:
            return self.eval_solution(archive)


def main():
    parser = argparse.ArgumentParser(
        description="Generate combinations of parameters to test based on the fixed input required and automatically generate sbatch scripts to submit jobs. Edit the python file to add/remove combinations of parameters to test"
    )
    parser.add_argument(
        dest="input",
        help="Input exon to run several evolutions.",
    )
    parser.add_argument(
        dest="optimization",
        help="Type of optimization to perform",
        choices=["nsgaii", "tpe"],
    )

    parser.add_argument(
        "--is_random",
        action="store_true",
        help="Whether optimization will be based on random search",
    )
    parser.add_argument(
        "-ff",
        "--fitness_function",
        metavar="",
        choices=["bin_filler", "increase_archive_diversity"],
        default="bin_filler",
        help="Fitness function to use",
    )

    parser.add_argument(
        "--default_yaml",
        metavar="",
        help="Default yaml file to change hyperparameters to test. Default: generate.yaml",
        default="generate.yaml",
    )

    args = parser.parse_args()
    config = get_yaml(args.default_yaml)
    _args = flatten_dict(config["generate"])

    df = tabular_file_to_genomics_df(
        args.input,
        col_index=0,
        is_0_based=False,
        header=0,
    )

    seqs, ss_idx = preprocessing(df, **_args)

    _input = {
        "seq_id": "chr10:89010439-89012181(+)_ENST00000652046",
        "seq": seqs["chr10:89010439-89012181(+)_ENST00000652046"],
        "ss_idx": ss_idx["chr10:89010439-89012181(+)_ENST00000652046"],
        "dry_run": False,
    }

    _input = get_score_of_input_sequence(_input, "spliceai", "mean")
 
    if args.is_random:
        study_name = f"fas_exon_random_{args.optimization}_{args.fitness_function}"
    else:
        study_name = f"fas_exon_{args.optimization}_{args.fitness_function}"

    if args.optimization == "tpe":
        TPESampler(
            study_name, 500, _input, args.fitness_function, args.is_random, _args
        )

    elif args.optimization == "nsgaii":
        NSGAIISampler(
            study_name, 500, _input, args.fitness_function, args.is_random, _args
        )


if __name__ == "__main__":
    main()
