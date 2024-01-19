import argparse
import os
from typing import List
import pandas as pd
import glob
import gzip
from datasetevaluation.dataset import Dataset
from datasetgeneration.archive import Archive


def load_archive_trackers(dress_output_directories: list) -> pd.DataFrame:
    """
    Reads archive trackers (if available) and returns additional
    information about the evolutionary process
    """
    archives_info = []
    tracking_header = "Run_id,Seed,Seq_id,Generation,Execution_time,Archive_size,Archive_diversity,\
Archive_empty_bin_ratio,Archive_diversity_per_bin,Archive_size_per_bin,Phenotype,Sequence,\
Splice_site_positions,Score,Delta_score".split(
        ","
    )
    print("Reading archive trackers to extract elapsed time and max generation ..")
    for single_evol in dress_output_directories:
        for file in glob.glob(single_evol + "/*archive_logger.csv.gz"):
            with gzip.open(file, "rb") as g:
                g.seek(-3000, os.SEEK_END)  # go 1000 bytes before end
                last_line = g.readlines()[-1].decode().rstrip().split(",")
                archives_info.append(
                    [
                        single_evol.split("/")[-1],
                        last_line[tracking_header.index("Seed")],
                        last_line[tracking_header.index("Generation")],
                        last_line[tracking_header.index("Execution_time")],
                    ]
                )
                g.close()

    df = pd.DataFrame(
        archives_info, columns=["Run_id", "Seed", "Generation", "Execution_time"]
    )
    df = df.astype(
        {"Run_id": str, "Seed": int, "Generation": int, "Execution_time": float}
    )
    return df


def load_archives(dress_output_directories: list, original_seq: str) -> pd.DataFrame:
    """
    Reads a list of dress output directories and returns a list of
    dataframes, each one representing evolutionary process across 5 seeds.
    """
    all_datasets = []
    print("Loading datasets ..")

    for single_evol in dress_output_directories:
        for file in glob.glob(single_evol + "/*dataset.csv.gz"):
            dataset = Dataset(original_seq, file)
            dataset.data["Run_id"] = single_evol.split("/")[-1]
            all_datasets.append(dataset)

    return all_datasets


def get_metrics(
    datasets: List[Dataset], archive_tracking_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Returns a dataframe with overall metrics for each run
    """

    _results = []
    for dataset in datasets:
        
        res = {"Run_id": dataset.data.iloc[0].Run_id, "Seed": dataset.data.iloc[0].Seed}
        res.update(dataset.metrics)
        res['Quality'] = dataset.quality
        _results.append(pd.DataFrame({k: [v] for k, v in res.items()}))

    results = pd.concat(_results)
    results["Size"] = results.Size.apply(lambda x: 1 if x >= 5000 else x / 5000)
    results["Low_count_bin_ratio"] = results.Low_count_bin_ratio.apply(lambda x: 1 - x)

    if archive_tracking_df is not None:
        results = pd.merge(
            results, archive_tracking_df, on=["Run_id", "Seed"], how="left"
        )
        results["Execution_time_normalized"] = results.Execution_time.apply(
            lambda x: 1 - min(x, 600) / 600
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Look at the output of several optuna best trials using different seeds"
    )
    parser.add_argument(
        dest="evolutions",
        help="File listing the directories (one per line) with generated datasets",
    )
    parser.add_argument(
        dest="original_seq",
        help="File with information about the original sequence",
    )
    parser.add_argument(
        "--check_tracking",
        action="store_true",
        help="If set, it also checks *archive_logger.csv.gz files to retrieve further "
        "info about the evolutionary run (generation achieved, execution time",
    )

    args = parser.parse_args()

    dress_output_directories = [line.rstrip("\n") for line in open(args.evolutions)]
    datasets = load_archives(dress_output_directories, args.original_seq)

    if args.check_tracking:
        archive_tracking_df = load_archive_trackers(dress_output_directories)
        stats = get_metrics(datasets, archive_tracking_df)
    else:
        stats = get_metrics(datasets, None)

    stats.sort_values(by="Quality", ascending=False).to_csv("2_final_stats.csv", index=False)
    grouped_df = (
        stats.groupby("Run_id")["Quality"]
        .agg(["mean", "std"])
        .sort_values(by="mean", ascending=False)
    )

    print(grouped_df)


if __name__ == "__main__":
    main()
