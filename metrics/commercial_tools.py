import os

import numpy as np
import pandas
from sklearn import metrics

dict_metrics = {
    "auc": lambda label, score: metrics.roc_auc_score(label, score),
    "acc": lambda label, score: metrics.balanced_accuracy_score(label, score > 0),
}


def compute_metrics(input_csv, output_csv, metrics_fun):
    table = pandas.read_csv(output_csv)
    list_algs = [_ for _ in table.columns if _ != "filename"]
    table = pandas.read_csv(input_csv).merge(
        table,
        on=[
            "filename",
        ],
    )
    assert "typ" in table
    # Prepare helpers
    typ_str = table["typ"].astype(str)
    real_mask = typ_str.str.startswith("real")
    # Evaluate only synthetic types (exclude any "real@..." entries)
    list_typs = sorted([_ for _ in set(typ_str) if not str(_).startswith("real")])
    # Extract dataset suffix for matching (e.g., "*@ffhq" -> "ffhq")
    dataset_col = typ_str.apply(lambda s: s.split("@", 1)[1] if "@" in s else None)

    # Identify unique generators
    list_gens = sorted(list(set([t.split("@")[0] for t in list_typs])))

    tab_metrics = pandas.DataFrame(index=list_algs, columns=list_gens)
    tab_metrics.loc[:, :] = np.nan

    for gen in list_gens:
        # Find all types belonging to this generator
        gen_types = [t for t in list_typs if t.startswith(gen + "@")]

        # Build mask for this generator's data
        gen_mask = np.zeros(len(table), dtype=bool)

        for typ in gen_types:
            ds = typ.split("@", 1)[1] if "@" in typ else None
            # Add synthetic data
            gen_mask |= typ_str == typ
            # Add corresponding real data
            if ds:
                gen_mask |= real_mask & (dataset_col == ds)

        tab_gen = table[gen_mask]

        if len(tab_gen) == 0:
            continue

        for alg in list_algs:
            score = tab_gen[alg].values
            # Label is 1 for synthetic (not real), 0 for real
            label = ~tab_gen["typ"].astype(str).str.startswith("real")

            # Require both classes to be present
            if (label.sum() == 0) or (label.sum() == len(label)):
                continue
            if np.all(np.isfinite(score)) == False:
                continue

            tab_metrics.loc[alg, gen] = metrics_fun(label, score)

    tab_metrics.loc[:, "AVG"] = tab_metrics.mean(1)

    return tab_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_csv", "-i", type=str, help="The path of the input csv file with the list of images"
    )
    parser.add_argument(
        "--out_csv", "-o", type=str, help="The path of the output csv file", default="./results.csv"
    )
    parser.add_argument(
        "--metrics", "-w", type=str, help="type of metrics ('auc' or 'acc')", default="auc"
    )
    parser.add_argument(
        "--save_tab", "-t", type=str, help="The path of the metrics csv file", default=None
    )
    args = vars(parser.parse_args())

    tab_metrics = compute_metrics(args["in_csv"], args["out_csv"], dict_metrics[args["metrics"]])
    tab_metrics.index.name = args["metrics"]
    print(tab_metrics.to_string(float_format=lambda x: "%5.3f" % x))

    if args["save_tab"] is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args["save_tab"])), exist_ok=True)
        tab_metrics.to_csv(args["save_tab"])
