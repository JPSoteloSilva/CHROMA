import numpy as np
import pandas
from sklearn import metrics

dict_metrics = {
    "auc": lambda label, score: metrics.roc_auc_score(label, score),
    "acc": lambda label, score: metrics.balanced_accuracy_score(label, score > 0),
}


def compute_metrics(input_csv, table, metrics_fun):
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

    tab_metrics = pandas.DataFrame(index=list_algs, columns=list_typs)
    tab_metrics.loc[:, :] = np.nan
    for typ in list_typs:
        # Compare current synthetic type vs matching "real@<dataset>" entries only
        ds = typ.split("@", 1)[1] if "@" in typ else None
        neg_mask = real_mask & (dataset_col == ds)
        pos_mask = typ_str == typ
        tab_typ = table[neg_mask | pos_mask]
        for alg in list_algs:
            score = tab_typ[alg].values
            # Labels: positives are rows of current synthetic type; negatives are matching
            # real@<dataset>
            label = tab_typ["typ"].values == typ
            # Require both classes to be present
            if (label.sum() == 0) or (label.sum() == len(label)):
                continue
            if not np.all(np.isfinite(score)):
                continue

            tab_metrics.loc[alg, typ] = metrics_fun(label, score)
    tab_metrics.loc[:, "AVG"] = tab_metrics.mean(1)

    return tab_metrics
