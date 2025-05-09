import polars as pl
import numpy as np
from jaxtyping import Float, Array
from os import getenv

ACCESS_TOKEN = getenv("HF_TOKEN")


def discretize_dataframe(df: pl.DataFrame, n_bins: int = 20):
    schema = make_schema(df)
    categorical_df = df.select(schema["types"]["categorical"])
    numerical_df = df.select(schema["types"]["numerical"]).with_columns(
        pl.all()
        .qcut(quantiles=n_bins, labels=[str(i) for i in range(n_bins)])
        .name.keep()
    )

    categorical_idxs = np.concatenate(
        [
            idx * np.ones(len(schema["var_metadata"][col]["levels"]))
            for idx, col in enumerate(categorical_df.columns)
        ]
        + [
            len(categorical_df.columns) + idx * np.ones(n_bins)
            for idx, col in enumerate(numerical_df.columns)
        ]
    ).astype(np.int32)

    df = pl.concat((categorical_df, numerical_df), how="horizontal")
    return schema, df, categorical_idxs


def to_dummies(df: pl.DataFrame):
    return df.to_dummies().select(pl.exclude("^.*_null$"))


def dummies_to_padded_array(df: pl.DataFrame, categorical_idxs: np.ndarray):
    data = df.to_numpy().astype(np.bool_)
    n_categories = max(categorical_idxs) + 1
    max_categories = max(sum(categorical_idxs == i) for i in range(n_categories))
    col_names = []

    padded_data = np.zeros(
        (data.shape[0], n_categories, max_categories), dtype=np.bool_
    )
    for i in range(n_categories):
        cat_length = sum(categorical_idxs == i)
        cat_start = np.where(categorical_idxs == i)[0][0]
        padded_data[:, i, :cat_length] = data[:, cat_start : cat_start + cat_length]
        col_names.append(df.columns[cat_start].rsplit("_", maxsplit=1)[0])

    return padded_data, col_names


def load_huggingface(dataset_path):
    splits = {
        "train": f"{dataset_path}/data-train-imputed-num.parquet",
        "test": f"{dataset_path}/data-test-full-num.parquet",
    }
    train_df = pl.read_parquet(
        f"hf://datasets/Large-Population-Model/model-building-evaluation/{splits['train']}",
        storage_options={"token": ACCESS_TOKEN},
    )
    test_df = pl.read_parquet(
        f"hf://datasets/Large-Population-Model/model-building-evaluation/{splits['test']}",
        storage_options={"token": ACCESS_TOKEN},
    )

    return train_df, test_df


def make_schema(df: pl.DataFrame):
    schema = {"types": {"numerical": [], "categorical": []}, "var_metadata": {}}
    for c in df.columns:
        if df[c].dtype == pl.Utf8:
            schema["types"]["categorical"].append(c)
            schema["var_metadata"][c] = {
                "levels": df[c].drop_nulls().unique().sort().to_list()
            }
        elif df[c].dtype == pl.Float64:
            schema["types"]["numerical"].append(c)
        else:
            raise ValueError(c)
    return schema


def from_dummies(df, separator="_"):
    col_exprs = {}

    for col in df.columns:
        name, value = col.rsplit(separator, maxsplit=1)
        expr = pl.when(pl.col(col) == 1).then(pl.lit(value))
        col_exprs.setdefault(name, []).append(expr)

    return df.select(
        pl.coalesce(exprs).alias(  # keep the first non-null expression value by row
            name
        )
        for name, exprs in col_exprs.items()
    )


def load_data(dataset_path):
    train_df, test_df = load_huggingface(dataset_path)
    df = pl.concat((train_df, test_df), how="vertical")
    schema, discretized_df, categorical_idxs = discretize_dataframe(df)
    dummies_df = to_dummies(discretized_df)
    bool_data, col_names = dummies_to_padded_array(dummies_df, categorical_idxs)

    bool_data = np.where(np.any(bool_data, axis=-1)[..., None], bool_data, True)
    data: Float[Array, "batch_size n_inputs input_dim"] = np.where(
        bool_data, 0, -np.inf
    )

    train_data = np.array(data[: len(train_df)])
    test_data = np.array(data[len(train_df) :])
    return train_data, test_data, col_names
