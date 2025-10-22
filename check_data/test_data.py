import scipy.stats
import pandas as pd
import numpy as np


def _clean_series(s: pd.Series) -> pd.Series:
    # Convert to numeric (coerce bad strings), drop NaN/inf
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s



def test_column_presence_and_type(data):

    # Disregard the reference dataset
    _, data = data

    required_columns = {
        "time_signature": pd.api.types.is_integer_dtype,
        "key": pd.api.types.is_integer_dtype,
        "danceability": pd.api.types.is_float_dtype,
        "energy": pd.api.types.is_float_dtype,
        "loudness": pd.api.types.is_float_dtype,
        "speechiness": pd.api.types.is_float_dtype,
        "acousticness": pd.api.types.is_float_dtype,
        "instrumentalness": pd.api.types.is_float_dtype,
        "liveness": pd.api.types.is_float_dtype,
        "valence": pd.api.types.is_float_dtype,
        "tempo": pd.api.types.is_float_dtype,
        "duration_ms": pd.api.types.is_integer_dtype,  # This is integer, not float as one might expect
        "text_feature": pd.api.types.is_string_dtype,
        "genre": pd.api.types.is_string_dtype
    }

    # Check column presence
    assert set(data.columns.values).issuperset(set(required_columns.keys()))

    for col_name, format_verification_funct in required_columns.items():

        assert format_verification_funct(data[col_name]), f"Column {col_name} failed test {format_verification_funct}"


def test_class_names(data):

    # Disregard the reference dataset
    _, data = data

    # Check that only the known classes are present
    known_classes = [
        "Dark Trap",
        "Underground Rap",
        "Trap Metal",
        "Emo",
        "Rap",
        "RnB",
        "Pop",
        "Hiphop",
        "techhouse",
        "techno",
        "trance",
        "psytrance",
        "trap",
        "dnb",
        "hardstyle",
    ]

    assert data["genre"].isin(known_classes).all()


def test_column_ranges(data):

    # Disregard the reference dataset
    _, data = data

    ranges = {
        "time_signature": (1, 5),
        "key": (0, 11),
        "danceability": (0, 1),
        "energy": (0, 1),
        "loudness": (-35, 5),
        "speechiness": (0, 1),
        "acousticness": (0, 1),
        "instrumentalness": (0, 1),
        "liveness": (0, 1),
        "valence": (0, 1),
        "tempo": (50, 250),
        "duration_ms": (20000, 1000000),
    }

    for col_name, (minimum, maximum) in ranges.items():

        assert data[col_name].dropna().between(minimum, maximum).all(), (
            f"Column {col_name} failed the test. Should be between {minimum} and {maximum}, "
            f"instead min={data[col_name].min()} and max={data[col_name].max()}"
        )


def test_kolmogorov_smirnov(data, ks_alpha):

    sample1, sample2 = data

    columns = [
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms"
    ]

    # Bonferroni correction for multiple hypothesis testing
    # (see my blog post on this topic to see where this comes from:
    # https://towardsdatascience.com/precision-and-recall-trade-off-and-multiple-hypothesis-testing-family-wise-error-rate-vs-false-71a85057ca2b)
    alpha_prime = 1 - (1 - ks_alpha)**(1 / len(columns))

    for col in columns:

        s1 = _clean_series(sample1[col])
        s2 = _clean_series(sample2[col])

        # Guard against empty after cleaning (skip or fail with message)
        assert len(s1) > 0 and len(s2) > 0, f"Empty series after cleaning for column: {col}"


        #ts, p_value = scipy.stats.ks_2samp(sample1[col], sample2[col])
        ts, p_value = scipy.stats.ks_2samp(s1[col], s2[col])
        
        # Fail if p_value is NaN (shouldn't happen after cleaning)
        assert not np.isnan(p_value), f"NaN p-value for column: {col}"
    
        # NOTE: as always, the p-value should be interpreted as the probability of
        # obtaining a test statistic (TS) equal or more extreme that the one we got
        # by chance, when the null hypothesis is true. If this probability is not
        # large enough, this dataset should be looked at carefully, hence we fail
        assert p_value > alpha_prime
