import re

import pandas as pd


LEAKAGE_COLS = {
    "time_to_relapse",
    "Brain_relapses",
    "GEO_accession_number",
    "GEO_asscession_number",
}

CLINICAL_COLS = {"lymph_node_status", "ER_Status"}

TARGET_COL = "relapse"


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are not valid ARM items.

    Removes (in order):
    1. Affymetrix spike-in/hybridisation control probes (``AFFX-*``).
    2. Outcome-adjacent columns that create leakage risk
       (``time_to_relapse``, ``Brain_relapses``).
    3. Sample identifier (``GEO_accession_number`` / typo variant).

    Parameters
    ----------
    df : pd.DataFrame
        Merged expression + clinical dataset (samples as rows,
        features as columns).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with only expression probes, clinical
        covariates (``lymph_node_status``, ``ER_Status``), and the
        target (``relapse``).
    """
    result = df.copy()

    known_typos = {"GEO_asscession_number": "GEO_accession_number"}
    result = result.rename(columns=known_typos)

    affx_pattern = re.compile(r"^AFFX-", re.IGNORECASE)
    affx_cols = [c for c in result.columns if affx_pattern.match(str(c))]
    leakage_present = LEAKAGE_COLS & set(result.columns)

    if affx_cols:
        result = result.drop(columns=affx_cols)

    if leakage_present:
        result = result.drop(columns=list(leakage_present))

    return result
