#!/usr/bin/env python3
"""
Breast Cancer Gene Expression Data Exploration and Preprocessing

Dataset: GSE2034 (Affymetrix HG-U133A microarray)
Shape: 286 samples × 22,289 columns

Per AGENTS.md:
- Drop AFFX control probes (spike-in/housekeeping controls)
- Drop GEO_accession_number (sample ID, not a feature)
- Drop outcome/leakage columns: relapse (target), time_to_relapse, Brain_relapses
- Preserve clinical covariates: lymph_node_status, ER_Status (categorical, NOT median-split)
- Preserve gene expression probes (~22,215) for feature selection + discretization
"""

# ============================================================
# Configuration
# ============================================================

# Base directory: ../../data relative to this script
BASE_DIR = Path(__file__).resolve().parent.parent.parent / "data"

DATA_PATH = BASE_DIR / "GSE2034_series_matrix.txt"
DATA_PATH_CL = BASE_DIR / "GSE2034_clinical.txt"

DATA_PATH_SAVE = BASE_DIR / "final"


# ============================================================
# Helper functions
# ============================================================

def find_line_start(filepath, prefix):
    """
    Find the zero-based line number where a line starts
    with the specified prefix.
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if line.startswith(prefix):
                return i

    return None


# ============================================================
# Main workflow
# ============================================================

def main():

    # --------------------------------------------------------
    # Check input files
    # --------------------------------------------------------

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Expression data not found:\n{DATA_PATH}"
        )

    if not DATA_PATH_CL.exists():
        raise FileNotFoundError(
            f"Clinical data not found:\n{DATA_PATH_CL}"
        )

    print("=" * 60)
    print("GSE2034 DATA PROCESSING")
    print("=" * 60)

    print(f"\nExpression file: {DATA_PATH}")
    print(f"Clinical file:   {DATA_PATH_CL}")


    # --------------------------------------------------------
    # 1. Preview expression data metadata
    # --------------------------------------------------------

    print("\n[1] Expression dataset preview")
    print("-" * 60)

    with open(DATA_PATH, "r", encoding="utf-8", errors="replace") as f:
        for _ in range(20):
            line = f.readline()

            if not line:
                break

            print(line.strip())


    # --------------------------------------------------------
    # 2. Find expression matrix start
    # --------------------------------------------------------

    expression_start = find_line_start(
        DATA_PATH,
        '"ID_REF"'
    )

    if expression_start is None:
        raise ValueError(
            'Could not find expression matrix header starting with "ID_REF".'
        )

    print(
        f"\nExpression matrix starts at line "
        f"{expression_start}"
    )


    # --------------------------------------------------------
    # 3. Load expression matrix
    # --------------------------------------------------------

    print("\n[2] Loading expression matrix")
    print("-" * 60)

    dfe = pd.read_csv(
        DATA_PATH,
        sep="\t",
        skiprows=expression_start
    )

    print("Original expression shape:", dfe.shape)

    print("\nFirst five rows:")
    print(dfe.iloc[:5, :6])


    # --------------------------------------------------------
    # 4. Set ID_REF as index
    # --------------------------------------------------------

    dfe = dfe.set_index("ID_REF")

    print("\nExpression data after setting ID_REF as index:")
    print(dfe.head())


    # --------------------------------------------------------
    # 5. Load clinical data
    # --------------------------------------------------------

    print("\n[3] Loading clinical data")
    print("-" * 60)

    with open(
        DATA_PATH_CL,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as f:

        for _ in range(20):
            line = f.readline()

            if not line:
                break

            print(line.strip())


    # --------------------------------------------------------
    # 6. Find clinical data header
    # --------------------------------------------------------

    clinical_start = find_line_start(
        DATA_PATH_CL,
        "PID"
    )

    if clinical_start is None:
        raise ValueError(
            "Could not find clinical data header starting with PID."
        )

    print(
        f"\nClinical data starts at line "
        f"{clinical_start}"
    )


    # --------------------------------------------------------
    # 7. Load clinical data
    # --------------------------------------------------------

    dfc = pd.read_csv(
        DATA_PATH_CL,
        sep="\t",
        skiprows=clinical_start
    )

    print("\nClinical data shape:", dfc.shape)

    print("\nFirst five rows:")
    print(dfc.head())


    # --------------------------------------------------------
    # 8. Fix column name typo
    # --------------------------------------------------------

    if "GEO_asscession_number" in dfc.columns:

        dfc = dfc.rename(
            columns={
                "GEO_asscession_number":
                "GEO_accession_number"
            }
        )


    # --------------------------------------------------------
    # 9. Remove PID
    # --------------------------------------------------------

    if "PID" in dfc.columns:
        dfc = dfc.drop("PID", axis=1)


    print("\nClinical data after cleaning:")
    print(dfc.head())


    # --------------------------------------------------------
    # 10. Remove unwanted expression column
    # --------------------------------------------------------

    if "!series_matrix_table_end" in dfe.columns:

        dfe = dfe.drop(
            ["!series_matrix_table_end"],
            axis=1
        )


    # --------------------------------------------------------
    # 11. Check expression and clinical data
    # --------------------------------------------------------

    print("\n[4] Dataset validation")
    print("-" * 60)

    print(
        "Missing count expression:",
        dfe.isnull().sum().sum()
    )

    print(
        "Missing count clinical:",
        dfc.isnull().sum().sum()
    )

    print(
        "Expression matrix:",
        dfe.shape
    )

    print(
        "Clinical data:",
        dfc.shape
    )


    # --------------------------------------------------------
    # 12. Check patient/sample IDs
    # --------------------------------------------------------

    dfe_patients = set(dfe.columns)

    dfc_patients = set(
        dfc["GEO_accession_number"]
    )

    print("\nPatient/sample ID comparison:")

    print(
        "Expression samples:",
        len(dfe_patients)
    )

    print(
        "Clinical samples:",
        len(dfc_patients)
    )

    print(
        "Common samples:",
        len(
            dfe_patients &
            dfc_patients
        )
    )


    # --------------------------------------------------------
    # 13. Transpose expression matrix
    # --------------------------------------------------------

    print("\n[5] Transposing expression matrix")
    print("-" * 60)

    dfe.index.name = "GEO_accession_number"

    dfe = dfe.T

    print(
        "Expression matrix after transpose:",
        dfe.shape
    )

    print(dfe.head())


    # --------------------------------------------------------
    # 14. Merge expression and clinical data
    # --------------------------------------------------------

    print("\n[6] Merging datasets")
    print("-" * 60)

    data = dfe.merge(
        dfc,
        left_index=True,
        right_on="GEO_accession_number"
    )

    print(
        "Final merged data shape:",
        data.shape
    )

    print("\nFinal dataset:")
    print(data.head())


    # --------------------------------------------------------
    # 15. Save processed data
    # --------------------------------------------------------

    print("\n[7] Saving processed data")
    print("-" * 60)

    DATA_PATH_SAVE.mkdir(
        parents=True,
        exist_ok=True
    )

    expression_output = (
        DATA_PATH_SAVE /
        "expression_matrix.csv"
    )

    clinical_output = (
        DATA_PATH_SAVE /
        "clinical_data.csv"
    )

    final_output = (
        DATA_PATH_SAVE /
        "final_data.csv"
    )


    dfe.to_csv(
        expression_output,
        index=False
    )

    dfc.to_csv(
        clinical_output,
        index=False
    )

    data.to_csv(
        final_output,
        index=False
    )


    # --------------------------------------------------------
    # 16. Completion summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)

    print(
        f"\nExpression matrix:\n"
        f"  {expression_output}"
    )

    print(
        f"\nClinical data:\n"
        f"  {clinical_output}"
    )

    print(
        f"\nFinal merged data:\n"
        f"  {final_output}"
    )


if __name__ == "__main__":
    main()
