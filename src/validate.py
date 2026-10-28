"""
validate.py
Pandera schema validation for incoming application data. This is the
gate every batch passes through before it reaches preprocessing.py --
it checks column types, plausible value ranges, and missing-data
thresholds, and reports clear errors rather than letting bad data
silently reach the model.
"""

import pandera.pandas as pa
from pandera import Column, Check, DataFrameSchema

# Only the columns that matter most for data quality are checked
# explicitly; strict=False lets the other ~110 columns in the dataset
# pass through unchecked rather than requiring a schema entry for all
# 122 of them.
application_schema = DataFrameSchema(
    {
        "SK_ID_CURR": Column(int, Check.ge(0), nullable=False, unique=True),
        "AMT_INCOME_TOTAL": Column(float, Check.gt(0), nullable=False),
        "AMT_CREDIT": Column(float, Check.gt(0), nullable=False),
        "AMT_ANNUITY": Column(float, Check.gt(0), nullable=True),
        "DAYS_BIRTH": Column(int, Check.lt(0), nullable=False),
        # DAYS_EMPLOYED legitimately contains the 365243 placeholder for
        # "not employed" -- that gets fixed downstream in preprocessing.py,
        # so it isn't rejected here.
        "DAYS_EMPLOYED": Column(int, nullable=False),
        "CODE_GENDER": Column(str, Check.isin(["M", "F", "XNA"]), nullable=False),
        "NAME_CONTRACT_TYPE": Column(str, nullable=False),
        "CNT_CHILDREN": Column(int, Check.ge(0), nullable=False),
        "EXT_SOURCE_1": Column(float, Check.in_range(0, 1), nullable=True),
        "EXT_SOURCE_2": Column(float, Check.in_range(0, 1), nullable=True),
        "EXT_SOURCE_3": Column(float, Check.in_range(0, 1), nullable=True),
    },
    coerce=True,
    strict=False,
)

# Fail the batch if any single column is missing more than this fraction
# of its values -- catches a badly corrupted or truncated file.
MAX_NULL_RATIO = 0.65


def validate_batch(df, schema=application_schema):
    """
    Returns (is_valid, errors). Does not raise on its own -- the caller
    decides what to do with a failed batch (reject it, log it, halt the
    pipeline), matching the validation-failure behavior described in the
    report guideline.
    """
    errors = []

    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        errors.append(str(e.failure_cases))

    # Only check null ratios on the columns this schema actually cares
    # about -- the dataset has ~50 building/apartment columns that are
    # legitimately mostly missing for most applicants, and checking
    # those would reject perfectly normal data.
    checked_cols = [c for c in application_schema.columns if c in df.columns]
    null_ratios = df[checked_cols].isnull().mean()
    bad_cols = null_ratios[null_ratios > MAX_NULL_RATIO]
    if not bad_cols.empty:
        errors.append(f"Columns exceeding {MAX_NULL_RATIO:.0%} missing: {bad_cols.to_dict()}")

    is_valid = len(errors) == 0
    return is_valid, errors