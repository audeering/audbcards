import pandas as pd
import pytest

import audformat

import audbcards


@pytest.mark.parametrize(
    "scheme_names, scheme_dtypes, labels, expected",
    [
        (
            ["emotion", "age", "gender", "language", "speaker"],
            ["str", "int", "str", "str", "int"],
            [
                ["happy", "sad"],
                None,
                ["female", "male"],
                ["DE", "EN"],
                "speaker",
            ],
            "emotion: [happy, sad], speaker: [age, gender, language], "
            "age, gender, language",
        ),
        (
            ["emotion", "age", "gender", "speaker", "audio_quality"],
            ["str", "int", "str", "int", "str"],
            [["happy", "sad"], None, ["female", "male"], "speaker", ["good", "bad"]],
            "emotion: [happy, sad], speaker: [age, gender], "
            "age, audio_quality, gender",
        ),
    ],
)
def test_format_schemes(scheme_names, scheme_dtypes, labels, expected):
    # Init database to contain schemes
    db = audformat.Database(name="db")
    for i, scheme_name in enumerate(scheme_names):
        # Create actual schemes
        if scheme_name == "speaker":
            db["speaker"] = audformat.MiscTable(
                pd.Index([0], dtype="Int8", name="speaker")
            )
            if "age" in scheme_names:
                db["speaker"]["age"] = audformat.Column(scheme_id="age")
            if "gender" in scheme_names:
                db["speaker"]["gender"] = audformat.Column(scheme_id="gender")
            if "language" in scheme_names:
                db["speaker"]["language"] = audformat.Column(scheme_id="language")
        db.schemes[scheme_name] = audformat.Scheme(
            dtype=scheme_dtypes[i],
            labels=labels[i],
        )
    # Generate scheme str with format_scheme()
    scheme_str = audbcards.core.utils.format_schemes(db.schemes)
    assert scheme_str == expected


@pytest.mark.parametrize("sample", [["a", "b", "c", "d", "e"]])
@pytest.mark.parametrize(
    "limit, replacement_text, expected",
    [
        (2, "...", ["a", "...", "e"]),
        (2, "###", ["a", "###", "e"]),
        (4, "...", ["a", "b", "...", "d", "e"]),
    ],
)
def test_limit_presented_samples(sample, limit, replacement_text, expected):
    limited_sample = audbcards.core.utils.limit_presented_samples(
        sample, limit, replacement_text
    )
    assert limited_sample == expected
