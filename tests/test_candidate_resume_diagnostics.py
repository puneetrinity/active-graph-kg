import pytest

from activekg.candidate_index.contracts import IndexContractError
from activekg.candidate_index.diagnostics import BOUNDS, ENUMS, diagnostic_payload


def test_no_text_or_unknown_keys_or_coercion():
    for value in (
        {"resume": "secret"},
        {"reason": "raw provider error"},
        {"completion_tokens": True},
        {"completion_tokens": "12"},
        {"finish_reason": "made_up"},
    ):
        with pytest.raises(IndexContractError):
            diagnostic_payload(value)


def test_unknown_not_zero():
    assert diagnostic_payload({"completion_tokens": None}) == {"version": 1}
    assert diagnostic_payload({"completion_tokens": 0})["completion_tokens"] == 0


@pytest.mark.parametrize("key,maximum", BOUNDS.items())
def test_numeric_boundaries(key, maximum):
    assert diagnostic_payload({key: maximum})[key] == maximum
    for invalid in (-1, maximum + 1, 1.2):
        with pytest.raises(IndexContractError):
            diagnostic_payload({key: invalid})


@pytest.mark.parametrize("key,values", ENUMS.items())
def test_enum_inventory(key, values):
    for value in values:
        assert diagnostic_payload({key: value})[key] == value
