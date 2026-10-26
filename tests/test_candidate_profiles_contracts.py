import json

import pytest

from activekg.candidate_profiles.contracts import parse_command, receipt

RAW = b'{"schemaVersion":1,"commandId":"11111111-1111-4111-8111-111111111111","subjectId":"22222222-2222-4222-8222-222222222222","bindingId":"33333333-3333-4333-8333-333333333333","sequence":1,"action":"ensure_subject"}'


def test_cross_language_hash_and_receipt():
    command, digest = parse_command(RAW)
    assert digest == "a8c757f9ce898180424e867102eaee22dcc3339a99770ca89588b4cae5b2fa65"
    result = receipt(command, digest)
    assert result["publicationActive"] is False
    assert result["profileVersion"] == result["preferenceVersion"] == 0
    assert "action" not in result


@pytest.mark.parametrize(
    "raw",
    [
        b" " + RAW,
        RAW.replace(b'"sequence":1', b'"sequence":"1"'),
        RAW.replace(b'"sequence":1', b'"sequence":true'),
        RAW.replace(b'"sequence":1', b'"sequence":1,"sequence":1'),
        RAW.replace(b'"sequence":1', b'"sequence":1,"email":"no"'),
        RAW.replace(b'"sequence":1', b'"sequence":9007199254740992'),
        RAW.replace(b'"sequence":1', b'"sequence":0'),
        RAW.replace(b'"schemaVersion":1', b'"schemaVersion":true'),
        RAW.replace(b'"ensure_subject"', b'"publish"'),
        json.dumps(json.loads(RAW), sort_keys=True, separators=(",", ":")).encode(),
        b"[" * 4097,
        b'"\xff"',
        b"null",
    ],
)
def test_rejects_noncanonical_or_invalid_input(raw):
    with pytest.raises((ValueError, UnicodeError)):
        parse_command(raw)
