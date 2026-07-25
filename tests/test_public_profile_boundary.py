import pytest

from activekg.api.global_memory import (
    _merge_nonempty,
    _normalize_email,
    _validated_public_market,
    linkedin_id_from_url,
    sanitize_public_profile,
)
from activekg.graph.candidate_identifiers import (
    IdentifierNormalizationError,
    normalize_identifier,
)


def test_public_profile_allowlist_removes_contact_pii_at_any_depth() -> None:
    public = sanitize_public_profile(
        {
            "crustdata_person_id": 42,
            "basic_profile": {
                "name": "Public Person",
                "headline": "Backend Engineer; contact headline@example.com or +1-415-555-0123",
                "summary": ("Public bio; private@example.com; +1-415-555-0123"),
                "email": "private@example.com",
                "application": {"resume": "BASIC_APPLICATION_SENTINEL"},
                "campaign": {"id": "BASIC_CAMPAIGN_SENTINEL"},
                "nested": {"phone_number": "+1-555-private"},
            },
            "metadata": {
                "updated_at": "2026-07-25T00:00:00Z",
                "recruiter_notes": "METADATA_NOTE_SENTINEL",
                "resume": "METADATA_RESUME_SENTINEL",
            },
            "social_handles": {
                "professional_network_identifier": {
                    "profile_url": "https://linkedin.com/in/public-person",
                    "email": "linkedin-private@example.com",
                },
                "generic_identifier": {
                    "type": "email",
                    "value": "generic-private@example.com",
                },
            },
            "experience": {
                "employment_details": {
                    "current": [
                        {
                            "title": "Engineer",
                            "description": ("Contact work@example.com or 9876543210"),
                            "contact_information": {"work_email": "work@example.com"},
                            "private_payload": {"email": "nested-private@example.com"},
                        }
                    ]
                }
            },
            "contact": {"emails": ["provider@example.com"]},
            "emails": ["provider@example.com"],
            "resume": {"text": "PRIVATE_RESUME_SENTINEL"},
            "recruiter_notes": "PRIVATE_NOTE_SENTINEL",
        }
    )

    rendered = repr(public)
    assert public["crustdata_person_id"] == 42
    assert public["basic_profile"]["name"] == "Public Person"
    for sentinel in (
        "private@example.com",
        "headline@example.com",
        "+1-555-private",
        "work@example.com",
        "provider@example.com",
        "PRIVATE_RESUME_SENTINEL",
        "PRIVATE_NOTE_SENTINEL",
        "BASIC_APPLICATION_SENTINEL",
        "BASIC_CAMPAIGN_SENTINEL",
        "METADATA_NOTE_SENTINEL",
        "METADATA_RESUME_SENTINEL",
        "linkedin-private@example.com",
        "generic-private@example.com",
        "nested-private@example.com",
        "+1-415-555-0123",
        "9876543210",
    ):
        assert sentinel not in rendered
    assert "[redacted]" in rendered


def test_public_profile_allowlist_rejects_objects_inside_allowed_leaf_fields() -> None:
    public = sanitize_public_profile(
        {
            "crustdata_person_id": {"email": "ROOT_TYPE_CONFUSION"},
            "basic_profile": {
                "name": "Public Person",
                "summary": {"email": "SUMMARY_TYPE_CONFUSION"},
                "languages": [
                    "English",
                    {"email": "LANGUAGE_TYPE_CONFUSION"},
                ],
            },
            "professional_network": {
                "connections": {"email": "CONNECTION_TYPE_CONFUSION"},
                "open_to_cards": [
                    "open_to_work",
                    {"email": "CARD_TYPE_CONFUSION"},
                ],
            },
            "experience": {
                "employment_details": {
                    "current": [
                        {
                            "title": "Engineer",
                            "company_industries": [
                                "Software",
                                {"email": "INDUSTRY_TYPE_CONFUSION"},
                            ],
                        }
                    ]
                }
            },
            "skills": {
                "professional_network_skills": [
                    "Python",
                    {"email": "SKILL_TYPE_CONFUSION"},
                ]
            },
        }
    )

    assert "crustdata_person_id" not in public
    assert "summary" not in public["basic_profile"]
    assert public["basic_profile"]["languages"] == ["English"]
    assert "connections" not in public["professional_network"]
    assert public["professional_network"]["open_to_cards"] == ["open_to_work"]
    assert public["experience"]["employment_details"]["current"][0]["company_industries"] == [
        "Software"
    ]
    assert public["skills"]["professional_network_skills"] == ["Python"]
    assert "TYPE_CONFUSION" not in repr(public)


def test_partial_public_reingest_cannot_erase_richer_profile() -> None:
    existing = {
        "basic_profile": {
            "name": "Public Person",
            "headline": "Principal Backend Engineer",
            "location": {"city": "Bengaluru", "country": "India"},
        },
        "skills": {"professional_network_skills": ["Python", "PostgreSQL"]},
    }
    partial = sanitize_public_profile(
        {
            "basic_profile": {"headline": "", "location": {}},
            "skills": {"professional_network_skills": []},
        }
    )

    merged = _merge_nonempty(existing, partial)
    assert merged == existing


def test_googlemail_aliases_share_one_suppression_hash() -> None:
    gmail, gmail_hash = _normalize_email("First.Last@gmail.com")
    googlemail, googlemail_hash = _normalize_email("firstlast@googlemail.com")
    assert gmail == googlemail == "firstlast@gmail.com"
    assert gmail_hash == googlemail_hash


def test_public_identity_lookup_uses_the_canonical_linkedin_anchor() -> None:
    assert (
        normalize_identifier(
            "linkedin_url",
            "https://www.linkedin.com/in/Public-Person/?trk=private_context",
        )
        == "https://linkedin.com/in/public-person"
    )
    assert (
        normalize_identifier("linkedin_url", "https://uk.linkedin.com/pub/Public-Person/1/2/3")
        == "https://linkedin.com/in/public-person"
    )
    with pytest.raises(IdentifierNormalizationError):
        normalize_identifier("linkedin_url", "https://evillinkedin.com/in/public-person")
    assert (
        linkedin_id_from_url("https://uk.linkedin.com/pub/Public-Person/1/2/3") == "public-person"
    )
    assert linkedin_id_from_url("https://evillinkedin.com/in/public-person") is None


def test_public_market_key_must_match_its_canonical_dimensions() -> None:
    valid = {
        "version": 1,
        "coarse_market_key": "",
        "role_family": "backend",
        "location_city": "bangalore",
        "location_country_code": "IN",
        "seniority_band": "senior",
    }
    # Avoid a fixture that can drift from the documented JSON contract.
    material = {
        "version": 1,
        "roleFamily": "backend",
        "locationCity": "bangalore",
        "locationCountryCode": "IN",
        "seniorityBand": "senior",
    }
    import hashlib
    import json

    valid["coarse_market_key"] = (
        "public-market:v1:"
        + hashlib.sha256(json.dumps(material, separators=(",", ":")).encode("utf-8")).hexdigest()
    )
    assert _validated_public_market(valid) == {
        "coarse_market_key": valid["coarse_market_key"],
        "role_family": "backend",
        "location_city": "bangalore",
        "location_country_code": "IN",
        "seniority_band": "senior",
    }
    with pytest.raises(ValueError, match="does not match"):
        _validated_public_market({**valid, "coarse_market_key": "public-market:v1:wrong"})
