from activekg.embedding.global_candidates import (
    EMBED_VERSION,
    PUBLIC_EMBED_VERSION,
    build_candidate_embedding_text,
    build_public_candidate_embedding_text,
)


def test_embedding_text_excludes_structured_person_name_fields() -> None:
    text = build_candidate_embedding_text(
        {
            "name": "CANONICAL_PERSON_NAME",
            "headline": None,
            "role_family": "backend",
            "seniority_band": "senior",
            "skills_normalized": ["Python", "PostgreSQL"],
            "location_city": "Bengaluru",
            "location_country_code": "IN",
            "crustdata_profile": {
                "name": "BLOB_PERSON_NAME",
                "professional_network_name": "ROOT_NETWORK_PERSON_NAME",
                "basic_profile": {
                    "name": "BASIC_PROFILE_PERSON_NAME",
                    "professional_network_name": "BASIC_NETWORK_PERSON_NAME",
                    "headline": "Backend platform specialist",
                },
                "experience": {
                    "employment_details": {
                        "current": [{"title": "Principal Engineer", "name": "Acme Cloud"}],
                    },
                },
                "education": {
                    "schools": [{"degree": "B.Tech", "school": "Example Institute"}],
                },
            },
        }
    )

    for name_field_value in (
        "CANONICAL_PERSON_NAME",
        "BLOB_PERSON_NAME",
        "ROOT_NETWORK_PERSON_NAME",
        "BASIC_PROFILE_PERSON_NAME",
        "BASIC_NETWORK_PERSON_NAME",
    ):
        assert name_field_value not in text

    assert "Backend platform specialist" in text
    assert "Principal Engineer at Acme Cloud" in text
    assert "education: B.Tech, Example Institute" in text
    assert "skills: Python, PostgreSQL" in text
    assert "Bengaluru, IN" in text


def test_name_only_profile_is_empty_and_versioned_for_resweep() -> None:
    assert EMBED_VERSION == 3
    assert build_candidate_embedding_text({"name": "NAME_ONLY_PROFILE"}) == ""


def test_public_embedding_uses_only_the_public_projection() -> None:
    text = build_public_candidate_embedding_text(
        {
            "headline": "PRIVATE_APPLICANT_HEADLINE",
            "skills_normalized": ["PRIVATE_APPLICANT_SKILL"],
            "public_headline": "Public backend engineer",
            "public_skills_normalized": ["Python"],
            "public_location_city": "Bengaluru",
            "public_location_country_code": "IN",
            "public_profile": {
                "basic_profile": {"headline": "Public backend engineer"},
                "experience": {"employment_details": {"current": [{"title": "Engineer"}]}},
            },
        }
    )

    assert "PRIVATE_APPLICANT_HEADLINE" not in text
    assert "PRIVATE_APPLICANT_SKILL" not in text
    assert "Public backend engineer" in text
    assert "skills: Python" in text
    assert PUBLIC_EMBED_VERSION == 1
