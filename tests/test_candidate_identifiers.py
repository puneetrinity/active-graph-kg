"""Unit tests for identifier normalization rules.

These run without a database — they exercise the pure string rules that every
upstream identifier must pass through before it can be persisted.
"""

from __future__ import annotations

import pytest

from activekg.graph.candidate_identifiers import (
    IdentifierNormalizationError,
    normalize_identifier,
)


class TestEmail:
    def test_lowercases_and_strips(self):
        assert normalize_identifier("email", "  Alice@Example.COM ") == "alice@example.com"

    def test_strips_mailto_prefix(self):
        assert normalize_identifier("email", "mailto:bob@foo.io") == "bob@foo.io"

    def test_gmail_dots_folded(self):
        assert (
            normalize_identifier("email", "a.l.i.c.e@gmail.com")
            == "alice@gmail.com"
        )

    def test_gmail_plus_alias_preserved(self):
        assert (
            normalize_identifier("email", "alice+jobs@gmail.com")
            == "alice+jobs@gmail.com"
        )

    def test_googlemail_aliases_to_gmail(self):
        assert (
            normalize_identifier("email", "bob@googlemail.com")
            == "bob@gmail.com"
        )

    def test_invalid_rejected(self):
        with pytest.raises(IdentifierNormalizationError):
            normalize_identifier("email", "not-an-email")


class TestPhone:
    def test_strips_formatting(self):
        assert normalize_identifier("phone", "+1 (415) 555-0199") == "+14155550199"

    def test_double_zero_rewritten_to_plus(self):
        assert normalize_identifier("phone", "0044 20 7946 0018") == "+442079460018"

    def test_bare_digits_kept(self):
        assert normalize_identifier("phone", "4155550199") == "4155550199"

    def test_too_short_rejected(self):
        with pytest.raises(IdentifierNormalizationError):
            normalize_identifier("phone", "12345")

    def test_too_long_rejected(self):
        with pytest.raises(IdentifierNormalizationError):
            normalize_identifier("phone", "+1234567890123456")


class TestLinkedIn:
    def test_canonical(self):
        assert (
            normalize_identifier("linkedin_url", "https://www.linkedin.com/in/alice/")
            == "https://linkedin.com/in/alice"
        )

    def test_case_and_query_stripped(self):
        assert (
            normalize_identifier(
                "linkedin_url",
                "http://LinkedIn.com/in/Alice?trk=feed",
            )
            == "https://linkedin.com/in/alice"
        )

    def test_pub_path_accepted(self):
        assert (
            normalize_identifier("linkedin_url", "linkedin.com/pub/bob/1/2/3")
            == "https://linkedin.com/in/bob"
        )

    def test_company_page_rejected(self):
        with pytest.raises(IdentifierNormalizationError):
            normalize_identifier("linkedin_url", "https://linkedin.com/company/acme")

    def test_wrong_host_rejected(self):
        with pytest.raises(IdentifierNormalizationError):
            normalize_identifier("linkedin_url", "https://example.com/in/alice")


class TestGitHub:
    def test_canonical(self):
        assert (
            normalize_identifier("github_url", "https://github.com/Alice/")
            == "https://github.com/alice"
        )

    def test_strips_repo_path(self):
        assert (
            normalize_identifier("github_url", "https://github.com/alice/project")
            == "https://github.com/alice"
        )


class TestMedium:
    def test_at_handle(self):
        assert (
            normalize_identifier("medium_url", "https://medium.com/@Alice")
            == "https://medium.com/@alice"
        )

    def test_subdomain_handle(self):
        assert (
            normalize_identifier("medium_url", "https://alice.medium.com/some-post")
            == "https://medium.com/@alice"
        )

    def test_non_medium_rejected(self):
        with pytest.raises(IdentifierNormalizationError):
            normalize_identifier("medium_url", "https://example.com/@alice")


class TestOpaqueIds:
    @pytest.mark.parametrize(
        "itype",
        ["signal_candidate_id", "vantahire_application_id", "vantahire_resume_id"],
    )
    def test_preserved_verbatim(self, itype: str):
        assert normalize_identifier(itype, "  XYZ-123 ") == "XYZ-123"

    @pytest.mark.parametrize(
        "itype",
        ["signal_candidate_id", "vantahire_application_id", "vantahire_resume_id"],
    )
    def test_empty_rejected(self, itype: str):
        with pytest.raises(IdentifierNormalizationError):
            normalize_identifier(itype, "   ")


class TestTypeRegistry:
    def test_unknown_type_rejected(self):
        with pytest.raises(IdentifierNormalizationError):
            normalize_identifier("facebook_url", "https://facebook.com/alice")

    def test_non_string_rejected(self):
        with pytest.raises(IdentifierNormalizationError):
            normalize_identifier("email", 42)  # type: ignore[arg-type]
