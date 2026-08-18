"""Tests for security limits, error counters, and Pydantic validation.

These tests can run without a database connection by setting ACTIVEKG_TEST_NO_DB=true.
For integration tests with real DB, unset the flag or set it to false.
"""

import os
from unittest.mock import patch

import pytest

# Enable test mode to avoid DB connection during import
os.environ["ACTIVEKG_TEST_NO_DB"] = "true"
os.environ["JWT_ENABLED"] = "false"  # Disable JWT for easier testing

from activekg.api.main import app, get_route_name, get_security_limits
from activekg.connectors.config_store import validate_connector_config


class TestSecurityLimitsEndpoint:
    """Test /_admin/security/limits endpoint."""

    def test_security_limits_default_config(self):
        """Test security limits with default configuration."""
        with patch.dict(os.environ, {"JWT_ENABLED": "false"}, clear=False):
            data = get_security_limits(None)

        assert data["external_payload_loading"] == {
            "enabled": False,
            "accepted_sources": ["inline_node_properties", "bounded_multipart_upload"],
        }
        assert data["ssrf_protection"] == {
            "enabled": False,
            "reason": "Remote payload-reference loading is unavailable.",
        }
        assert data["file_payload_ref_loading"] == {
            "enabled": False,
            "reason": "Local file payload-reference loading is unavailable.",
        }

        # Verify request limits
        assert "request_limits" in data
        assert data["request_limits"]["max_request_body_bytes"] == 10485760
        assert data["request_limits"]["max_request_body_mb"] == 10.0
        assert "Content-Length header" in data["request_limits"]["enforced_for"]
        assert "chunked transfers" in data["request_limits"]["enforced_for"]

    def test_security_limits_do_not_expose_stale_url_configuration(self):
        with patch.dict(
            os.environ,
            {
                "JWT_ENABLED": "false",
                "ACTIVEKG_URL_ALLOWLIST": "example.com,trusted-api.com",
            },
            clear=False,
        ):
            data = get_security_limits(None)

        assert "url_allowlist" not in data["ssrf_protection"]
        assert "example.com" not in str(data)
        assert "trusted-api.com" not in str(data)

    def test_security_limits_do_not_expose_stale_file_configuration(self):
        with patch.dict(
            os.environ,
            {
                "JWT_ENABLED": "false",
                "ACTIVEKG_FILE_BASEDIRS": "/opt/data,/mnt/uploads",
            },
            clear=False,
        ):
            data = get_security_limits(None)

        assert "file_access" not in data
        assert "/opt/data" not in str(data)
        assert "/mnt/uploads" not in str(data)


class TestRouteNameExtraction:
    """Test route name extraction for metrics."""

    def test_get_route_name_with_path_params(self):
        """Test that route name uses template, not actual values."""

        # Create a mock request with route info
        class MockRoute:
            path = "/nodes/{node_id}"

        class MockScope:
            def __getitem__(self, key):
                if key == "route":
                    return MockRoute()
                return {}

        class MockRequest:
            scope = {"route": MockRoute()}
            url = type("URL", (), {"path": "/nodes/abc-123"})()
            app = app

        request = MockRequest()
        route_name = get_route_name(request)

        # Should return template, not actual ID
        assert route_name == "/nodes/{node_id}"

    def test_get_route_name_fallback(self):
        """Test route name fallback to raw path when template not available."""

        class MockRequest:
            scope = {}
            url = type("URL", (), {"path": "/custom/path"})()
            app = app

        request = MockRequest()
        route_name = get_route_name(request)

        # Should fallback to raw path
        assert route_name == "/custom/path"


class TestPydanticValidation:
    """Test Pydantic validation for connector configs."""

    def test_s3_config_validation_valid(self):
        """Test valid S3 config passes validation."""
        config = {
            "bucket": "my-bucket",
            "prefix": "data/",
            "region": "us-west-2",
            "access_key_id": "AKIA1234567890ABCDEF",
            "secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        }

        validated = validate_connector_config("s3", config)

        assert validated["bucket"] == "my-bucket"
        assert validated["region"] == "us-west-2"
        assert validated["access_key_id"] == "AKIA1234567890ABCDEF"

    def test_s3_config_validation_invalid_access_key(self):
        """Test invalid S3 config (access key too short) raises error."""
        config = {
            "bucket": "my-bucket",
            "access_key_id": "SHORT",  # Too short (min 16 chars)
            "secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        }

        with pytest.raises(ValueError, match="Invalid s3 config"):
            validate_connector_config("s3", config)

    def test_s3_config_validation_invalid_secret_key(self):
        """Test invalid S3 config (secret key too short) raises error."""
        config = {
            "bucket": "my-bucket",
            "access_key_id": "AKIA1234567890ABCDEF",
            "secret_access_key": "TOO_SHORT",  # Too short (min 32 chars)
        }

        with pytest.raises(ValueError, match="Invalid s3 config"):
            validate_connector_config("s3", config)

    def test_gcs_config_validation_valid(self):
        """Test valid GCS config passes validation."""
        config = {
            "bucket": "my-gcs-bucket",
            "prefix": "data/",
            "service_account_json_path": "/path/to/service-account.json",
        }

        validated = validate_connector_config("gcs", config)

        assert validated["bucket"] == "my-gcs-bucket"
        assert validated["service_account_json_path"] == "/path/to/service-account.json"

    def test_drive_config_validation_valid(self):
        """Test valid Drive config passes validation."""
        config = {
            "folder_id": "1a2b3c4d5e6f",
            "service_account_json_path": "/path/to/service-account.json",
        }

        validated = validate_connector_config("drive", config)

        assert validated["folder_id"] == "1a2b3c4d5e6f"

    def test_unknown_provider_skips_validation(self):
        """Test unknown provider skips validation but returns config."""
        config = {"custom_field": "value"}

        validated = validate_connector_config("unknown_provider", config)

        # Should return config unchanged
        assert validated == config

    def test_poll_interval_validation(self):
        """Test poll_interval is validated (must be >= 60 and <= 3600)."""
        # Valid: within range
        config = {
            "bucket": "my-bucket",
            "access_key_id": "AKIA1234567890ABCDEF",
            "secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "poll_interval_seconds": 600,
        }
        validated = validate_connector_config("s3", config)
        assert validated["poll_interval_seconds"] == 600

        # Invalid: too small
        config_too_small = {
            "bucket": "my-bucket",
            "access_key_id": "AKIA1234567890ABCDEF",
            "secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "poll_interval_seconds": 30,  # Less than 60
        }
        with pytest.raises(ValueError):
            validate_connector_config("s3", config_too_small)

        # Invalid: too large
        config_too_large = {
            "bucket": "my-bucket",
            "access_key_id": "AKIA1234567890ABCDEF",
            "secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "poll_interval_seconds": 7200,  # More than 3600
        }
        with pytest.raises(ValueError):
            validate_connector_config("s3", config_too_large)


class TestErrorMetrics:
    """Test error metrics with route names and error types."""

    def test_error_metrics_track_route_template(self):
        """Test that error metrics use route templates, not raw paths."""
        from activekg.observability.metrics import api_errors_total, record_api_error

        # Record an error with route template
        before_count = api_errors_total.labels(
            endpoint="/nodes/{node_id}", status="404", error_type="not_found"
        )._value.get()

        record_api_error("/nodes/{node_id}", 404, "not_found")

        after_count = api_errors_total.labels(
            endpoint="/nodes/{node_id}", status="404", error_type="not_found"
        )._value.get()

        assert after_count == before_count + 1

    def test_error_metrics_with_error_types(self):
        """Test that error metrics track error types."""
        from activekg.observability.metrics import api_errors_total, record_api_error

        # Test different error types
        before_validation = api_errors_total.labels(
            endpoint="/nodes", status="422", error_type="validation_error"
        )._value.get()

        record_api_error("/nodes", 422, "validation_error")

        after_validation = api_errors_total.labels(
            endpoint="/nodes", status="422", error_type="validation_error"
        )._value.get()

        assert after_validation == before_validation + 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
