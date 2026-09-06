"""TLS verify stays on. CA bundles / Windows ROOT are trusted; verify=False is not default."""

from __future__ import annotations

import sys
from pathlib import Path

from scientist_lab.llm.http_transport import resolve_tls_verify


def test_default_tls_verify_is_not_disabled() -> None:
    verify, source = resolve_tls_verify(verify_tls=True, environ={})
    assert verify is not False
    assert source in {"default", "windows_root_store"}


def test_llm_ca_bundle_wins(tmp_path: Path) -> None:
    bundle = tmp_path / "corp-ca.pem"
    bundle.write_text(
        "-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----\n",
        encoding="utf-8",
    )
    other = tmp_path / "ssl-cert.pem"
    other.write_text(
        "-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----\n",
        encoding="utf-8",
    )
    verify, source = resolve_tls_verify(
        verify_tls=True,
        environ={
            "LLM_CA_BUNDLE": str(bundle),
            "SSL_CERT_FILE": str(other),
            "REQUESTS_CA_BUNDLE": str(other),
        },
    )
    assert verify == str(bundle)
    assert source == "LLM_CA_BUNDLE"


def test_requests_ca_bundle_is_used_when_no_windows_root(tmp_path: Path) -> None:
    bundle = tmp_path / "corp-ca.pem"
    bundle.write_text(
        "-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----\n",
        encoding="utf-8",
    )
    verify, source = resolve_tls_verify(
        verify_tls=True,
        environ={"REQUESTS_CA_BUNDLE": str(bundle)},
    )
    assert verify is not False
    if sys.platform == "win32" and source == "windows_root_store":
        assert verify != str(bundle)
        return
    assert verify == str(bundle)
    assert source == "REQUESTS_CA_BUNDLE"


def test_ssl_cert_file_used_when_no_windows_root(tmp_path: Path) -> None:
    bundle = tmp_path / "ssl-cert.pem"
    bundle.write_text(
        "-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----\n",
        encoding="utf-8",
    )
    verify, source = resolve_tls_verify(
        verify_tls=True,
        environ={
            "SSL_CERT_FILE": str(bundle),
            "REQUESTS_CA_BUNDLE": str(tmp_path / "missing.pem"),
        },
    )
    assert verify is not False
    if sys.platform == "win32" and source == "windows_root_store":
        assert verify != str(bundle)
        return
    assert verify == str(bundle)
    assert source == "SSL_CERT_FILE"


def test_conda_ssl_cert_file_does_not_hide_windows_root(tmp_path: Path) -> None:
    bundle = tmp_path / "cacert.pem"
    bundle.write_text(
        "-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----\n",
        encoding="utf-8",
    )
    verify, source = resolve_tls_verify(
        verify_tls=True,
        environ={"SSL_CERT_FILE": str(bundle)},
    )
    assert verify is not False
    if sys.platform == "win32":
        assert source == "windows_root_store"
        assert verify != str(bundle)
    else:
        assert source in {"SSL_CERT_FILE", "default"}


def test_explicit_verify_tls_false_is_opt_in_only() -> None:
    verify, source = resolve_tls_verify(verify_tls=False, environ={})
    assert verify is False
    assert source == "verify_tls_disabled"


def test_missing_bundle_path_does_not_disable_tls(tmp_path: Path) -> None:
    verify, source = resolve_tls_verify(
        verify_tls=True,
        environ={"REQUESTS_CA_BUNDLE": str(tmp_path / "nope.pem")},
    )
    assert verify is not False
    assert source in {"default", "windows_root_store"}
