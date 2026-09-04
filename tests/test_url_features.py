"""Unit tests for the URL feature extractor (Objective O3).

Every test uses a URL whose correct answer is knowable by inspection, so a
failure means the extractor disagrees with the published feature definition
rather than with an opinion. Network-dependent features are exercised with
use_network=False so the suite runs offline and deterministically.
"""
import math
import pytest

from url_features import (extract, extract_extra, extract_raw_url_features,
                          DERIVABLE, LEXICAL, HOST_BASED, NOT_DERIVABLE,
                          RAW_URL_FEATURES, UCI_FEATURES, SHORTENERS)

OFF = dict(use_network=False)


# ---------------------------------------------------------------- structure --
def test_schema_is_complete_and_ordered():
    f = extract("https://example.com", **OFF)
    assert list(f.keys()) == UCI_FEATURES
    assert len(f) == 30


def test_partition_is_exhaustive_and_disjoint():
    assert len(DERIVABLE) == 13
    assert len(LEXICAL) == 9 and len(HOST_BASED) == 4
    assert set(LEXICAL) | set(HOST_BASED) == set(DERIVABLE)
    assert set(LEXICAL) & set(HOST_BASED) == set()
    assert len(NOT_DERIVABLE) == 17
    assert set(DERIVABLE) | set(NOT_DERIVABLE) == set(UCI_FEATURES)


def test_unobtainable_features_are_marked_unknown_not_guessed():
    """0 means indeterminate. Imputing a value would present a guess as evidence."""
    f = extract("https://example.com", **OFF)
    assert all(f[k] == 0 for k in NOT_DERIVABLE)


# ------------------------------------------------------------------ lexical --
@pytest.mark.parametrize("url,expected", [
    ("http://192.168.1.1/login", -1),
    ("http://0x7f.0x00.0x00.0x01/x", -1),
    ("https://www.bbc.co.uk/news", 1),
])
def test_ip_address_detection(url, expected):
    assert extract(url, **OFF)["having_ip_address"] == expected


@pytest.mark.parametrize("url,expected", [
    ("http://a.com", 1),                        # < 54 chars
    ("http://a.com/" + "x" * 50, 0),            # 54 to 75
    ("http://a.com/" + "x" * 100, -1),          # > 75
])
def test_url_length_bands(url, expected):
    assert extract(url, **OFF)["url_length"] == expected


def test_shortener_detection_covers_known_services():
    assert extract("https://bit.ly/abc", **OFF)["shortining_service"] == -1
    assert extract("https://tinyurl.com/abc", **OFF)["shortining_service"] == -1
    assert extract("https://www.gov.uk/abc", **OFF)["shortining_service"] == 1
    assert "bit.ly" in SHORTENERS


def test_at_symbol_hides_real_destination():
    assert extract("http://good.com@evil.tk/x", **OFF)["having_at_symbol"] == -1
    assert extract("http://good.com/x", **OFF)["having_at_symbol"] == 1


def test_prefix_suffix_only_fires_on_the_registered_domain():
    """A hyphen in a subdomain is not the same signal as one in the domain."""
    assert extract("http://pay-pal.com", **OFF)["prefix_suffix"] == -1
    assert extract("http://free-gift.example.com", **OFF)["prefix_suffix"] == 1


@pytest.mark.parametrize("url,expected", [
    ("http://example.com", 1),
    ("http://www.example.com", 1),          # www is not counted
    ("http://login.example.com", 0),
    ("http://a.b.c.example.com", -1),
])
def test_subdomain_depth(url, expected):
    assert extract(url, **OFF)["having_sub_domain"] == expected


def test_nonstandard_port_flagged_but_80_and_443_are_not():
    assert extract("http://example.com:8080/x", **OFF)["port"] == -1
    assert extract("http://example.com:80/x", **OFF)["port"] == 1
    assert extract("https://example.com:443/x", **OFF)["port"] == 1


def test_https_token_detects_the_word_inside_the_hostname():
    assert extract("http://https-paypal.tk/x", **OFF)["https_token"] == -1
    assert extract("https://paypal.com/x", **OFF)["https_token"] == 1


def test_scheme_is_added_when_missing():
    """A user pasting 'example.com' should not crash the extractor."""
    assert extract("example.com", **OFF)["having_ip_address"] == 1


# -------------------------------------------------------------------- extra --
def test_extra_features_count_and_values():
    e = extract_extra("https://a-b.example.com/x/y?q=1")
    assert len(e) == 9
    assert e["n_hyphens"] == 1
    assert e["path_depth"] == 2
    assert e["has_query"] == 1
    assert 0 <= e["digit_ratio"] <= 1
    assert e["host_entropy"] > 0


def test_objective_o3_feature_count_is_met():
    """O3 required at least 15 engineered features."""
    total = len(DERIVABLE) + len(extract_extra("https://example.com"))
    assert total == 22 and total >= 15


# ------------------------------------------------------------------ robust --
@pytest.mark.parametrize("bad", ["", "   ", "not a url", "http://", "://x", "ftp://a.b/c"])
def test_malformed_input_does_not_raise(bad):
    f = extract(bad, **OFF)
    assert len(f) == 30
    assert all(v in (-1, 0, 1) for v in f.values())


def test_all_values_are_in_the_uci_coding():
    for u in ["https://www.google.com", "http://1.2.3.4@x-y.tk:99/a//b?c=1"]:
        assert all(v in (-1, 0, 1) for v in extract(u, **OFF).values())


def test_production_raw_url_schema_is_network_free_and_locked():
    features = extract_raw_url_features("http://192.0.2.10:8080/login?account=123%2Fabc")
    assert list(features) == RAW_URL_FEATURES
    assert len(features) == 26
    assert features["has_ip_host"] == 1
    assert features["has_nonstandard_port"] == 1
    assert features["has_login_token"] == 1
    assert features["has_percent_encoding"] == 1
