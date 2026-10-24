"""Parity tests: the Rust extension must match the pure-Python reference."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import arabic_norm as rs
from arabic_norm import reference as py

FLAGS = (
    "strip_tashkeel",
    "strip_tatweel",
    "normalize_alif",
    "normalize_ya",
    "ta_marbuta",
    "remove_punct",
)
TOK_FLAGS = FLAGS[:-1]

SAMPLES = [
    "",
    "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
    "إِمَامٌ",
    "أَحْمَدُ وَإِبْرَاهِيمُ وَآدَمُ ٱلْكِتَابُ",
    "مُوسَى عِيسَى",
    "العـــلم نـــور",
    "مدرسة جميلة",
    "قَالَ: «الْعِلْمُ نُورٌ»، وَ... ؟ ؛ ﴿الحمد لله﴾",
    "Hadith no. 12 (Bukhari) — 'sahih'!",
    "line1\r\rline2\n\tline3\u00a0x\u2003y\x1cz",
    "١٢٣ 123 ۱۲۳",
    "😀 مرحبا 🌙",
    "latin-1 «café» only",
]
# The extension reads CPython's code units directly, so it also accepts strings
# that are not valid UTF-8 (lone surrogates), exactly like the reference.
SURROGATE_SAMPLES = ["lone surrogate \ud800 أ", "\udfff\u064e\ud83d"]


@pytest.mark.parametrize("text", SAMPLES)
def test_default_matches(text: str) -> None:
    assert rs.normalize(text) == py.normalize(text)
    assert rs.tokenize(text) == py.tokenize(text)


@pytest.mark.parametrize("text", SAMPLES)
@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize("value", [True, False])
def test_each_flag_matches(text: str, flag: str, value: bool) -> None:
    assert rs.normalize(text, **{flag: value}) == py.normalize(text, **{flag: value})


@pytest.mark.parametrize("text", SURROGATE_SAMPLES)
def test_surrogates_normalize(text: str) -> None:
    assert rs.normalize(text) == py.normalize(text)
    assert rs.normalize_batch([text]) == [py.normalize(text)]
    assert rs.tokenize(text) == py.tokenize(text)
    assert rs.tokenize_batch_joined([text]) == py.tokenize_batch_joined([text])


def test_unchanged_input_is_returned_as_is() -> None:
    s = "already normal text"
    assert rs.normalize(s) is s


def test_known_outputs() -> None:
    assert rs.normalize("إِمَامٌ") == "امام"
    assert rs.normalize("مُوسَى") == "موسي"
    assert rs.normalize("مدرسة") == "مدرسة"
    assert rs.normalize("مدرسة", ta_marbuta=True) == "مدرسه"
    assert rs.normalize("العـــلم") == "العلم"
    assert rs.normalize("إِمَامٌ", strip_tashkeel=False, normalize_alif=False) == "إِمَامٌ"
    assert rs.tokenize("قَالَ: «الْعِلْمُ نُورٌ»، وَ...") == ["قال", "العلم", "نور", "و"]


def test_batch_matches_single() -> None:
    texts = SAMPLES * 50
    assert rs.normalize_batch(texts) == [rs.normalize(t) for t in texts]
    assert rs.normalize_batch(texts, remove_punct=True, ta_marbuta=True) == py.normalize_batch(
        texts, remove_punct=True, ta_marbuta=True
    )
    assert rs.tokenize_batch_joined(texts) == py.tokenize_batch_joined(texts)


def test_naive_loop_baseline_matches() -> None:
    for t in SAMPLES:
        assert py.normalize_loop(t) == py.normalize(t)


# Alphabet weighted towards the interesting code points.
arabic = st.characters(min_codepoint=0x0600, max_codepoint=0x06FF)
specials = st.sampled_from(
    list("أإآٱاىيةهـ،؛؟«»﴿﴾…—–!?.,:;()[]\"' \t\r\n\u00a0\u2003\x1c\x1d\x1e\x1f\u0085\u2028")
)
anything = st.characters()  # includes lone surrogates
text_strategy = st.text(st.one_of(arabic, specials, anything), max_size=200)


@settings(max_examples=2000, deadline=None)
@given(text=text_strategy, flags=st.fixed_dictionaries({f: st.booleans() for f in FLAGS}))
def test_property_normalize(text: str, flags: dict[str, bool]) -> None:
    assert rs.normalize(text, **flags) == py.normalize(text, **flags)


@settings(max_examples=2000, deadline=None)
@given(text=text_strategy, flags=st.fixed_dictionaries({f: st.booleans() for f in TOK_FLAGS}))
def test_property_tokenize(text: str, flags: dict[str, bool]) -> None:
    assert rs.tokenize(text, **flags) == py.tokenize(text, **flags)


@settings(max_examples=200, deadline=None)
@given(texts=st.lists(text_strategy, max_size=30))
def test_property_batch(texts: list[str]) -> None:
    assert rs.normalize_batch(texts) == py.normalize_batch(texts)
    assert rs.tokenize_batch_joined(texts) == py.tokenize_batch_joined(texts)
