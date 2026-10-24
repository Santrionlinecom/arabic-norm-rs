//! Fast Arabic text normalization for retrieval (RAG) preprocessing.
//!
//! Every rule is a pure per-character mapping, so the Rust implementation and
//! the pure-Python reference in `python/arabic_norm/reference.py` can be
//! checked against each other character by character.

use pyo3::prelude::*;

mod ucs;

/// Normalization switches. Mirrors the keyword arguments of the Python API.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub strip_tashkeel: bool,
    pub strip_tatweel: bool,
    pub normalize_alif: bool,
    pub normalize_ya: bool,
    pub ta_marbuta: bool,
    pub remove_punct: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            strip_tashkeel: true,
            strip_tatweel: true,
            normalize_alif: true,
            normalize_ya: true,
            ta_marbuta: false,
            remove_punct: false,
        }
    }
}

/// Harakat, tanwin, shadda, sukun, Quranic annotation marks and superscript alif.
#[inline]
pub fn is_tashkeel(c: char) -> bool {
    matches!(c,
        '\u{0610}'..='\u{061A}'
        | '\u{064B}'..='\u{065F}'
        | '\u{0670}'
        | '\u{06D6}'..='\u{06DC}'
        | '\u{06DF}'..='\u{06E8}'
        | '\u{06EA}'..='\u{06ED}')
}

/// ASCII punctuation plus common Arabic / typographic punctuation.
#[inline]
pub fn is_punct(c: char) -> bool {
    c.is_ascii_punctuation()
        || matches!(
            c,
            '\u{060C}' // ، comma
            | '\u{061B}' // ؛ semicolon
            | '\u{061F}' // ؟ question mark
            | '\u{066A}'..='\u{066D}' // ٪ ٫ ٬ ٭
            | '\u{06D4}' // ۔ full stop
            | '\u{00AB}' | '\u{00BB}' // « »
            | '\u{2010}'..='\u{2027}' // dashes, quotes, bullets, ellipsis
            | '\u{2030}'..='\u{205E}' // per-mille, primes, misc punctuation
            | '\u{FD3E}' | '\u{FD3F}' // ﴾ ﴿ ornate parentheses
        )
}

/// Same definition of whitespace as Python's `str.isspace()`.
#[inline]
pub fn is_py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{001C}'..='\u{001F}').contains(&c)
}

/// What happens to one character.
pub(crate) enum Edit {
    Keep,
    Drop,
    Replace(char),
}

#[inline(always)]
pub(crate) fn edit(c: char, o: &Options) -> Edit {
    if c.is_ascii() {
        // ASCII is never Arabic; only punctuation may change.
        return if o.remove_punct && c.is_ascii_punctuation() {
            Edit::Replace(' ')
        } else {
            Edit::Keep
        };
    }
    if o.strip_tashkeel && is_tashkeel(c) {
        return Edit::Drop;
    }
    if o.strip_tatweel && c == '\u{0640}' {
        return Edit::Drop;
    }
    match c {
        '\u{0623}' | '\u{0625}' | '\u{0622}' | '\u{0671}' if o.normalize_alif => {
            Edit::Replace('\u{0627}')
        }
        '\u{0649}' if o.normalize_ya => Edit::Replace('\u{064A}'),
        '\u{0629}' if o.ta_marbuta => Edit::Replace('\u{0647}'),
        _ if o.remove_punct && is_punct(c) => Edit::Replace(' '),
        _ => Edit::Keep,
    }
}

/// Normalize one string. Allocates once; unchanged runs are copied with a
/// single `memcpy` instead of being re-encoded character by character.
pub fn normalize_str(text: &str, o: &Options) -> String {
    let mut out = String::with_capacity(text.len());
    let mut run = 0; // start of the pending unchanged run
    for (i, c) in text.char_indices() {
        match edit(c, o) {
            Edit::Keep => {}
            Edit::Drop => {
                out.push_str(&text[run..i]);
                run = i + c.len_utf8();
            }
            Edit::Replace(r) => {
                out.push_str(&text[run..i]);
                out.push(r);
                run = i + c.len_utf8();
            }
        }
    }
    out.push_str(&text[run..]);
    out
}

/// Split an already-normalized string on Python-compatible whitespace.
pub fn split_tokens(normalized: &str) -> impl Iterator<Item = &str> {
    normalized.split(is_py_space).filter(|t| !t.is_empty())
}

/// UTF-8 equivalent of the Python `tokenize_batch_joined` for one string.
pub fn tokens_joined(text: &str, o: &Options) -> String {
    let norm = normalize_str(text, o);
    let mut out = String::with_capacity(norm.len());
    for (i, tok) in split_tokens(&norm).enumerate() {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(tok);
    }
    out
}

#[pymodule]
mod _core {
    use super::*;
    use pyo3::types::{PyList, PyString};

    fn opts(flags: [bool; 6]) -> Options {
        let [
            strip_tashkeel,
            strip_tatweel,
            normalize_alif,
            normalize_ya,
            ta_marbuta,
            remove_punct,
        ] = flags;
        Options {
            strip_tashkeel,
            strip_tatweel,
            normalize_alif,
            normalize_ya,
            ta_marbuta,
            remove_punct,
        }
    }

    /// Normalize a single string.
    #[pyfunction]
    #[pyo3(signature = (text, *, strip_tashkeel=true, strip_tatweel=true, normalize_alif=true,
                        normalize_ya=true, ta_marbuta=false, remove_punct=false))]
    #[allow(clippy::too_many_arguments)]
    fn normalize<'py>(
        py: Python<'py>,
        text: &Bound<'py, PyString>,
        strip_tashkeel: bool,
        strip_tatweel: bool,
        normalize_alif: bool,
        normalize_ya: bool,
        ta_marbuta: bool,
        remove_punct: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let o = opts([
            strip_tashkeel,
            strip_tatweel,
            normalize_alif,
            normalize_ya,
            ta_marbuta,
            remove_punct,
        ]);
        ucs::normalize_one(py, text, &o)
    }

    /// Normalize a list of strings in parallel (rayon) with the GIL released.
    #[pyfunction]
    #[pyo3(signature = (texts, *, strip_tashkeel=true, strip_tatweel=true, normalize_alif=true,
                        normalize_ya=true, ta_marbuta=false, remove_punct=false))]
    #[allow(clippy::too_many_arguments)]
    fn normalize_batch<'py>(
        py: Python<'py>,
        texts: Vec<Bound<'py, PyString>>,
        strip_tashkeel: bool,
        strip_tatweel: bool,
        normalize_alif: bool,
        normalize_ya: bool,
        ta_marbuta: bool,
        remove_punct: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let o = opts([
            strip_tashkeel,
            strip_tatweel,
            normalize_alif,
            normalize_ya,
            ta_marbuta,
            remove_punct,
        ]);
        ucs::normalize_many(py, &texts, &o)
    }

    /// Normalize (punctuation always removed) and split into tokens.
    #[pyfunction]
    #[pyo3(signature = (text, *, strip_tashkeel=true, strip_tatweel=true, normalize_alif=true,
                        normalize_ya=true, ta_marbuta=false))]
    fn tokenize<'py>(
        py: Python<'py>,
        text: &Bound<'py, PyString>,
        strip_tashkeel: bool,
        strip_tatweel: bool,
        normalize_alif: bool,
        normalize_ya: bool,
        ta_marbuta: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let o = opts([
            strip_tashkeel,
            strip_tatweel,
            normalize_alif,
            normalize_ya,
            ta_marbuta,
            true,
        ]);
        ucs::tokenize_one(py, text, &o)
    }

    /// Tokenize many strings in parallel with the GIL released. Returns one
    /// space-joined token string per input, which pandas / Polars can split
    /// with a plain `str.split(" ")`.
    #[pyfunction]
    #[pyo3(signature = (texts, *, strip_tashkeel=true, strip_tatweel=true, normalize_alif=true,
                        normalize_ya=true, ta_marbuta=false))]
    fn tokenize_batch_joined<'py>(
        py: Python<'py>,
        texts: Vec<Bound<'py, PyString>>,
        strip_tashkeel: bool,
        strip_tatweel: bool,
        normalize_alif: bool,
        normalize_ya: bool,
        ta_marbuta: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let o = opts([
            strip_tashkeel,
            strip_tatweel,
            normalize_alif,
            normalize_ya,
            ta_marbuta,
            true,
        ]);
        ucs::tokenize_many_joined(py, &texts, &o)
    }

    /// Number of rayon worker threads (set `RAYON_NUM_THREADS` to change).
    #[pyfunction]
    fn num_threads() -> usize {
        rayon::current_num_threads()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_harakat_and_unifies_alif() {
        let o = Options::default();
        assert_eq!(normalize_str("إِمَامٌ", &o), "امام");
        assert_eq!(normalize_str("مُوسَى", &o), "موسي");
        assert_eq!(normalize_str("العـــلم", &o), "العلم");
    }

    #[test]
    fn tokenizes() {
        let o = Options {
            remove_punct: true,
            ..Options::default()
        };
        assert_eq!(
            tokens_joined("قَالَ: «الْعِلْمُ نُورٌ»، وَ...", &o),
            "قال العلم نور و"
        );
    }
}
