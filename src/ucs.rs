//! Zero-copy fast path over CPython's internal string storage (PEP 393).
//!
//! CPython stores a `str` as an array of 1-, 2- or 4-byte code units. Arabic
//! text is almost always the 2-byte kind. Converting it to UTF-8 for Rust and
//! back again costs more than the normalization itself, so these functions
//! read the code units in place and build the result in the same kind.
//!
//! Not available under the stable ABI (`abi3`), which is why the crate builds
//! one wheel per CPython minor version.

use crate::{Edit, Options, edit, is_py_space};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};
use rayon::prelude::*;
use std::os::raw::c_void;

/// A code unit of a PEP 393 string.
trait Unit: Copy + Send + Sync {
    fn code(self) -> u32;
    fn from_code(c: u32) -> Self;
}
impl Unit for u8 {
    fn code(self) -> u32 {
        self as u32
    }
    fn from_code(c: u32) -> Self {
        c as u8
    }
}
impl Unit for u16 {
    fn code(self) -> u32 {
        self as u32
    }
    fn from_code(c: u32) -> Self {
        c as u16
    }
}
impl Unit for u32 {
    fn code(self) -> u32 {
        self
    }
    fn from_code(c: u32) -> Self {
        c
    }
}

/// `None` means "no change": the caller can return the input object as is.
///
/// Replacement characters always fit the source kind: a 1-byte string can only
/// contain ASCII/Latin-1 punctuation (replaced by a space), and every Arabic
/// replacement is in the BMP, i.e. no wider than the 2-byte source it came from.
fn normalize_units<T: Unit>(src: &[T], o: &Options) -> Option<Vec<T>> {
    let mut out: Option<Vec<T>> = None;
    let mut run = 0;
    for (i, &u) in src.iter().enumerate() {
        // Lone surrogates are not `char`s; they are never edited.
        let Some(c) = char::from_u32(u.code()) else {
            continue;
        };
        let e = edit(c, o);
        if matches!(e, Edit::Keep) {
            continue;
        }
        let buf = out.get_or_insert_with(|| Vec::with_capacity(src.len()));
        buf.extend_from_slice(&src[run..i]);
        if let Edit::Replace(r) = e {
            buf.push(T::from_code(r as u32));
        }
        run = i + 1;
    }
    if let Some(buf) = out.as_mut() {
        buf.extend_from_slice(&src[run..]);
    }
    out
}

/// Raw view of a live `str` object's code units. Valid while the object is
/// referenced; Python strings are immutable, so concurrent reads are sound.
#[derive(Clone, Copy)]
struct View {
    kind: u32,
    data: *const c_void,
    len: usize,
}
// SAFETY: the view only ever reads immutable memory owned by a `str` object
// that the caller keeps alive for the whole parallel section.
unsafe impl Send for View {}
unsafe impl Sync for View {}

enum Out {
    Same,
    K1(Vec<u8>),
    K2(Vec<u16>),
    K4(Vec<u32>),
}

fn view(s: &Bound<'_, PyString>) -> View {
    let p = s.as_ptr();
    // SAFETY: `p` is a valid, ready (PEP 393) unicode object.
    unsafe {
        View {
            kind: ffi::PyUnicode_KIND(p),
            data: ffi::PyUnicode_DATA(p),
            len: ffi::PyUnicode_GET_LENGTH(p) as usize,
        }
    }
}

fn run(v: View, o: &Options) -> Out {
    // SAFETY: `data` points to `len` code units of width `kind`.
    unsafe {
        match v.kind {
            ffi::PyUnicode_1BYTE_KIND => {
                let s = std::slice::from_raw_parts(v.data as *const u8, v.len);
                normalize_units(s, o).map_or(Out::Same, Out::K1)
            }
            ffi::PyUnicode_2BYTE_KIND => {
                let s = std::slice::from_raw_parts(v.data as *const u16, v.len);
                normalize_units(s, o).map_or(Out::Same, Out::K2)
            }
            _ => {
                let s = std::slice::from_raw_parts(v.data as *const u32, v.len);
                normalize_units(s, o).map_or(Out::Same, Out::K4)
            }
        }
    }
}

fn to_py<'py>(
    py: Python<'py>,
    src: &Bound<'py, PyString>,
    out: Out,
) -> PyResult<Bound<'py, PyAny>> {
    let (kind, ptr, len) = match &out {
        Out::Same => return Ok(src.clone().into_any()),
        Out::K1(v) => (
            ffi::PyUnicode_1BYTE_KIND,
            v.as_ptr() as *const c_void,
            v.len(),
        ),
        Out::K2(v) => (
            ffi::PyUnicode_2BYTE_KIND,
            v.as_ptr() as *const c_void,
            v.len(),
        ),
        Out::K4(v) => (
            ffi::PyUnicode_4BYTE_KIND,
            v.as_ptr() as *const c_void,
            v.len(),
        ),
    };
    // SAFETY: buffer is `len` units of `kind`; CPython copies it and computes
    // the narrowest storage, so results are canonical `str` objects.
    unsafe {
        let obj = ffi::PyUnicode_FromKindAndData(kind as i32, ptr, len as ffi::Py_ssize_t);
        Bound::from_owned_ptr_or_err(py, obj)
    }
}

pub fn normalize_one<'py>(
    py: Python<'py>,
    text: &Bound<'py, PyString>,
    o: &Options,
) -> PyResult<Bound<'py, PyAny>> {
    let out = run(view(text), o);
    to_py(py, text, out)
}

pub fn normalize_many<'py>(
    py: Python<'py>,
    texts: &[Bound<'py, PyString>],
    o: &Options,
) -> PyResult<Bound<'py, PyList>> {
    let views: Vec<View> = texts.iter().map(view).collect();
    let o = *o;
    // `texts` keeps every object alive while the GIL is released.
    let outs: Vec<Out> = py.detach(|| views.par_iter().map(|&v| run(v, &o)).collect());
    let items = texts
        .iter()
        .zip(outs)
        .map(|(s, out)| to_py(py, s, out))
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, items)
}

/// Normalize (punctuation removed by the caller's options), drop leading and
/// trailing whitespace and collapse every whitespace run into one space:
/// exactly `" ".join(normalize(t, remove_punct=True).split())`.
fn tokens_units<T: Unit>(src: &[T], o: &Options) -> Vec<T> {
    let mut out = Vec::with_capacity(src.len());
    let mut pending_space = false;
    for &u in src {
        let unit = match char::from_u32(u.code()) {
            None => u, // lone surrogate: an ordinary non-space character
            Some(c) => {
                let c = match edit(c, o) {
                    Edit::Drop => continue,
                    Edit::Replace(r) => r,
                    Edit::Keep => c,
                };
                if is_py_space(c) {
                    pending_space = !out.is_empty();
                    continue;
                }
                T::from_code(c as u32)
            }
        };
        if pending_space {
            out.push(T::from_code(' ' as u32));
            pending_space = false;
        }
        out.push(unit);
    }
    out
}

fn run_tokens(v: View, o: &Options) -> Out {
    // SAFETY: `data` points to `len` code units of width `kind`.
    unsafe {
        match v.kind {
            ffi::PyUnicode_1BYTE_KIND => Out::K1(tokens_units(
                std::slice::from_raw_parts(v.data as *const u8, v.len),
                o,
            )),
            ffi::PyUnicode_2BYTE_KIND => Out::K2(tokens_units(
                std::slice::from_raw_parts(v.data as *const u16, v.len),
                o,
            )),
            _ => Out::K4(tokens_units(
                std::slice::from_raw_parts(v.data as *const u32, v.len),
                o,
            )),
        }
    }
}

fn new_str<'py, T>(py: Python<'py>, kind: u32, units: &[T]) -> PyResult<Bound<'py, PyAny>> {
    // SAFETY: `units` holds `len` code units of width `kind`.
    unsafe {
        let obj = ffi::PyUnicode_FromKindAndData(
            kind as i32,
            units.as_ptr() as *const c_void,
            units.len() as ffi::Py_ssize_t,
        );
        Bound::from_owned_ptr_or_err(py, obj)
    }
}

fn split_list<'py, T: Unit>(
    py: Python<'py>,
    kind: u32,
    joined: &[T],
) -> PyResult<Bound<'py, PyList>> {
    let space = ' ' as u32;
    let items = joined
        .split(|u| u.code() == space)
        .filter(|t| !t.is_empty())
        .map(|t| new_str(py, kind, t))
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, items)
}

pub fn tokenize_one<'py>(
    py: Python<'py>,
    text: &Bound<'py, PyString>,
    o: &Options,
) -> PyResult<Bound<'py, PyList>> {
    match run_tokens(view(text), o) {
        Out::K1(v) => split_list(py, ffi::PyUnicode_1BYTE_KIND, &v),
        Out::K2(v) => split_list(py, ffi::PyUnicode_2BYTE_KIND, &v),
        Out::K4(v) => split_list(py, ffi::PyUnicode_4BYTE_KIND, &v),
        Out::Same => unreachable!("run_tokens always builds a buffer"),
    }
}

pub fn tokenize_many_joined<'py>(
    py: Python<'py>,
    texts: &[Bound<'py, PyString>],
    o: &Options,
) -> PyResult<Bound<'py, PyList>> {
    let views: Vec<View> = texts.iter().map(view).collect();
    let o = *o;
    let outs: Vec<Out> = py.detach(|| views.par_iter().map(|&v| run_tokens(v, &o)).collect());
    let items = texts
        .iter()
        .zip(outs)
        .map(|(s, out)| to_py(py, s, out))
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, items)
}
