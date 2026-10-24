def normalize(
    text: str,
    *,
    strip_tashkeel: bool = True,
    strip_tatweel: bool = True,
    normalize_alif: bool = True,
    normalize_ya: bool = True,
    ta_marbuta: bool = False,
    remove_punct: bool = False,
) -> str: ...
def normalize_batch(
    texts: list[str],
    *,
    strip_tashkeel: bool = True,
    strip_tatweel: bool = True,
    normalize_alif: bool = True,
    normalize_ya: bool = True,
    ta_marbuta: bool = False,
    remove_punct: bool = False,
) -> list[str]: ...
def tokenize(
    text: str,
    *,
    strip_tashkeel: bool = True,
    strip_tatweel: bool = True,
    normalize_alif: bool = True,
    normalize_ya: bool = True,
    ta_marbuta: bool = False,
) -> list[str]: ...
def tokenize_batch_joined(
    texts: list[str],
    *,
    strip_tashkeel: bool = True,
    strip_tatweel: bool = True,
    normalize_alif: bool = True,
    normalize_ya: bool = True,
    ta_marbuta: bool = False,
) -> list[str]: ...
def num_threads() -> int: ...
