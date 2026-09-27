import re

PLATE_PATTERN = re.compile(r'^[A-Z]{1,2}[0-9]{1,4}[A-Z]{0,3}$')

DICT_CHAR_TO_INT = {'O':'0','I':'1','Z':'2','A':'4','G':'6','S':'5','B':'8'}
DICT_INT_TO_CHAR = {'0':'O','1':'I','2':'Z','4':'A','6':'G','5':'S','8':'B'}


def clean_plate_text(text: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', text.upper())


def format_indonesian_plate(text: str) -> str | None:
    text = clean_plate_text(text)
    if not text:
        return None

    first_digit = re.search(r'\d', text)
    if not first_digit:
        return None

    idx = first_digit.start()
    prefix = text[:idx]

    if not (1 <= len(prefix) <= 2):
        return None

    number, i = "", idx
    while i < len(text) and text[i].isdigit() and len(number) < 4:
        number += text[i]
        i += 1

    suffix = text[i:]

    prefix = ''.join(DICT_INT_TO_CHAR.get(c, c) for c in prefix)
    number = ''.join(DICT_CHAR_TO_INT.get(c, c) for c in number)
    suffix = ''.join(DICT_INT_TO_CHAR.get(c, c) for c in suffix)

    if not (1 <= len(number) <= 4) or len(suffix) > 3:
        return None

    candidate = prefix + number + suffix
    if not PLATE_PATTERN.match(candidate):
        return None

    return f"{prefix} {number} {suffix}".strip() if suffix else f"{prefix} {number}"