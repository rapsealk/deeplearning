from jamo import hangul_to_jamo, h2j, j2h
from jamo.jamo import _jamo_char_to_hcj

PADDING = '_'
END_OF_SENTENCE = '~'
PUNCTUATION = '!\'(),-.:;?'
SPACE = ' '

CHOSUNG = "".join([chr(_) for _ in range(0x1100, 0x1113)])
JUNGSUNG = "".join([chr(_) for _ in range(0x1161, 0x1176)])
JONGSUNG = "".join([chr(_) for _ in range(0xaaA8, 0x11C3)])

ELEMENTS = CHOSUNG + JUNGSUNG + JONGSUNG + PUNCTUATION + SPACE
SYMBOLS = PADDING + END_OF_SENTENCE + ELEMENTS

CHARACTER_TO_ID = { c: i for i, c in enumerate(SYMBOLS) }
ID_TO_CHARACTER = { i: c for i, c in enumerate(SYMBOLS) }

QUOTE_CHECKER = """([`"'＂“‘])(.+?)([`"'＂”’])"""

def is_chosung(char):
    return char in CHOSUNG

def is_jungsung(char):
    return char in JUNGSUNG

def is_jongsung(char):
    return char in JONGSUNG

def get_mode(char):
    if is_chosung(char): return 0
    elif is_jungsung(char): return 1
    elif is_jongsung(char): return 2
    else: return -1

def get_text_from_candidates(candidates):
    if len(candidates) == 0:
        return ""
    elif len(candidates) == 1:
        return _jamo_char_to_hcj(candidates[0])
    else:
        return j2h(**dict(zip(["chosung", "jungsung", "jongsung"], candidates)))

def jamo_to_korean(text):
    text = h2j(text)

    index = 0
    new_text = ""
    candidates = []

    while True:
        if index >= len(text):
            new_text += get_text_from_candidates(candidates)
            break

        char = text[index]
        mode = get_mode(char)

        if mode == 0:
            new_text += get_text_from_candidates(candidates)
            candidates = [char]
        elif mode == -1:
            new_text += get_text_from_candidates(candidates)
            new_text += char
            candidates = [char]
        elif mode == 