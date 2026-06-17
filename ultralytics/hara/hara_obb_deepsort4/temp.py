import unicodedata

chars = {
    "Zero Width Space": "\u200B",
    "Hangul Filler": "\u3164",
    "Braille Pattern Blank": "\u2800",
    "Normal Space": "\u0020",
}

print("a\u200B\u3164\u2800\u200B\u3164\u2800\u3164\u3164\u200Ba")

for desc, ch in chars.items():
    code_point = f"U+{ord(ch):04X}"
    unicode_name = unicodedata.name(ch, "UNKNOWN")

    print(f"{desc}")
    print(f"  Character between brackets: [{ch}]")
    print(f"  Code point: {code_point}")
    print(f"  Unicode name: {unicode_name}")
    print(f"  Length: {len(ch)}")
    print()