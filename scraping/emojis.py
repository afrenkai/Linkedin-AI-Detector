import re


def get_emoji_codes(
    inp_fname: str = "emoji_list.html", out_fname: str = "unicode_regex.txt"
) -> None:
    emoji_codes = []
    PAT = r"\bU[+][0-9A-Z]{3,5}\b"
    with open(inp_fname, "r") as f:
        for line in f.readlines():
            matcher = str(re.findall(PAT, line)[0]) if re.findall(PAT, line) else None
            if matcher and matcher not in emoji_codes:
                emoji_codes.append(matcher)
    f.close()
    for i in range(len(emoji_codes)):
        # case for 5 long, which targets seq of 8
        if len(emoji_codes[i].split("U+")[1]) > 4:
            # print(emoji_codes[i].split("U+")[1])
            emoji_codes[i] = emoji_codes[i].replace("U+", ("\\U000"))
        else:
            emoji_codes[i] = emoji_codes[i].replace("U+", "\\u")

    codes_str = ""
    for item in emoji_codes:
        codes_str += item + "|"

    with open(out_fname, "w") as f:
        f.write(codes_str)


if __name__ == "__main__":
    get_emoji_codes()
