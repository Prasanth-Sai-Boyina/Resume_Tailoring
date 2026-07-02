import re

import pypandoc


def latex_to_md_string(tex: str) -> str:
    if hasattr(pypandoc, "convert_text"):
        md = pypandoc.convert_text(tex, "gfm", format="latex")
    else:
        with open("temp.tex", "w", encoding="utf-8") as handle:
            handle.write(tex)
        md = pypandoc.convert_file("temp.tex", "gfm")

    return clean_markdown(md)


def clean_markdown(md: str) -> str:
    md = re.sub(r"\n{3,}", "\n\n", md)
    md = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", md)
    return md.strip()


def latex_to_markdown(input_path: str, output_path: str) -> str:
    with open(input_path, encoding="utf-8") as handle:
        tex = handle.read()

    md = latex_to_md_string(tex)

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(md)

    return md
