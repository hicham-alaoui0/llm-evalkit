import re


def normalize(s: str) -> str:
    """Lowercase and smooth over formatting that doesn't change meaning:
    "1,000" == "1000", "10 %" == "10%", "per cent" == "%", Oxford commas, curly quotes, dashes."""
    s = s.lower().replace("’", "'").replace("–", "-").replace("—", "-")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    s = s.replace(",", " ")
    s = re.sub(r"\s+%", "%", s)
    s = s.replace("per cent", "%").replace("percent", "%")
    return re.sub(r"\s+", " ", s).strip()
