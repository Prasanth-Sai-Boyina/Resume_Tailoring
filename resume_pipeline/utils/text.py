import re
import string

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "ought", "used", "we", "you", "they", "he", "she", "it", "i", "our",
    "your", "their", "this", "that", "these", "those", "as", "not", "no",
    "nor", "so", "yet", "both", "either", "neither", "such", "than",
    "whether", "while", "also", "well", "just", "up", "about", "into",
    "through", "during", "including", "until", "against", "among", "across",
    "any", "each", "few", "more", "other", "some", "such", "own", "same",
    "who", "which", "what", "when", "where", "how", "all", "both", "every",
    "must", "work", "experience", "ability", "knowledge", "strong", "good",
    "skills", "skill", "team", "role", "candidate", "job", "position",
    "looking", "seeking", "required", "preferred", "plus", "bonus", "etc",
    "help", "make", "build", "develop", "use", "using", "working", "work",
    "years", "year", "month", "day", "time", "new", "based", "across",
}

_PRESERVE = {
    "ml", "ai", "nlp", "cv", "sql", "api", "aws", "gcp", "ci", "cd",
    "ux", "ui", "qa", "ios", "c++", "c#", "r", "go",
}


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)
    return text.split()


def extract_keywords(text: str, min_length: int = 2) -> set[str]:
    tokens = _tokenize(text)
    keywords: set[str] = set()

    for token in tokens:
        if token in _PRESERVE:
            keywords.add(token)
            continue
        if len(token) < min_length:
            continue
        if token in _STOPWORDS:
            continue
        clean = token.rstrip(".")
        if clean:
            keywords.add(clean)

    return keywords


def sanitize_for_prompt(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")
