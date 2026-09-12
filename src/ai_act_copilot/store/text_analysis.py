"""Language-aware analysis for the lexical index.

Keyword search on legal text lives or dies on two details: numbers must survive
tokenisation ("Article 6" and "Annex III" are the query), and French needs its own
stopwords, elisions and stemming, or "traitement des donnees" never matches "traitements
de donnees".

Source stays ASCII: the typographic apostrophe is written as an escape so it cannot be
confused with a quote when reading or editing this file.
"""

import re
import unicodedata
from functools import lru_cache
from typing import Any

import snowballstemmer

from ai_act_copilot.models import Language

APOSTROPHE = "’"  # typographic apostrophe, normalised to the ASCII one

# Words, or numbers possibly written as "6.2" - the dotted form appears in guidance sections.
_TOKEN = re.compile(r"\d+(?:\.\d+)*|[^\W\d_]+", re.UNICODE)

# The apostrophe already splits tokens, so an elision arrives as its own fragment:
# "l'article" tokenises to "l" + "article". Those fragments carry no meaning.
_ELISIONS = frozenset(
    ("l", "d", "j", "m", "t", "s", "n", "c", "qu", "jusqu", "lorsqu", "puisqu", "quoiqu")
)

_FRENCH_WORDS = """
    a au aux avec ce ces dans de des du elle en et eux il ils je la le les leur lui ma mais
    me meme mes moi mon ne nos notre nous on ou par pas pour qu que qui sa se ses son sur ta
    te tes toi ton tu un une vos votre vous y etre avoir fait faire plus tout tous toute
    toutes autre autres comme si sans sous entre chaque tel telle telles tels ainsi alors
    donc car ni or cette cet celui celle ceux celles dont lorsque afin selon
    est sont ont eu ete etait seront peut peuvent doit doivent quel quelle quels quelles
    quoi comment pourquoi quand ou lequel laquelle lesquels lesquelles vers apres avant
    pendant depuis chez leurs notamment ainsi lors doit-il est-ce
"""

_ENGLISH_WORDS = """
    a an and are as at be been being but by for from had has have he her his i if in into is
    it its of on or she that the their them there these they this to was were what when
    where which who will with would you your shall may such other than then those any each
    all also both more most no not only same so some very do does did how why whose
    about after before during under between within must can could should would
"""

FRENCH_STOPWORDS = frozenset(_FRENCH_WORDS.split()) | _ELISIONS
ENGLISH_STOPWORDS = frozenset(_ENGLISH_WORDS.split())

# Short queries carry no stopwords: "article 6 paragraphe 2" would otherwise look English.
# These words appear in French legal questions and in no English one.
_FRENCH_MARKER_WORDS = """
    paragraphe paragraphes annexe considerant considerants reglement systeme systemes
    donnees traitement deployeur fournisseur interdites entreprise
"""
_FRENCH_MARKERS = frozenset(_FRENCH_MARKER_WORDS.split())
_FRENCH_ACCENTS = frozenset("àâçèéêëîïôùû")

_STOPWORDS = {Language.FR: FRENCH_STOPWORDS, Language.EN: ENGLISH_STOPWORDS}
_SNOWBALL = {Language.FR: "french", Language.EN: "english"}


@lru_cache(maxsize=4)
def _stemmer(language: Language) -> Any:
    return snowballstemmer.stemmer(_SNOWBALL[language])


def analyse(text: str, language: Language) -> list[str]:
    """Lowercase, drop stopwords and elisions, stem - numbers are left untouched."""
    stopwords = _STOPWORDS[language]
    stemmer = _stemmer(language)
    tokens: list[str] = []
    for match in _TOKEN.finditer(text.lower().replace(APOSTROPHE, "'")):
        token = match.group(0)
        if token in stopwords:
            continue
        if token[0].isdigit():
            tokens.append(token)  # never stem a number: it is the citation
        else:
            tokens.append(str(stemmer.stemWord(token)))
    return tokens


def detect_language(text: str) -> Language:
    """Guess the language of a query, defaulting to English when nothing points either way."""
    if _FRENCH_ACCENTS & set(text.lower()):
        return Language.FR

    words = {
        _strip_accents(match.group(0).lower())
        for match in _TOKEN.finditer(text.replace(APOSTROPHE, "'"))
    }
    if words & _FRENCH_MARKERS:
        return Language.FR
    french = len(words & {_strip_accents(word) for word in FRENCH_STOPWORDS})
    english = len(words & ENGLISH_STOPWORDS)
    return Language.FR if french > english else Language.EN


def _strip_accents(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char)
    )
