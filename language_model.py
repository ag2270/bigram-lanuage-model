import random

PUNCT = '.!()[]":;?'
STOP = "."


# ==========================================
# 1. Token Extraction & Text Preprocessing
# ==========================================

def add_spaces_around_punct(s: str) -> str:
    """
    Insert one space before AND after every char in PUNCT,
    EXCEPT: do NOT add spaces around '.' when it is part of a decimal number.
    Decimal rule: a dot is part of a decimal if it has a digit immediately to its right
    (e.g., .5, 3.14, 2.0). Preserve newlines; do not collapse or trim existing spaces.
    """
    result = []
    n = len(s)
    i = 0
    while i < n:
        character = s[i]
        if character in PUNCT:
            # Check if period is part of a decimal number
            if character == '.' and i + 1 < n and s[i + 1].isdigit():
                result.append(character)
            else:
                result.append(" " + character + " ")
        else:
            result.append(character)
        i += 1
    return "".join(result)


def read_tokens(file_name: str) -> list:
    """Read all tokens from the given file, preserving order."""
    tokens = []
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            spaced_line = add_spaces_around_punct(line)
            tokens.extend(spaced_line.split())
    return tokens


def build_bigram_counts(tokens: list) -> dict:
    """
    Build bigram transition counts:
    bigram_counts[current_word][next_word] = count
    """
    bigram_counts = {}
    for i in range(len(tokens) - 1):
        current_word = tokens[i]
        next_word = tokens[i + 1]

        if current_word == '.':
            continue

        if current_word not in bigram_counts:
            bigram_counts[current_word] = {}

        if next_word not in bigram_counts[current_word]:
            bigram_counts[current_word][next_word] = 0

        bigram_counts[current_word][next_word] += 1

    return bigram_counts


# ==========================================
# 2. Probabilistic Sentence Generation
# ==========================================

def pick_start(tokens: list) -> str:
    """Pick a random starting word that is not a STOP punctuation mark."""
    while True:
        token = random.choice(tokens)
        if token != STOP:
            return token


def choose_next_word(current_word: str, bigram_counts: dict, tokens: list) -> str:
    """
    Choose the next word given the current word using weighted frequency distribution.
    Falls back to a random token if the word is unseen.
    """
    if current_word in bigram_counts and bigram_counts[current_word]:
        next_words_dict = bigram_counts[current_word]
        words = list(next_words_dict.keys())
        counts = list(next_words_dict.values())
        chosen_word = random.choices(words, weights=counts, k=1)[0]
        return chosen_word
    else:
        return random.choice(tokens)


def genSentence(tokens: list, bigram_counts: dict, max_len: int = 50) -> str:
    """Generate a sentence based on learned bigram transition probabilities."""
    sentence = []
    start_word = pick_start(tokens)
    sentence.append(start_word)

    while len(sentence) < max_len:
        current_word = sentence[-1]
        if current_word == STOP:
            break
        next_word = choose_next_word(current_word, bigram_counts, tokens)
        sentence.append(next_word)

    if sentence[-1] != STOP:
        sentence.append(STOP)

    return " ".join(sentence)
