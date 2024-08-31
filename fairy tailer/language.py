import re
from collections import Counter, defaultdict
import random

def tokenize(text):
    """Convert text into a list of words."""
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    return words

def build_vocabulary(words):
    """Create a set of unique words."""
    return set(words)

def count_unigrams(words):
    """Count occurrences of each word."""
    return Counter(words)

def count_bigrams(words):
    """Count occurrences of each word pair."""
    bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
    return Counter(bigrams)

def uniform_probabilities(vocabulary):
    """Compute uniform probabilities for all words in the vocabulary."""
    vocab_size = len(vocabulary)
    return {word: 1 / vocab_size for word in vocabulary}

def unigram_probabilities(unigram_counts):
    """Compute probabilities of words based on unigram counts."""
    total_count = sum(unigram_counts.values())
    return {word: count / total_count for word, count in unigram_counts.items()}

def bigram_probabilities(bigram_counts, unigram_counts):
    """Compute probabilities of word pairs based on bigram counts."""
    bigram_probs = defaultdict(lambda: defaultdict(float))
    for (w1, w2), count in bigram_counts.items():
        bigram_probs[w1][w2] = count / unigram_counts[w1]  # P(w2|w1)
    return bigram_probs

def load_text(file_path):
    """Read text from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def load_book(filename):
    """Load a book and create a 2D list (corpus) where each row is a sentence and each column is a word or symbol."""
    corpus = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            sentence = line.strip().split()  # Split the line into words
            corpus.append(sentence)          # Append the list of words to the corpus
    return corpus

def generate_text_unigram(unigram_probs, length=50):
    """Generate text based on unigram probabilities."""
    words = list(unigram_probs.keys())
    text = []

    for _ in range(length):
        next_word = random.choices(words, weights=unigram_probs.values())[0]
        text.append(next_word)

    return ' '.join(text)

def generate_text_bigram(bigram_probs, start_word, length=50):
    """Generate text based on bigram probabilities."""
    current_word = start_word
    text = [current_word]

    for _ in range(length - 1):
        next_word_probs = bigram_probs.get(current_word, {})

        if not next_word_probs:
            # If no valid bigram found, pick a random word to continue
            next_word = random.choice(list(bigram_probs.keys()))
        else:
            next_word = random.choices(list(next_word_probs.keys()), weights=next_word_probs.values())[0]

        text.append(next_word)
        current_word = next_word

    return ' '.join(text)
