#1. Text Preparation

import re
from collections import Counter, defaultdict

def tokenize(text):
    """Convert text into a list of words."""
    text = text.lower()  # Convert text to lowercase
    words = re.findall(r'\b\w+\b', text)  # Find all words
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

def load_text(file_path):
    """Read text from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

#2. Implement Language Models

#Unigram Model: Calculate word probabilities.

def unigram_probabilities(unigram_counts):
    """Compute probabilities of words based on counts."""
    total_count = sum(unigram_counts.values())  # Total number of words
    return {word: count / total_count for word, count in unigram_counts.items()}

#Bigram Model: Calculate pair probabilities.

def bigram_probabilities(bigram_counts, unigram_counts):
    """Compute probabilities of word pairs based on counts."""
    bigram_probs = defaultdict(lambda: defaultdict(float))
    for (w1, w2), count in bigram_counts.items():
        bigram_probs[w1][w2] = count / unigram_counts[w1]  # P(w2|w1)
    return bigram_probs

#3. Generate Text

import random

def generate_text_unigram(unigram_probs, length=50):
    """Generate text using unigram probabilities."""
    words = list(unigram_probs.keys())
    return ' '.join(random.choices(words, weights=unigram_probs.values(), k=length))

def generate_text_bigram(bigram_probs, start_word, length=50):
    """Generate text using bigram probabilities."""
    current_word = start_word
    text = [current_word]
    for _ in range(length - 1):
        if current_word not in bigram_probs:
            break
        next_words = list(bigram_probs[current_word].keys())
        if not next_words:
            break
        next_word = random.choices(next_words, weights=bigram_probs[current_word].values())[0]
        text.append(next_word)
        current_word = next_word
    return ' '.join(text)

