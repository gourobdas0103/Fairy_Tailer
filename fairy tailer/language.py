import re
from collections import defaultdict, Counter

def load_book(filename):
    """Load a book and create a 2D list (corpus) where each row is a sentence and each column is a word."""
    corpus = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            sentence = re.findall(r'\b\w+\b', line.lower())
            if sentence:
                corpus.append(sentence)
    return corpus

def tokenize(text):
    """Tokenize the text into words."""
    return re.findall(r'\b\w+\b', text.lower())

def build_vocabulary(corpus):
    """Build vocabulary from the corpus."""
    vocabulary = set()
    for sentence in corpus:
        vocabulary.update(sentence)
    return vocabulary

def count_unigrams(corpus):
    """Count unigrams in the corpus."""
    unigram_counts = Counter()
    for sentence in corpus:
        unigram_counts.update(sentence)
    return unigram_counts

def count_bigrams(corpus):
    """Count bigrams in the corpus."""
    bigram_counts = defaultdict(Counter)
    for sentence in corpus:
        for i in range(len(sentence) - 1):
            word1, word2 = sentence[i], sentence[i + 1]
            if word1 not in bigram_counts:
                bigram_counts[word1] = Counter()
            bigram_counts[word1][word2] += 1
    return bigram_counts

def unigram_probabilities(unigram_counts):
    """Calculate unigram probabilities."""
    total_count = sum(unigram_counts.values())
    return {word: count / total_count for word, count in unigram_counts.items()}

def bigram_probabilities(bigram_counts, unigram_counts):
    """Calculate bigram probabilities based on bigram and unigram counts."""
    bigram_probs = defaultdict(dict)
    for w1, counter in bigram_counts.items():
        if unigram_counts[w1] > 0:
            bigram_probs[w1] = {w2: count / unigram_counts[w1] for w2, count in counter.items()}
        else:
            print(f"Warning: '{w1}' has a count of zero in unigram counts, skipping bigram probabilities for '{w1}'.")
    return bigram_probs

def generate_text_unigram(vocabulary, unigram_probs, length=100):
    """Generate text based on unigram model."""
    import random
    return ' '.join(random.choices(list(vocabulary), weights=unigram_probs.values(), k=length))

def generate_text_bigram(bigram_probs, start_word, length=100):
    """Generate text based on bigram model starting with a given word."""
    import random
    current_word = start_word
    text = [current_word]
    for _ in range(length - 1):
        next_words = list(bigram_probs.get(current_word, {}).keys())
        if next_words:
            next_word = random.choices(next_words, weights=bigram_probs[current_word].values())[0]
            text.append(next_word)
            current_word = next_word
        else:
            break
    return ' '.join(text)

def make_start_corpus(corpus):
    """Create a new corpus containing only the starting word of each sentence."""
    return [[sentence[0]] for sentence in corpus if sentence]

def build_unigram_probs(unigrams, unigram_counts, total_count):
    """Build a list of unigram probabilities."""
    return [unigram_counts.get(word, 0) / total_count for word in unigrams]
