import re
from collections import defaultdict, Counter
import random

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
            bigram_counts[word1][word2] += 1
    return bigram_counts

def unigram_probabilities(unigram_counts):
    """Calculate unigram probabilities."""
    total_count = sum(unigram_counts.values())
    return {word: count / total_count for word, count in unigram_counts.items()}

def bigram_probabilities(bigram_counts, unigram_counts):
    """Calculate bigram probabilities based on bigram and unigram counts."""
    bigram_probs = defaultdict(lambda: {"words": [], "probs": []})
    for w1, counter in bigram_counts.items():
        if unigram_counts[w1] > 0:
            words = list(counter.keys())
            probs = [count / unigram_counts[w1] for count in counter.values()]
            bigram_probs[w1]["words"] = words
            bigram_probs[w1]["probs"] = probs
    return bigram_probs

def build_unigram_probs(unigrams, unigram_counts, total_count):
    """Build a list of unigram probabilities."""
    return [unigram_counts.get(word, 0) / total_count for word in unigrams]

def generate_text_unigram(vocabulary, unigram_probs, length=100):
    """Generate text based on unigram model."""
    return ' '.join(random.choices(list(vocabulary), weights=unigram_probs.values(), k=length))

def generate_text_bigram(bigram_probs, start_word, length=100):
    """Generate text based on bigram model starting with a given word."""
    current_word = start_word
    text = [current_word]
    for _ in range(length - 1):
        if current_word in bigram_probs and bigram_probs[current_word]["words"]:
            next_words = bigram_probs[current_word]["words"]
            next_probs = bigram_probs[current_word]["probs"]
            next_word = random.choices(next_words, weights=next_probs)[0]
            text.append(next_word)
            current_word = next_word
        else:
            break
    return ' '.join(text)

def make_start_corpus(corpus):
    """Create a new corpus containing only the starting word of each sentence."""
    return [sentence[0] for sentence in corpus if sentence]

def get_top_words(count, words, probs, ignore_list):
    """Get the top `count` words with highest probabilities, ignoring words in `ignore_list`."""
    word_prob_pairs = [(word, prob) for word, prob in zip(words, probs) if word not in ignore_list]
    sorted_words = sorted(word_prob_pairs, key=lambda x: x[1], reverse=True)
    return dict(sorted_words[:count])

def generate_text_from_unigrams(count, words, probs):
    """Generate text based on unigram probabilities."""
    return ' '.join(random.choices(words, weights=probs, k=count))
