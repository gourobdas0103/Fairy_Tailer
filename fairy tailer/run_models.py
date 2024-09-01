from language import load_book, build_vocabulary, count_unigrams, count_bigrams, unigram_probabilities, bigram_probabilities, generate_text_unigram, generate_text_bigram, make_start_corpus, build_unigram_probs, get_top_words

# Load and process books
corpus_andersen = load_book(r'C:\Users\gourob\Desktop\Infinizy\fairy tailer\andersen.txt')
corpus_grimm = load_book(r'C:\Users\gourob\Desktop\Infinizy\fairy tailer\grimm.txt')

# Build vocabulary
vocabulary_andersen = build_vocabulary(corpus_andersen)
vocabulary_grimm = build_vocabulary(corpus_grimm)

# Count unigrams and bigrams
unigram_counts_andersen = count_unigrams(corpus_andersen)
bigram_counts_andersen = count_bigrams(corpus_andersen)

unigram_counts_grimm = count_unigrams(corpus_grimm)
bigram_counts_grimm = count_bigrams(corpus_grimm)

# Calculate total word counts
total_count_andersen = sum(unigram_counts_andersen.values())
total_count_grimm = sum(unigram_counts_grimm.values())

# Build unigram probabilities
unigram_probs_list_andersen = build_unigram_probs(vocabulary_andersen, unigram_counts_andersen, total_count_andersen)
unigram_probs_list_grimm = build_unigram_probs(vocabulary_grimm, unigram_counts_grimm, total_count_grimm)

# Calculate probabilities
unigram_probs_andersen = unigram_probabilities(unigram_counts_andersen)
bigram_probs_andersen = bigram_probabilities(bigram_counts_andersen, unigram_counts_andersen)

unigram_probs_grimm = unigram_probabilities(unigram_counts_grimm)
bigram_probs_grimm = bigram_probabilities(bigram_counts_grimm, unigram_counts_grimm)

# Generate text
text_unigram_andersen = generate_text_unigram(vocabulary_andersen, unigram_probs_andersen, length=100)
text_bigram_andersen = generate_text_bigram(bigram_probs_andersen, start_word='it', length=100)

text_unigram_grimm = generate_text_unigram(vocabulary_grimm, unigram_probs_grimm, length=100)
text_bigram_grimm = generate_text_bigram(bigram_probs_grimm, start_word='once', length=100)

# Print generated text
print("Generated text (unigram model - Andersen):")
print(text_unigram_andersen)
print("\nGenerated text (bigram model - Andersen):")
print(text_bigram_andersen)

print("\nGenerated text (unigram model - Grimm):")
print(text_unigram_grimm)
print("\nGenerated text (bigram model - Grimm):")
print(text_bigram_grimm)

# Print start words
start_words_andersen = make_start_corpus(corpus_andersen)
start_words_grimm = make_start_corpus(corpus_grimm)

print("\nStart words in Andersen corpus:")
print(start_words_andersen)

print("\nStart words in Grimm corpus:")
print(start_words_grimm)

# Print unigram probabilities
print("\nUnigram probabilities (Andersen):")
for word, prob in zip(vocabulary_andersen, unigram_probs_list_andersen):
    print(f"{word}: {prob:.4f}")

print("\nUnigram probabilities (Grimm):")
for word, prob in zip(vocabulary_grimm, unigram_probs_list_grimm):
    print(f"{word}: {prob:.4f}")

# Print top words
ignore_list = []  # Add any words you want to ignore here
top_words_andersen = get_top_words(10, vocabulary_andersen, unigram_probs_list_andersen, ignore_list)
top_words_grimm = get_top_words(10, vocabulary_grimm, unigram_probs_list_grimm, ignore_list)

print("\nTop words (Andersen):")
for word, prob in top_words_andersen.items():
    print(f"{word}: {prob:.4f}")

print("\nTop words (Grimm):")
for word, prob in top_words_grimm.items():
    print(f"{word}: {prob:.4f}")
