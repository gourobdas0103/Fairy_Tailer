from language import load_book, tokenize, build_vocabulary, count_unigrams, count_bigrams, unigram_probabilities, bigram_probabilities, generate_text_unigram, generate_text_bigram, make_start_corpus

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
