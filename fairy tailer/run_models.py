from language import load_book, build_vocabulary, count_unigrams, count_bigrams, unigram_probabilities, bigram_probabilities, generate_text_unigram, generate_text_bigram

# Load text from files as 2D list (corpus)
corpus_andersen = load_book(r'C:\Users\gourob\Desktop\Infinizy\fairy tailer\andersen.txt')
corpus_grimm = load_book(r'C:\Users\gourob\Desktop\Infinizy\fairy tailer\grimm.txt')

# Build vocabulary and count unigrams and bigrams for Andersen text
vocabulary_andersen = build_vocabulary(corpus_andersen)
unigram_counts_andersen = count_unigrams(corpus_andersen)
bigram_counts_andersen = count_bigrams([word for sentence in corpus_andersen for word in sentence])

# Compute probabilities
unigram_probs_andersen = unigram_probabilities(unigram_counts_andersen)
bigram_probs_andersen = bigram_probabilities(bigram_counts_andersen, unigram_counts_andersen)

# Generate text using unigram and bigram models
print("Generated text (unigram model):")
print(generate_text_unigram(unigram_probs_andersen, length=50))

print("\nGenerated text (bigram model):")
print(generate_text_bigram(bigram_probs_andersen, start_word='once', length=50))
