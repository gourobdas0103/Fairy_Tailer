from language import load_book, tokenize, build_vocabulary, count_unigrams, count_bigrams, unigram_probabilities, bigram_probabilities, generate_text_unigram, generate_text_bigram

# Load text from files as 2D list (corpus)
corpus_andersen = load_book(r'C:\Users\gourob\Desktop\Infinizy\fairy tailer\andersen.txt')
corpus_grimm = load_book(r'C:\Users\gourob\Desktop\Infinizy\fairy tailer\grimm.txt')

# Flatten the corpus into a single list of words
words_andersen = [word for sentence in corpus_andersen for word in sentence]
words_grimm = [word for sentence in corpus_grimm for word in sentence]

# Tokenize and build models for Andersen text
unigram_counts_andersen = count_unigrams(words_andersen)
bigram_counts_andersen = count_bigrams(words_andersen)
vocabulary_andersen = build_vocabulary(words_andersen)

# Compute probabilities
unigram_probs_andersen = unigram_probabilities(unigram_counts_andersen)
bigram_probs_andersen = bigram_probabilities(bigram_counts_andersen, unigram_counts_andersen)

# Generate text using unigram and bigram models
print("Generated text (unigram model):")
print(generate_text_unigram(unigram_probs_andersen, length=50))

print("\nGenerated text (bigram model):")
print(generate_text_bigram(bigram_probs_andersen, start_word='once', length=50))
