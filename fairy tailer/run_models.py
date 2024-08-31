from language import load_text, tokenize, build_vocabulary, count_unigrams, count_bigrams, unigram_probabilities, bigram_probabilities, generate_text_unigram, generate_text_bigram

# Load text from files
text_andersen = load_text(r'C:\Users\gourob\Desktop\Infinizy\fairy tailer\andersen.txt')
text_grimm = load_text(r'C:\Users\gourob\Desktop\Infinizy\fairy tailer\grimm.txt')

# Tokenize and build models for Andersen text
words_andersen = tokenize(text_andersen)
unigram_counts_andersen = count_unigrams(words_andersen)
bigram_counts_andersen = count_bigrams(words_andersen)
unigram_probs_andersen = unigram_probabilities(unigram_counts_andersen)
bigram_probs_andersen = bigram_probabilities(bigram_counts_andersen, unigram_counts_andersen)

# Generate text using unigram and bigram models
print("Generated text (unigram model):")
print(generate_text_unigram(unigram_probs_andersen, length=50))

print("\nGenerated text (bigram model):")
print(generate_text_bigram(bigram_probs_andersen, start_word='once', length=50))
