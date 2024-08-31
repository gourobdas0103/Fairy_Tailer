import unittest
from language import *

class TestLanguageModels(unittest.TestCase):

    def setUp(self):
        self.text = "Once upon a time in a land far away, there was a kingdom ruled by a wise king."
        self.words = tokenize(self.text)
        self.unigram_counts = count_unigrams(self.words)
        self.bigram_counts = count_bigrams(self.words)
        self.vocabulary = build_vocabulary(self.words)
        self.unigram_probs = unigram_probabilities(self.unigram_counts)
        self.bigram_probs = bigram_probabilities(self.bigram_counts, self.unigram_counts)

    def test_tokenize(self):
        self.assertEqual(self.words, ['once', 'upon', 'a', 'time', 'in', 'a', 'land', 'far', 'away', 'there', 'was', 'a', 'kingdom', 'ruled', 'by', 'a', 'wise', 'king'])

    def test_build_vocabulary(self):
        self.assertEqual(self.vocabulary, {'once', 'upon', 'a', 'time', 'in', 'land', 'far', 'away', 'there', 'was', 'kingdom', 'ruled', 'by', 'wise', 'king'})

    def test_count_unigrams(self):
        self.assertEqual(self.unigram_counts['a'], 4)

    def test_count_bigrams(self):
        self.assertEqual(self.bigram_counts[('a', 'time')], 1)

    def test_unigram_probabilities(self):
        self.assertAlmostEqual(self.unigram_probs['a'], 4 / 16)

    def test_bigram_probabilities(self):
        self.assertAlmostEqual(self.bigram_probs['a']['time'], 1 / 4)

    def test_generate_text_unigram(self):
        generated_text = generate_text_unigram(self.unigram_probs, length=10)
        self.assertTrue(len(generated_text.split()) == 10)

    def test_generate_text_bigram(self):
        generated_text = generate_text_bigram(self.bigram_probs, start_word='a', length=10)
        self.assertTrue(len(generated_text.split()) == 10)

if __name__ == '__main__':
    unittest.main()
