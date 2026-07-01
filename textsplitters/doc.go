// Package textsplitters provides deterministic document chunking utilities.
//
// Optional Python splitters such as NLTK, spaCy, KoNLPy, and
// sentence-transformers are represented as small Go adapter interfaces instead
// of hard dependencies:
//   - SentenceTokenizer adapts sentence-list tokenizers such as NLTK
//     sent_tokenize, spaCy sents, or KoNLPy Kkma sentences.
//   - SentenceSpanTokenizer adapts span-based sentence tokenizers such as
//     NLTK Punkt span_tokenize while preserving original inter-sentence
//     whitespace.
//   - TokenIDTokenizer adapts integer tokenizers such as tiktoken,
//     sentence-transformers tokenizers, or provider-specific tokenizers.
package textsplitters
