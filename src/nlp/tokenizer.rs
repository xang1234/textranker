//! Unicode-aware tokenization
//!
//! This module provides UAX #29 compliant word and sentence segmentation
//! with support for CJK, emoji, contractions, and other Unicode scripts.

use crate::types::{PosTag, Sentence, Token};
use unicode_segmentation::UnicodeSegmentation;

/// A Unicode-aware tokenizer following UAX #29
#[derive(Debug, Clone, Default)]
pub struct Tokenizer {
    /// Minimum token length to consider
    min_token_length: usize,
}

impl Tokenizer {
    /// Create a new tokenizer with default settings
    pub fn new() -> Self {
        Self {
            min_token_length: 1,
        }
    }

    /// Set minimum token length
    pub fn with_min_length(mut self, min_length: usize) -> Self {
        self.min_token_length = min_length;
        self
    }

    /// Tokenize text into sentences and tokens
    ///
    /// This performs basic tokenization without POS tagging.
    /// For full NLP features, use spaCy preprocessing via Python.
    pub fn tokenize(&self, text: &str) -> (Vec<Sentence>, Vec<Token>) {
        let mut sentences = Vec::new();
        let mut tokens = Vec::new();
        let mut token_idx = 0;

        // Split into sentences first
        let sentence_bounds = self.sentence_boundaries(text);

        for (sent_idx, (start, end)) in sentence_bounds.iter().enumerate() {
            let sent_text = &text[*start..*end];
            let mut sentence = Sentence::new(sent_text.trim(), *start, *end, sent_idx);
            sentence.start_token = token_idx;

            // Tokenize words within sentence
            for (word_start, word) in sent_text.unicode_word_indices() {
                let abs_start = start + word_start;
                let abs_end = abs_start + word.len();

                // Skip tokens that are too short
                if word.chars().count() < self.min_token_length {
                    continue;
                }

                // Skip pure punctuation/symbols
                if !word.chars().any(|c| c.is_alphanumeric()) {
                    continue;
                }

                // Create token with basic heuristic POS tagging
                let pos = self.guess_pos(word);
                let lemma = self.basic_lemmatize(word);

                let token = Token::new(word, lemma, pos, abs_start, abs_end, sent_idx, token_idx);

                tokens.push(token);
                token_idx += 1;
            }

            sentence.end_token = token_idx;
            sentences.push(sentence);
        }

        (sentences, tokens)
    }

    /// Find sentence boundaries in text
    fn sentence_boundaries(&self, text: &str) -> Vec<(usize, usize)> {
        let mut boundaries = Vec::new();
        let mut start = 0;

        // Use unicode sentence boundaries, but also handle newlines
        for (idx, _) in text.split_sentence_bound_indices() {
            if idx > start {
                let segment = &text[start..idx];
                // Skip empty segments
                if !segment.trim().is_empty() {
                    boundaries.push((start, idx));
                }
            }
            start = idx;
        }

        // Handle last segment
        if start < text.len() {
            let segment = &text[start..];
            if !segment.trim().is_empty() {
                boundaries.push((start, text.len()));
            }
        }

        // If no sentences found, treat entire text as one sentence
        if boundaries.is_empty() && !text.trim().is_empty() {
            boundaries.push((0, text.len()));
        }

        boundaries
    }

    /// Basic heuristic POS tagging (for when spaCy is not available)
    ///
    /// This is intentionally simple - for accurate POS tags, use spaCy preprocessing.
    fn guess_pos(&self, word: &str) -> PosTag {
        let lower = word.to_lowercase();

        if let Some(pos) = self.function_word_pos(&lower) {
            return pos;
        }

        // Check for common patterns
        if word
            .chars()
            .next()
            .map(|c| c.is_uppercase())
            .unwrap_or(false)
            && word.chars().skip(1).all(|c| c.is_lowercase())
        {
            // Capitalized word (might be proper noun or sentence start)
            return PosTag::ProperNoun;
        }

        // Common adjective suffixes
        if lower.ends_with("ful")
            || lower.ends_with("less")
            || lower.ends_with("ous")
            || lower.ends_with("ive")
            || lower.ends_with("able")
            || lower.ends_with("ible")
            || lower.ends_with("al")
            || lower.ends_with("ic")
        {
            return PosTag::Adjective;
        }

        // Common verb suffixes
        if lower.ends_with("ing") || lower.ends_with("ed") || lower.ends_with("ize") {
            return PosTag::Verb;
        }

        // Common adverb suffix
        if lower.ends_with("ly") {
            return PosTag::Adverb;
        }

        // Common noun suffixes
        if lower.ends_with("tion")
            || lower.ends_with("ness")
            || lower.ends_with("ment")
            || lower.ends_with("ity")
            || lower.ends_with("er")
            || lower.ends_with("or")
        {
            return PosTag::Noun;
        }

        // Numbers
        if word
            .chars()
            .all(|c| c.is_ascii_digit() || c == '.' || c == ',')
        {
            return PosTag::Numeral;
        }

        // Default to noun (most content words are nouns)
        PosTag::Noun
    }

    fn function_word_pos(&self, lower: &str) -> Option<PosTag> {
        let pos = match lower {
            // Determiners
            "a" | "an" | "the" | "this" | "that" | "these" | "those" | "my" | "your" | "his"
            | "her" | "its" | "our" | "their" | "some" | "any" | "each" | "every" | "no" => {
                PosTag::Determiner
            }
            // Conjunctions
            "and" | "or" | "but" | "nor" | "so" | "yet" | "if" | "because" | "while"
            | "though" | "although" | "when" | "unless" | "until" | "since" => PosTag::Conjunction,
            // Prepositions
            "of" | "to" | "in" | "for" | "on" | "with" | "at" | "from" | "by" | "about" | "as"
            | "into" | "like" | "through" | "after" | "over" | "between" | "out" | "against"
            | "during" | "without" | "before" | "under" | "around" | "among" => PosTag::Preposition,
            // Pronouns
            "i" | "you" | "he" | "she" | "it" | "we" | "they" | "me" | "him" | "us"
            | "them" | "myself" | "yourself" | "ourselves" | "themselves" => PosTag::Pronoun,
            // Common particles
            "not" | "n't" => PosTag::Particle,
            _ => return None,
        };
        Some(pos)
    }

    /// Basic lemmatization (for when spaCy is not available)
    ///
    /// This handles simple English morphology. For accurate lemmas, use spaCy.
    fn basic_lemmatize(&self, word: &str) -> String {
        let lower = word.to_lowercase();

        // Handle common English suffixes
        if lower.ends_with("ies") && lower.len() > 4 {
            return format!("{}y", &lower[..lower.len() - 3]);
        }
        if lower.ends_with("es") && lower.len() > 3 {
            let stem = &lower[..lower.len() - 2];
            if stem.ends_with("ss")
                || stem.ends_with("sh")
                || stem.ends_with("ch")
                || stem.ends_with('x')
                || stem.ends_with('o')
            {
                return stem.to_string();
            }
        }
        if lower.ends_with('s') && lower.len() > 2 && !lower.ends_with("ss") {
            return lower[..lower.len() - 1].to_string();
        }
        if lower.ends_with("ing") && lower.len() > 5 {
            let stem = &lower[..lower.len() - 3];
            // Check for doubled consonant (running -> run)
            let chars: Vec<char> = stem.chars().collect();
            if chars.len() >= 2 && chars[chars.len() - 1] == chars[chars.len() - 2] {
                return stem[..stem.len() - 1].to_string();
            }
            return stem.to_string();
        }
        if lower.ends_with("ed") && lower.len() > 4 {
            let stem = &lower[..lower.len() - 2];
            // Check for doubled consonant
            let chars: Vec<char> = stem.chars().collect();
            if chars.len() >= 2 && chars[chars.len() - 1] == chars[chars.len() - 2] {
                return stem[..stem.len() - 1].to_string();
            }
            if lower.ends_with("ied") {
                return format!("{}y", &lower[..lower.len() - 3]);
            }
            return stem.to_string();
        }

        lower
    }

    /// Check if a character is CJK
    pub fn is_cjk(c: char) -> bool {
        matches!(c,
            '\u{4E00}'..='\u{9FFF}' |   // CJK Unified Ideographs
            '\u{3400}'..='\u{4DBF}' |   // CJK Extension A
            '\u{20000}'..='\u{2A6DF}' | // CJK Extension B
            '\u{2A700}'..='\u{2B73F}' | // CJK Extension C
            '\u{2B740}'..='\u{2B81F}' | // CJK Extension D
            '\u{F900}'..='\u{FAFF}' |   // CJK Compatibility
            '\u{3000}'..='\u{303F}' |   // CJK Punctuation
            '\u{3040}'..='\u{309F}' |   // Hiragana
            '\u{30A0}'..='\u{30FF}' |   // Katakana
            '\u{AC00}'..='\u{D7AF}'     // Hangul
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokenization() {
        let tokenizer = Tokenizer::new();
        let (sentences, tokens) = tokenizer.tokenize("Hello world. This is a test.");

        assert_eq!(sentences.len(), 2);
        assert!(tokens.len() >= 6); // At least "Hello", "world", "This", "is", "a", "test"
    }

    #[test]
    fn test_unicode_handling() {
        let tokenizer = Tokenizer::new();
        let (sentences, tokens) = tokenizer.tokenize("Café résumé naïve. 日本語テスト。");

        assert!(!sentences.is_empty());
        // Should handle accented characters and CJK
        assert!(tokens.iter().any(|t| t.text.contains('é')));
    }

    #[test]
    fn test_basic_lemmatization() {
        let tokenizer = Tokenizer::new();

        assert_eq!(tokenizer.basic_lemmatize("running"), "run");
        assert_eq!(tokenizer.basic_lemmatize("cats"), "cat");
        assert_eq!(tokenizer.basic_lemmatize("studies"), "study");
        assert_eq!(tokenizer.basic_lemmatize("boxes"), "box");
    }

    #[test]
    fn test_pos_guessing() {
        let tokenizer = Tokenizer::new();

        assert_eq!(tokenizer.guess_pos("beautiful"), PosTag::Adjective);
        assert_eq!(tokenizer.guess_pos("running"), PosTag::Verb);
        assert_eq!(tokenizer.guess_pos("quickly"), PosTag::Adverb);
        assert_eq!(tokenizer.guess_pos("information"), PosTag::Noun);
    }

    #[test]
    fn test_cjk_detection() {
        assert!(Tokenizer::is_cjk('中'));
        assert!(Tokenizer::is_cjk('日'));
        assert!(Tokenizer::is_cjk('あ'));
        assert!(Tokenizer::is_cjk('ア'));
        assert!(!Tokenizer::is_cjk('A'));
        assert!(!Tokenizer::is_cjk('1'));
    }

    #[test]
    fn test_empty_input() {
        let tokenizer = Tokenizer::new();
        let (sentences, tokens) = tokenizer.tokenize("");

        assert!(sentences.is_empty());
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_emoji_handling() {
        let tokenizer = Tokenizer::new();
        let (sentences, _tokens) = tokenizer.tokenize("Hello 👋 world! How are you? 🎉");

        // Should still identify sentence boundaries
        assert!(sentences.len() >= 2);
    }
}
