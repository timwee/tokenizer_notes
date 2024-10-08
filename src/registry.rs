use fancy_regex::Regex;
use phf::phf_map;
pub type Rank = u32;
use lazy_static::lazy_static;

pub static CL100K_BASE_SPECIAL_TOKENS: phf::Map<&'static str, Rank> = phf_map! {
    "<|endoftext|>" => 100257,
    "<|fim_prefix|>" => 100258,
    "<|fim_middle|>" => 100259,
    "<|fim_suffix|>" => 100260,
    "<|endofprompt|>" => 100276,
};

// Here's a breakdown of the pattern:
// 1. '(?i:[sdmt]|ll|ve|re):
// Matches contractions starting with an apostrophe.
// (?i:...) makes the match case-insensitive.
// Matches 's, 'd, 'm, 't, 'll, 've, or 're.

// 2. |[^\r\n\p{L}\p{N}]?+\p{L}+:
// Matches any letter sequence, optionally preceded by a single character that is not a letter, number, or line break.
// \p{L} represents any Unicode letter.
// \p{N} represents any Unicode number.

// 3. |\p{N}{1,3}:
// Matches 1 to 3 consecutive numbers.

// 4. | ?[^\s\p{L}\p{N}]++[\r\n]*:
// Matches any sequence of characters that are not whitespace, letters, or numbers, optionally preceded by a space and followed by any number of line breaks.

// 5. |\s*[\r\n]:
// Matches any line break, optionally preceded by whitespace.

// 6. |\s+(?!\S):
// Matches one or more whitespace characters at the end of a string.

// 7. |\s+:
// Matches one or more whitespace characters.

// This regex seems designed to tokenize text, breaking it into words, numbers, punctuation, and whitespace. It can handle contractions, numbers up to 3 digits, various punctuation, and different types of whitespace including line breaks.
// The pattern uses alternation (|) to match any of these sub-patterns, effectively creating a tokenizer that can split text into various components.
// in tiktoken
// (?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+
lazy_static! {
  pub static ref CL100K_BASE_PAT: Regex = Regex::new(r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+").unwrap();
}
