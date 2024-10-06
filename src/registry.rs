use phf::phf_map;
use regex::Regex;
pub type Rank = u32;
use lazy_static::lazy_static;

pub static CL100K_BASE_SPECIAL_TOKENS: phf::Map<&'static str, Rank> = phf_map! {
    "<|endoftext|>" => 100257,
    "<|fim_prefix|>" => 100258,
    "<|fim_middle|>" => 100259,
    "<|fim_suffix|>" => 100260,
    "<|endofprompt|>" => 100276,
};

// TODO: not sure if the regex is correct, the original is
// r"(?i:'(?:s|d|m|t|ll|ve|re))|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(!?\S)|\s+"
// in tiktoken
// (?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+
lazy_static! {
  pub static ref CL100K_BASE_PAT: Regex = Regex::new(r"(?i:'(?:s|d|m|t|ll|ve|re))|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+$|\s+").unwrap();
}