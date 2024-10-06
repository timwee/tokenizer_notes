use std::collections::HashMap;
use crate::registry::{Rank, CL100K_BASE_SPECIAL_TOKENS, CL100K_BASE_PAT};
use phf::phf_map;
use regex::Regex;
use crate::io::read_bpe_openai;
use fancy_regex;

struct BPENaive<'a, 'b> {
    vocab: HashMap<Vec<u8>, Rank>,
    special_tokens: &'b phf::Map<&'static str, Rank>,
    split_pat: &'a Regex,
}

impl<'a, 'b> BPENaive<'a, 'b> {
    fn new(
        vocab: HashMap<Vec<u8>, Rank>,
        special_tokens: &'b phf::Map<&'static str, Rank>,
        split_pat: &'a Regex,
    ) -> Self {
        Self {
            vocab,
            special_tokens,
            split_pat,
        }
    }

    fn _byte_pair_encode(&self, piece: &[u8]) -> Vec<Rank> {
      assert!(piece.len() > 0);
      todo!()
    }

    fn encode(&self, text: &str) -> Vec<Rank> {
        let mut values = Vec::new();
        self.split_pat.find_iter(text).for_each(|m| {
            let k = m.as_str().as_bytes();

            match self.vocab.get(k) {
                Some(rank) => values.push(*rank),
                None => {
                    values.extend(&self._byte_pair_encode(k));
                }
            }
        });
        values
    }

    fn decode(&self, ranks: &[Rank]) -> String {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_encode() {
      let vocab = read_bpe_openai("encoding_data/cl100k_base.tiktoken").unwrap();
      let bpe = BPENaive::new(vocab, &CL100K_BASE_SPECIAL_TOKENS, &CL100K_BASE_PAT);
      assert_eq!(bpe.encode("0"), vec![15]);
      assert_eq!(bpe.encode("rer"), vec![38149]);
    }
}
