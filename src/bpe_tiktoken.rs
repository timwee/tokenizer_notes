use std::collections::HashMap;
use crate::registry::{Rank, CL100K_BASE_SPECIAL_TOKENS, CL100K_BASE_PAT};
use phf::phf_map;
// use regex::Regex;
use crate::io::read_bpe_openai;
use fancy_regex::Regex;

struct BPETiktoken<'b> {
    vocab: HashMap<Vec<u8>, Rank>,
    special_tokens: &'b phf::Map<&'static str, Rank>,
    split_pat: Regex,
    special_tokens_regex: Regex,
}

impl<'b> BPETiktoken<'b> {
    fn new(
        vocab: HashMap<Vec<u8>, Rank>,
        special_tokens: &'b phf::Map<&'static str, Rank>,
        split_pat: Regex,
    ) -> Self {
      let special_tokens_regex = {
            let _parts = special_tokens
                .keys()
                .map(|s| fancy_regex::escape(s))
                .collect::<Vec<_>>();
            Regex::new(&_parts.join("|")).unwrap()           
        };
        Self {
            vocab,
            special_tokens,
            split_pat: split_pat.clone(),
            special_tokens_regex,
        }
    }

    fn cl100k_base() -> Self {
        let vocab = read_bpe_openai("encoding_data/cl100k_base.tiktoken").unwrap();
        Self::new(vocab, &CL100K_BASE_SPECIAL_TOKENS, CL100K_BASE_PAT.clone())
    }

    // Note that we hash bytes when indexing into `vocab`, not token pairs. As long as we train BPE
    // the way we currently do, this is equivalent. An easy way to break this would be to decouple
    // merge priority from token index or to prevent specific token merges.
    // Given a piece, returns the (start, Rank) of each BPE token in the piece.
    fn _byte_pair_merge(&self, piece: &[u8]) -> Vec<(usize, Rank)> {

      // find the rank of the byte pair starting at each byte, RANK_MAX if not in vocab.
      let mut parts: Vec<(usize, Rank)> = Vec::with_capacity(piece.len() + 1);
      let mut min_rank: (Rank, usize) = (Rank::MAX, usize::MAX);
      for i in 0..piece.len() - 1 {
          let rank = *self.vocab.get(&piece[i..i + 2]).unwrap_or(&Rank::MAX);
          if rank < min_rank.0 {
              min_rank = (rank, i);
          }
          parts.push((i, rank));
      }
      parts.push((piece.len() - 1, Rank::MAX));
      parts.push((piece.len(), Rank::MAX));

      let get_rank = {
        #[inline(always)]
        |parts: &Vec<(usize, Rank)>, i: usize| {
            if (i + 3) < parts.len() {
                // Similar to `piece[i..i + 2]` above. The +3 is because we haven't yet deleted
                // parts[i + 1], see comment in the main loop.
                // We use the .0 (usize) field because there may be merged elements already
                //    so we may need the whole subpiece.
                *self.vocab
                    .get(&piece[parts[i].0..parts[i + 3].0])
                    .unwrap_or(&Rank::MAX)
            } else {
                Rank::MAX
            }
        }
      };

      // Run the merge loop until we can't merge any more.

      // If you have n parts and m merges, this does O(mn) work.
      // We could do something with a heap and do O(m log n) work.
      // n is often very small so considerations like cache-locality outweigh the algorithmic
      // complexity downsides of the `parts` vector.
      while min_rank.0 != Rank::MAX {
        // Do the merge for the current min ranking pair by removing [i + 1]th element
        //   and updating the i-th and (i-1)th elements.
        // The pair at these 2 positions are now:
        // - (i-1)th: The i-1th element + the merged i-th and (i+1)th elements
        // - ith: The merged i-th and (i+1)th elements, and the (i+2)th element
        let i = min_rank.1;
        // Update parts[i] and parts[i - 1] before removing parts[i + 1], since
        // `parts.remove(i + 1)` will thrash the cache.
        if i > 0 {
          parts[i - 1].1 = get_rank(&parts, i - 1);
        }
        parts[i].1 = get_rank(&parts, i);
        parts.remove(i + 1);

        // find the next candidate to merge
        min_rank = (Rank::MAX, usize::MAX);
        for (i, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
          if rank < min_rank.0 {
            min_rank = (rank, i);
          }
        }
      }
      parts
    }

    // If the piece does not exist in the vocab, we need to find a tokenization
    //  of the piece, accounting for byte pairs, and return the ranks of the tokens.
    fn _byte_pair_encode(&self, piece: &[u8]) -> Vec<Rank> {
      assert!(piece.len() > 1);
      self._byte_pair_merge(piece)
        .windows(2)
        .map(|parts| self.vocab[&piece[parts[0].0..parts[1].0]])
        .collect()
    }

    fn encode(&self, text: &str) -> Vec<Rank> {
      let mut values = Vec::new();
      self.split_pat.find_iter(text).for_each(|m| {
        eprintln!("match: {:?}", m);
        let k = m.expect("Failed to get match").as_str().as_bytes();

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
      let bpe = BPETiktoken::cl100k_base();
      assert_eq!(bpe.encode("0"), vec![15]);
      assert_eq!(bpe.encode("rer"), vec![38149]);
      assert_eq!(bpe.encode("'rer"), vec![2351, 81]);
      assert_eq!(bpe.encode("today\n "), vec![31213, 198, 220]);
      assert_eq!(bpe.encode("today\n \n"), vec![31213, 27907]);
      assert_eq!(bpe.encode("today\n  \n"), vec![31213, 14211]);
      assert_eq!(bpe.encode("hello world"), vec![15339, 1917]);
      assert_eq!(bpe.encode("👍"), [9468, 239, 235]);
    }
}
