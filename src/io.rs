use base64::prelude::*;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

pub fn read_bpe_openai(path: &str) -> Result<HashMap<Vec<u8>, u64>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut vocab = HashMap::new();
    for line in reader.lines() {
        let line = line?;
        let parts = line.split_whitespace().collect::<Vec<&str>>();
        if parts.len() == 2 {
            let token = BASE64_STANDARD.decode(parts[0])?;
            let id = parts[1].parse::<u64>()?;
            vocab.insert(token, id);
        }
    }
    Ok(vocab)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_bpe_openai() {
        let vocab = read_bpe_openai("encoding_data/cl100k_base.tiktoken").unwrap();
        assert_eq!(*vocab.get("0".as_bytes()).unwrap(), 15);
        assert_eq!(*vocab.get("rer".as_bytes()).unwrap(), 38149);
    }
}
