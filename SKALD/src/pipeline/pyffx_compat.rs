use hmac::{Hmac, Mac};
use sha1::Sha1;
use std::collections::BTreeMap;

const DEFAULT_ROUNDS: usize = 10;
const SHA1_DIGEST_SIZE: usize = 20;

type HmacSha1 = Hmac<Sha1>;

#[derive(Debug, Clone)]
pub struct AlphabetMap {
    pub alphabet: Vec<char>,
    index_map: BTreeMap<char, u32>,
}

impl AlphabetMap {
    pub fn new(alphabet: &str) -> Result<Self, String> {
        let chars: Vec<char> = alphabet.chars().collect();
        if chars.len() < 2 {
            return Err("alphabet must contain at least 2 characters".to_string());
        }
        let mut index_map = BTreeMap::new();
        for (i, c) in chars.iter().copied().enumerate() {
            if index_map.insert(c, i as u32).is_some() {
                return Err(format!("duplicate character in alphabet: {c}"));
            }
        }
        Ok(Self {
            alphabet: chars,
            index_map,
        })
    }

    pub fn radix(&self) -> u32 {
        self.alphabet.len() as u32
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        let mut out = Vec::with_capacity(text.chars().count());
        for c in text.chars() {
            let idx = self
                .index_map
                .get(&c)
                .copied()
                .ok_or_else(|| format!("non-alphabet character: {c}"))?;
            out.push(idx);
        }
        Ok(out)
    }

    pub fn decode(&self, values: &[u32]) -> Result<String, String> {
        let mut out = String::with_capacity(values.len());
        for &v in values {
            let i = v as usize;
            if i >= self.alphabet.len() {
                return Err(format!("digit out of range for alphabet: {v}"));
            }
            out.push(self.alphabet[i]);
        }
        Ok(out)
    }
}

fn pyffx_chars_per_hash(radix: u32) -> usize {
    // int(digest_size * math.log(256, radix))
    ((SHA1_DIGEST_SIZE as f64) * (256f64.ln() / (radix as f64).ln())) as usize
}

fn u32_le_bytes(v: u32, out: &mut Vec<u8>) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn fill_round_digits(
    key: &[u8],
    radix: u32,
    round_index: usize,
    right: &[u32],
    out_digits: &mut [u32],
    key_buf: &mut Vec<u8>,
) {
    key_buf.clear();
    u32_le_bytes(round_index as u32, key_buf);
    for &d in right {
        u32_le_bytes(d, key_buf);
    }

    let chars_per_hash = pyffx_chars_per_hash(radix).max(1);
    let mut generated = 0usize;
    let mut counter = 0u32;

    while generated < out_digits.len() {
        let mut mac = HmacSha1::new_from_slice(key).expect("HMAC key");
        mac.update(key_buf);
        mac.update(&counter.to_le_bytes());
        let digest = mac.finalize().into_bytes();

        // Convert digest to big integer and emit radix digits exactly like pyffx:
        // d = int(h.hexdigest(), 16); repeat divmod(d, radix).
        let mut d = [0u8; SHA1_DIGEST_SIZE];
        d.copy_from_slice(&digest);

        for _ in 0..chars_per_hash {
            if generated >= out_digits.len() {
                break;
            }
            let rem = divmod_be_bytes_in_place(&mut d, radix as u16);
            out_digits[generated] = rem as u32;
            generated += 1;
        }

        key_buf.clear();
        key_buf.extend_from_slice(&digest);
        counter = counter.wrapping_add(1);
    }
}

fn divmod_be_bytes_in_place(bytes: &mut [u8], divisor: u16) -> u16 {
    let mut rem: u16 = 0;
    for b in bytes.iter_mut() {
        let cur = (rem << 8) | (*b as u16);
        *b = (cur / divisor) as u8;
        rem = cur % divisor;
    }
    rem
}

pub fn fpe_encrypt_checked(key: &[u8], text: &str, alphabet: &str) -> Result<String, String> {
    let map = AlphabetMap::new(alphabet)?;
    let mut v = map.encode(text)?;
    if v.is_empty() {
        return Ok(String::new());
    }
    let radix = map.radix();
    let split = v.len() / 2;
    let mut a = v[..split].to_vec();
    let mut b = v[split..].to_vec();
    let mut round_digits = vec![0_u32; a.len().max(b.len())];
    let mut key_buf = Vec::with_capacity((b.len() + 2) * 4);

    for i in 0..DEFAULT_ROUNDS {
        round_digits.resize(a.len(), 0);
        fill_round_digits(key, radix, i, &b, &mut round_digits, &mut key_buf);
        let mut c = vec![0_u32; a.len()];
        for j in 0..a.len() {
            c[j] = (a[j] + round_digits[j]) % radix;
        }
        a = b;
        b = c;
    }

    v.clear();
    v.extend_from_slice(&a);
    v.extend_from_slice(&b);
    map.decode(&v)
}

pub fn fpe_decrypt_checked(key: &[u8], text: &str, alphabet: &str) -> Result<String, String> {
    let map = AlphabetMap::new(alphabet)?;
    let mut v = map.encode(text)?;
    if v.is_empty() {
        return Ok(String::new());
    }
    let radix = map.radix();
    let split = v.len() / 2;
    let mut a = v[..split].to_vec();
    let mut b = v[split..].to_vec();
    let mut round_digits = vec![0_u32; a.len().max(b.len())];
    let mut key_buf = Vec::with_capacity((b.len() + 2) * 4);

    for i in (0..DEFAULT_ROUNDS).rev() {
        let old_b = a;
        let c = b;
        round_digits.resize(c.len(), 0);
        fill_round_digits(key, radix, i, &old_b, &mut round_digits, &mut key_buf);
        let mut old_a = vec![0_u32; c.len()];
        for j in 0..c.len() {
            old_a[j] = (c[j] + radix - (round_digits[j] % radix)) % radix;
        }
        a = old_a;
        b = old_b;
    }

    v.clear();
    v.extend_from_slice(&a);
    v.extend_from_slice(&b);
    map.decode(&v)
}

pub fn fpe_encrypt(key: &[u8], text: &str, alphabet: &str) -> String {
    fpe_encrypt_checked(key, text, alphabet).unwrap_or_else(|_| text.to_string())
}

pub fn fpe_decrypt(key: &[u8], text: &str, alphabet: &str) -> String {
    fpe_decrypt_checked(key, text, alphabet).unwrap_or_else(|_| text.to_string())
}

#[cfg(test)]
mod tests {
    use super::{fpe_decrypt_checked, fpe_encrypt_checked};

    #[test]
    fn pyffx_vector_digits_key_foo() {
        let key = b"foo";
        let alphabet = "0123456789";
        let plain = "123456";
        let enc = fpe_encrypt_checked(key, plain, alphabet).expect("encrypt");
        assert_eq!(enc, "979962");
        let dec = fpe_decrypt_checked(key, &enc, alphabet).expect("decrypt");
        assert_eq!(dec, plain);
    }

    #[test]
    fn pyffx_vector_leading_zeros() {
        let key = b"foo";
        let alphabet = "0123456789";
        let plain = "000001";
        let enc = fpe_encrypt_checked(key, plain, alphabet).expect("encrypt");
        assert_eq!(enc, "897283");
        let dec = fpe_decrypt_checked(key, &enc, alphabet).expect("decrypt");
        assert_eq!(dec, plain);
    }

    #[test]
    fn pyffx_vector_alnum() {
        let key = b"foo";
        let alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        let plain = "ABCDE1234F";
        let enc = fpe_encrypt_checked(key, plain, alphabet).expect("encrypt");
        assert_eq!(enc, "LQOZ1T7HUQ");
        let dec = fpe_decrypt_checked(key, &enc, alphabet).expect("decrypt");
        assert_eq!(dec, plain);
    }

    #[test]
    fn pyffx_vector_hex_key_digits() {
        let key = hex::decode("00112233445566778899aabbccddeeff").expect("hex");
        let alphabet = "0123456789";
        let plain = "9876543210";
        let enc = fpe_encrypt_checked(&key, plain, alphabet).expect("encrypt");
        assert_eq!(enc, "9926847934");
        let dec = fpe_decrypt_checked(&key, &enc, alphabet).expect("decrypt");
        assert_eq!(dec, plain);
    }

    #[test]
    fn pyffx_vector_hex_key_alpha() {
        let key = hex::decode("00112233445566778899aabbccddeeff").expect("hex");
        let alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let plain = "ABCDE";
        let enc = fpe_encrypt_checked(&key, plain, alphabet).expect("encrypt");
        assert_eq!(enc, "TOBFD");
        let dec = fpe_decrypt_checked(&key, &enc, alphabet).expect("decrypt");
        assert_eq!(dec, plain);
    }
}
