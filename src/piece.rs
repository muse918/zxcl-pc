#![allow(dead_code)]

pub type Piece = u8;

pub const I: Piece = 0;
pub const J: Piece = 1;
pub const L: Piece = 2;
pub const O: Piece = 3;
pub const S: Piece = 4;
pub const T: Piece = 5;
pub const Z: Piece = 6;

pub const PIECE_COUNT: usize = 7;
pub const FULL_BAG: u8 = 0b111_1111;
pub const PIECE_ORDER: &[u8; 7] = b"IJLOSTZ";

#[inline]
pub fn piece_char(p: Piece) -> char {
    PIECE_ORDER[p as usize] as char
}

#[inline]
pub fn piece_from_char(c: char) -> Option<Piece> {
    match c {
        'I' | 'i' => Some(I),
        'J' | 'j' => Some(J),
        'L' | 'l' => Some(L),
        'O' | 'o' => Some(O),
        'S' | 's' => Some(S),
        'T' | 't' => Some(T),
        'Z' | 'z' => Some(Z),
        _ => None,
    }
}

pub fn pieces(mask: u8) -> impl Iterator<Item = Piece> {
    (0u8..PIECE_COUNT as u8).filter(move |&p| (mask & (1u8 << p)) != 0)
}

#[inline]
pub fn after_reveal(mask: u8, p: Piece) -> u8 {
    let bit = 1u8 << p;
    debug_assert!((mask & bit) != 0);
    let rest = mask & !bit;
    if rest == 0 {
        FULL_BAG
    } else {
        rest
    }
}

pub fn mask_to_string(mask: u8) -> String {
    let mut s = String::new();
    for p in pieces(mask) {
        s.push(piece_char(p));
    }
    s
}
