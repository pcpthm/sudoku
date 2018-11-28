#![feature(core_intrinsics, integer_atomics)]

extern crate failure_derive;
extern crate lazy_static;

mod solve;

pub use self::solve::{solve, Solution};

use failure_derive::Fail;
use std::intrinsics::assume;
use std::num::NonZeroU8;
use std::sync::atomic::AtomicU64;

pub static REC_COUNT: AtomicU64 = AtomicU64::new(0);

const DIM1: usize = 3;
const DIM2: usize = DIM1 * DIM1;
const DIM3: usize = DIM1 * DIM1 * DIM1;
const DIM4: usize = DIM2 * DIM2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Digit(NonZeroU8);

impl Digit {
    pub unsafe fn new_unchecked(d: u8) -> Self {
        Digit(NonZeroU8::new_unchecked(d))
    }

    pub fn new(d: u8) -> Self {
        assert!(1 <= d && d as usize <= DIM2, "invalid digit");
        unsafe { Self::new_unchecked(d) }
    }

    pub fn get(self) -> u8 {
        self.0.get()
    }

    pub fn get_index(self) -> usize {
        let d = self.0.get() as usize - 1;
        unsafe {
            assume(d < DIM2);
        }
        d
    }

    pub fn all() -> impl Iterator<Item = Digit> {
        (1..=(DIM2 as u8)).map(|d| unsafe { Digit::new_unchecked(d) })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct SquareIndex(usize);

impl SquareIndex {
    pub unsafe fn new_unchecked(i: usize) -> Self {
        SquareIndex(i)
    }

    pub fn new(i: usize) -> Self {
        assert!(i < DIM4, "invalid index");
        unsafe { Self::new_unchecked(i) }
    }

    pub fn get(self) -> usize {
        let i = self.0;
        unsafe {
            assume(i < DIM4);
        }
        i
    }

    pub fn all() -> impl Iterator<Item = SquareIndex> {
        (0..DIM4).map(|i| unsafe { SquareIndex::new_unchecked(i) })
    }
}

#[derive(Clone)]
pub struct Grid {
    array: [Option<Digit>; DIM4],
}

impl std::ops::Deref for Grid {
    type Target = [Option<Digit>];
    fn deref(&self) -> &Self::Target {
        &self.array
    }
}

impl std::cmp::PartialEq<Grid> for Grid {
    fn eq(&self, other: &Grid) -> bool {
        self as &[Option<Digit>] == other as &[Option<Digit>]
    }
}
impl std::cmp::Eq for Grid {}

impl Grid {
    fn new(array: [Option<Digit>; DIM4]) -> Self {
        Grid { array }
    }

    fn empty() -> Self {
        Self::new([None; DIM4])
    }

    pub fn get(&self, i: SquareIndex) -> Option<Digit> {
        self.array[i.get()]
    }

    pub fn set(&mut self, i: SquareIndex, d: Option<Digit>) {
        self.array[i.get()] = d;
    }

    pub fn squares<'a>(&'a self) -> impl Iterator<Item = Option<Digit>> + 'a {
        self.array[..].iter().cloned()
    }

    pub fn squares_with_indices<'a>(
        &'a self,
    ) -> impl Iterator<Item = (SquareIndex, Option<Digit>)> + 'a {
        SquareIndex::all().map(move |i| (i, self.get(i)))
    }
}

impl std::fmt::Debug for Grid {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self[..], f)
    }
}

impl std::fmt::Display for Grid {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        for d in self.squares() {
            d.map(|d| d.get()).unwrap_or(0).fmt(f)?;
        }
        Ok(())
    }
}

#[derive(Debug, Fail)]
pub enum ProblemParseError {
    #[fail(display = "length must be {} but is {}.", _0, _1)]
    InvalidLength(usize, usize),
    #[fail(display = "invalid character '{}' at position {}.", _1, _0)]
    InvalidCharacter(usize, char),
}

pub fn parse(input: &str) -> Result<Grid, ProblemParseError> {
    if input.len() != DIM4 {
        return Err(ProblemParseError::InvalidLength(DIM4, input.len()));
    }
    let mut array = [None; DIM4];
    for (i, c) in input.chars().enumerate() {
        array[i] = match c.to_digit((DIM2 + 1) as u32) {
            Some(0) => None,
            Some(d) => Some(Digit::new(d as u8)),
            None => return Err(ProblemParseError::InvalidCharacter(i, c)),
        }
    }
    Ok(Grid::new(array))
}
