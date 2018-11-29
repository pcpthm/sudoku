#![allow(clippy::needless_range_loop, clippy::identity_op)]

use super::{Digit, Grid, SquareIndex, DIM1, DIM2, DIM4};
use core::arch::x86_64::{__m128i, _mm_and_si128, _mm_or_si128, _mm_xor_si128};
use lazy_static::lazy_static;
use std::mem::transmute;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, Not, Shr};

#[repr(align(16))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GridMask(u128);

impl GridMask {
    pub const fn new(value: u128) -> Self {
        GridMask(value)
    }
    pub fn is_zero(self) -> bool {
        self.0 == 0
    }
    pub fn get(self) -> u128 {
        self.0
    }
    pub fn has_bit(self, i: SquareIndex) -> bool {
        self.0 >> i.get() & 1 != 0
    }
    pub fn into_u64_parts(self) -> [u64; 2] {
        unsafe { transmute::<u128, [u64; 2]>(self.0) }
    }
    pub fn as_u64_parts_mut(&mut self) -> &mut [u64; 2] {
        unsafe { &mut *(&mut self.0 as *mut u128 as *mut [u64; 2]) }
    }
    pub fn count_ones(self) -> u32 {
        let [x, y] = self.into_u64_parts();
        x.count_ones() + y.count_ones()
    }
    pub fn is_single(self) -> bool {
        self.count_ones() == 1
    }
}

impl From<u128> for GridMask {
    fn from(value: u128) -> Self {
        GridMask::new(value)
    }
}

impl From<__m128i> for GridMask {
    fn from(value: __m128i) -> Self {
        GridMask(unsafe { transmute(value) })
    }
}

impl From<GridMask> for __m128i {
    fn from(value: GridMask) -> Self {
        unsafe { transmute(value.0) }
    }
}

impl Not for GridMask {
    type Output = Self;
    fn not(self) -> Self {
        (!self.get()).into()
    }
}
impl BitAnd for GridMask {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self {
        unsafe { _mm_and_si128(self.into(), rhs.into()).into() }
    }
}
impl BitAndAssign for GridMask {
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}
impl BitOr for GridMask {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        unsafe { _mm_or_si128(self.into(), rhs.into()).into() }
    }
}
impl BitOrAssign for GridMask {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}
impl BitXor for GridMask {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self {
        unsafe { _mm_xor_si128(self.into(), rhs.into()).into() }
    }
}
impl Shr<usize> for GridMask {
    type Output = Self;
    fn shr(self, rhs: usize) -> Self {
        Self::new(self.get() >> rhs)
    }
}

const GRID_ALL: GridMask = GridMask::new((1u128 << DIM4) - 1);

struct MaskIter(u128);

fn iter_mask_indices(mask: GridMask) -> impl Iterator<Item = SquareIndex> {
    MaskIter(mask.get() & GRID_ALL.get())
}

impl Iterator for MaskIter {
    type Item = SquareIndex;
    fn next(&mut self) -> Option<Self::Item> {
        if self.0 == 0 {
            None
        } else {
            let i = unsafe { SquareIndex::new_unchecked(self.0.trailing_zeros() as usize) };
            self.0 &= self.0 - 1;
            Some(i)
        }
    }
}

const UNIT_TYPES: usize = 3;

struct Constants {
    all_units: [GridMask; DIM2 * UNIT_TYPES],
    adj: [GridMask; DIM4],
    same_col: [[GridMask; DIM2]; DIM2],
    same_box: [[GridMask; DIM2]; DIM1],
    same_row: [[GridMask; DIM2]; DIM2],
}

impl Constants {
    pub fn new() -> Self {
        let mut cols = [GridMask::default(); DIM2];
        let mut rows = [GridMask::default(); DIM2];
        let mut boxes = [[GridMask::default(); DIM1]; DIM1];

        let c = (0..DIM2)
            .map(|i| 1u128 << (i * DIM2))
            .fold(0u128, BitOr::bitor);
        let r = (1u128 << DIM2) - 1;
        let b = (0..DIM1)
            .flat_map(|y| (0..DIM1).map(move |x| 1u128 << (y * DIM2 + x)))
            .fold(0u128, BitOr::bitor);
        for i in 0..DIM2 {
            cols[i] = (c << i).into();
            rows[i] = (r << (i * DIM2)).into();
        }
        for y in 0..DIM1 {
            for x in 0..DIM1 {
                boxes[y][x] = (b << ((y * DIM2 + x) * DIM1)).into();
            }
        }

        let mut all_units = [GridMask::default(); DIM2 * UNIT_TYPES];
        for i in 0..DIM2 {
            all_units[i] = cols[i];
            all_units[DIM2 + i] = rows[i];
            all_units[DIM2 * 2 + i] = boxes[i / DIM1][i % DIM1];
        }

        let mut adj = [GridMask::default(); DIM4];
        for y in 0..DIM2 {
            for x in 0..DIM2 {
                adj[y * DIM2 + x] = rows[y] | cols[x] | boxes[y / 3][x / 3];
            }
        }

        let mut same_col = [[GridMask::default(); DIM2]; DIM2];
        let mut same_box = [[GridMask::default(); DIM2]; DIM1];
        let mut same_row = [[GridMask::default(); DIM2]; DIM2];

        for d in 0..DIM2 {
            for col in 0..DIM2 {
                same_col[col][d] = GridMask::new(1u128 << (col * DIM2 + d));
            }
            for stack in 0..DIM1 {
                same_box[stack][d] = (0..DIM1)
                    .map(|sc| same_col[stack * DIM1 + sc][d])
                    .fold(GridMask::default(), BitOr::bitor);
            }
            let row_mask = (0..DIM1)
                .map(|s| same_box[s][d])
                .fold(GridMask::default(), BitOr::bitor);
            for col in 0..DIM2 {
                same_row[col][d] = row_mask | GridMask::new(((1u128 << DIM2) - 1) << (col * DIM2));
            }
        }

        Constants {
            all_units,
            adj,
            same_col,
            same_box,
            same_row,
        }
    }
}

lazy_static! {
    static ref MASK: Constants = Constants::new();
}

#[derive(Debug, Clone)]
pub struct State {
    // (digit_masks[d - 1] >> i & 1): Is digit d a candidate of square i?
    // If a square is solved, there are no remaining candidates for the square.
    digit_masks: [GridMask; DIM2],

    // mask of unsolved squares
    unsolved_mask: GridMask,

    square_masks: [GridMask; DIM2],
    unsolved_cols: GridMask,
    unsolved_rows: [u16; DIM2],
    unsolved_boxes: [u16; DIM2],
}

fn update_digit_masks(
    i: SquareIndex,
    d: Digit,
    digit_masks: &mut [GridMask; DIM2],
    unsolved_mask: &mut GridMask,
    constants: &Constants,
) {
    digit_masks[d.get_index()] &= !constants.adj[i.get()];

    let mut f = move |b: usize| {
        let mask = !(1u64 << (i.get() - b * 64));
        for m in &mut digit_masks[..] {
            m.as_u64_parts_mut()[b] &= mask;
        }
        unsolved_mask.as_u64_parts_mut()[b] &= mask;
    };

    if i.get() < 64 {
        f(0)
    } else {
        f(1)
    }
}

fn update_square_masks(
    i: SquareIndex,
    d: Digit,
    square_masks: &mut [GridMask; DIM2],
    unsolved_cols: &mut GridMask,
    unsolved_rows: &mut [u16; DIM2],
    unsolved_boxes: &mut [u16; DIM2],
    constants: &Constants,
) {
    const DIV3: [u8; 9] = [0, 0, 0, 1, 1, 1, 2, 2, 2];
    let d = d.get_index();
    let row = i.get() / DIM2;
    let col = i.get() % DIM2;
    let band = DIV3[row] as usize;
    let stack = DIV3[col] as usize;

    let same_col = *unsafe { constants.same_col.get_unchecked(col).get_unchecked(d) };
    let same_box = *unsafe { constants.same_box.get_unchecked(stack).get_unchecked(d) };
    let same_row = *unsafe { constants.same_row.get_unchecked(col).get_unchecked(d) };

    *unsafe { unsolved_rows.get_unchecked_mut(row) } &= !(1u16 << d);
    *unsolved_cols &= !same_col;
    *unsafe { unsolved_boxes.get_unchecked_mut(band * DIM1 + stack) } &= !(1u16 << d);

    let mut masks = [same_col; 3];
    *unsafe { masks.get_unchecked_mut(band) } = same_box;
    for r in 0..DIM2 {
        square_masks[r] &= !masks[r / DIM1];
    }
    square_masks[row] = square_masks[row] & !same_row | same_col;
}

trait BitParallelCounter<T>: Sized + Copy {
    fn new(x: T, y: T, z: T) -> Self;
    fn compute(x: Self, y: Self, z: Self) -> T;
}

trait BitNum:
    Copy
    + BitAnd<Self, Output = Self>
    + BitOr<Self, Output = Self>
    + BitXor<Self, Output = Self>
    + Not<Output = Self>
{
}
impl BitNum for u16 {}
impl BitNum for u32 {}
impl BitNum for GridMask {}

#[derive(Debug, Clone, Copy)]
struct FindSingle<T>(T, T);

impl<T: Shr<usize, Output = T>> Shr<usize> for FindSingle<T> {
    type Output = Self;
    fn shr(self, rhs: usize) -> Self::Output {
        FindSingle(self.0 >> rhs, self.1 >> rhs)
    }
}

impl<T: BitNum> BitParallelCounter<T> for FindSingle<T> {
    fn new(x: T, y: T, z: T) -> Self {
        FindSingle((x ^ y ^ z) & !(x & y & z), x | y | z)
    }

    fn compute(x: FindSingle<T>, y: FindSingle<T>, z: FindSingle<T>) -> T {
        x.0 & !(y.1 | z.1) | y.0 & !(x.1 | z.1) | z.0 & !(x.1 | y.1)
    }
}

#[derive(Debug, Clone, Copy)]
struct FindPair<T>(T, T, T);

impl<T: Shr<usize, Output = T>> Shr<usize> for FindPair<T> {
    type Output = Self;
    fn shr(self, rhs: usize) -> Self::Output {
        FindPair(self.0 >> rhs, self.1 >> rhs, self.2 >> rhs)
    }
}

impl<T: BitNum> BitParallelCounter<T> for FindPair<T> {
    fn new(x: T, y: T, z: T) -> Self {
        FindPair(
            (x ^ y ^ z) & !(x & y & z),
            x | y | z,
            !(x ^ y ^ z) & (x | y | z),
        )
    }

    fn compute(x: FindPair<T>, y: FindPair<T>, z: FindPair<T>) -> T {
        let parts = FindPair::compute_parts(x, y, z);
        parts.0 | parts.1
    }
}

impl<T: BitNum> FindPair<T> {
    pub fn compute_parts(x: FindPair<T>, y: FindPair<T>, z: FindPair<T>) -> (T, T) {
        (
            x.0 & y.0 & !z.1 | x.0 & z.0 & !y.1 | y.0 & z.0 & !x.1,
            x.2 & !(y.1 | z.1) | y.2 & !(x.1 | z.1) | z.2 & !(x.1 | y.1),
        )
    }
}

fn find_hidden_col<Find: BitParallelCounter<GridMask>>(
    square_masks: &[GridMask; DIM2],
    unsolved_cols: &GridMask,
) -> Option<(usize, Digit)> {
    let get = |x, y, z| Find::new(square_masks[x], square_masks[y], square_masks[z]);
    let mask = *unsolved_cols & Find::compute(get(0, 1, 2), get(3, 4, 5), get(6, 7, 8));
    if !mask.is_zero() {
        let k = mask.get().trailing_zeros() as usize;
        let col = k / DIM2;
        return Some((col, unsafe { Digit::new_unchecked((k % DIM2) as u8 + 1) }));
    }
    None
}

fn find_hidden_row<Find: BitParallelCounter<u32>>(
    square_masks: &[GridMask; DIM2],
    unsolved_rows: &[u16; DIM2],
) -> Option<(usize, Digit)>
where
    Find: Shr<usize, Output = Find>,
{
    for row in 0..DIM2 {
        let row_mask = square_masks[row].get();
        let find = Find::new(
            (row_mask >> (0 * DIM1 * DIM2)) as u32,
            (row_mask >> (1 * DIM1 * DIM2)) as u32,
            (row_mask >> (2 * DIM1 * DIM2)) as u32,
        );
        let mask =
            unsolved_rows[row] & Find::compute(find, find >> DIM2, find >> (2 * DIM2)) as u16;
        if mask != 0 {
            return Some((DIM2 + row, unsafe {
                Digit::new_unchecked(mask.trailing_zeros() as u8 + 1)
            }));
        }
    }

    None
}

fn find_hidden_box<Find: BitParallelCounter<GridMask>>(
    square_masks: &[GridMask; DIM2],
    unsolved_boxes: &[u16; DIM2],
) -> Option<(usize, Digit)>
where
    Find: Shr<usize, Output = Find>,
{
    for band in 0..DIM1 {
        let x = Find::new(
            square_masks[band * DIM1 + 0],
            square_masks[band * DIM1 + 1],
            square_masks[band * DIM1 + 2],
        );
        let y = Find::compute(x, x >> DIM2, x >> (2 * DIM2));
        for stack in 0..DIM1 {
            let boxi = band * DIM1 + stack;
            let mask = unsolved_boxes[boxi] & ((y.get() >> (stack * DIM1 * DIM2)) as u16);
            if mask != 0 {
                return Some((DIM2 * 2 + boxi, unsafe {
                    Digit::new_unchecked(mask.trailing_zeros() as u8 + 1)
                }));
            }
        }
    }

    None
}

impl State {
    pub fn new() -> Self {
        State {
            digit_masks: [GRID_ALL; DIM2],
            unsolved_mask: GRID_ALL,
            square_masks: [GRID_ALL; DIM2],
            unsolved_cols: GRID_ALL,
            unsolved_rows: [(1u16 << DIM2) - 1; DIM2],
            unsolved_boxes: [(1u16 << DIM2) - 1; DIM2],
        }
    }

    pub fn digit_mask(&self, d: Digit) -> GridMask {
        self.digit_masks[d.get_index()]
    }

    pub fn unsolved_mask(&self) -> GridMask {
        self.unsolved_mask
    }

    pub fn is_candidate(&self, i: SquareIndex, d: Digit) -> bool {
        self.digit_mask(d).has_bit(i)
    }

    pub fn square_mask(&self, i: SquareIndex) -> u32 {
        let row = i.get() / DIM2;
        let col = i.get() % DIM2;
        (self.square_masks[row].get() >> (col * DIM2)) as u32 & ((1u32 << DIM2) - 1)
    }

    pub fn assign(&mut self, i: SquareIndex, d: Digit) {
        // eprintln!("\nassign({:?}, {:?})", i, d);
        debug_assert!(self.is_candidate(i, d));
        let constants: &Constants = &MASK;
        update_digit_masks(
            i,
            d,
            &mut self.digit_masks,
            &mut self.unsolved_mask,
            constants,
        );
        update_square_masks(
            i,
            d,
            &mut self.square_masks,
            &mut self.unsolved_cols,
            &mut self.unsolved_rows,
            &mut self.unsolved_boxes,
            constants,
        );

        for i in SquareIndex::all() {
            if !self.unsolved_mask.has_bit(i) {
                let mask = self.square_mask(i);
                debug_assert!(mask.count_ones() == 1);
            } else {
                for d in Digit::all() {
                    let a = self.digit_mask(d).has_bit(i);
                    let b = self.square_mask(i) >> d.get_index() & 1 != 0;
                    debug_assert!(a == b);
                }
            }
        }
    }

    pub fn check_square_masks(&self) -> bool {
        let mut total = GridMask::default();
        for row_mask in &self.square_masks {
            total |= *row_mask;
        }
        total == GRID_ALL
    }

    pub fn find_hidden_single(&self) -> Option<(SquareIndex, Digit)> {
        find_hidden_col::<FindSingle<_>>(&self.square_masks, &self.unsolved_cols)
            .or_else(|| find_hidden_row::<FindSingle<_>>(&self.square_masks, &self.unsolved_rows))
            .or_else(|| find_hidden_box::<FindSingle<_>>(&self.square_masks, &self.unsolved_boxes))
            .map(|(unit, d)| {
                let mask = self.digit_mask(d) & MASK.all_units[unit];
                debug_assert!(mask.count_ones() == 1);
                let i = unsafe { SquareIndex::new_unchecked(mask.get().trailing_zeros() as usize) };
                (i, d)
            })
    }

    pub fn find_hidden_pair(&self) -> Option<[(SquareIndex, Digit); 2]> {
        find_hidden_col::<FindPair<_>>(&self.square_masks, &self.unsolved_cols)
            .or_else(|| find_hidden_row::<FindPair<_>>(&self.square_masks, &self.unsolved_rows))
            .or_else(|| find_hidden_box::<FindPair<_>>(&self.square_masks, &self.unsolved_boxes))
            .map(|(unit, d)| {
                let mask = self.digit_mask(d) & MASK.all_units[unit];
                debug_assert!(mask.count_ones() == 2);
                let [x, y] = mask.into_u64_parts();
                let (i, j) = if x != 0 {
                    let i = unsafe { SquareIndex::new_unchecked(x.trailing_zeros() as usize) };
                    let x = x & (x - 1);
                    (
                        i,
                        if x != 0 {
                            unsafe { SquareIndex::new_unchecked(x.trailing_zeros() as usize) }
                        } else {
                            unsafe { SquareIndex::new_unchecked(y.trailing_zeros() as usize + 64) }
                        },
                    )
                } else {
                    let i = unsafe { SquareIndex::new_unchecked(y.trailing_zeros() as usize + 64) };
                    let y = y & (y - 1);
                    (i, unsafe {
                        SquareIndex::new_unchecked(y.trailing_zeros() as usize + 64)
                    })
                };
                [(i, d), (j, d)]
            })
    }
}

pub fn find_naked_singles(state: &State) -> GridMask {
    let get = |x, y, z| {
        FindSingle::new(
            state.digit_mask(Digit::new(x)),
            state.digit_mask(Digit::new(y)),
            state.digit_mask(Digit::new(z)),
        )
    };
    FindSingle::compute(get(1, 2, 3), get(4, 5, 6), get(7, 8, 9))
}

pub fn find_naked_pair(state: &State) -> Option<SquareIndex> {
    let get = |x, y, z| {
        FindPair::new(
            state.digit_mask(Digit::new(x)),
            state.digit_mask(Digit::new(y)),
            state.digit_mask(Digit::new(z)),
        )
    };
    let (a, b) = FindPair::compute_parts(get(1, 2, 3), get(4, 5, 6), get(7, 8, 9));
    if !b.is_zero() {
        iter_mask_indices(b).next()
    } else {
        iter_mask_indices(a).next()
    }
}

#[derive(Debug)]
enum SolutionState {
    Invalid,
    Solved,
    Unsolved,
}

fn solution_state(state: &State) -> SolutionState {
    let squares_span = Digit::all()
        .map(|d| state.digit_mask(d))
        .fold(GridMask::default(), BitOr::bitor);
    let unsolved_squares = state.unsolved_mask();
    if squares_span != unsolved_squares {
        return SolutionState::Invalid;
    }

    if !state.check_square_masks() {
        return SolutionState::Invalid;
    }

    if unsolved_squares.is_zero() {
        SolutionState::Solved
    } else {
        SolutionState::Unsolved
    }
}

struct Solver {
    partial: Grid,
    solution: Option<Grid>,
    solution_count: u64,
}

impl Solver {
    pub fn new() -> Self {
        Solver {
            partial: Grid::empty(),
            solution: None,
            solution_count: 0,
        }
    }

    fn put(&mut self, state: &mut State, i: SquareIndex, d: Digit) {
        state.assign(i, d);
        self.partial.set(i, Some(d))
    }

    fn add_solved(&mut self) {
        if self.solution.is_none() {
            self.solution = Some(self.partial.clone());
        }
        self.solution_count += 1;
    }

    pub fn apply_hints(&mut self, state: &mut State, grid: &Grid) -> bool {
        for (i, hint) in grid.squares_with_indices() {
            if let Some(d) = hint {
                if !state.is_candidate(i, d) {
                    return false;
                }
                self.put(state, i, d);
            }
        }
        true
    }

    fn apply_naked_singles(&mut self, state: &mut State, mask: GridMask) -> Option<bool> {
        if mask.is_zero() {
            return Some(false);
        }
        for (b, part) in mask.into_u64_parts().iter_mut().enumerate() {
            while *part != 0 {
                let lsb = *part & (!*part + 1);
                *part ^= lsb;
                let i =
                    unsafe { SquareIndex::new_unchecked(lsb.trailing_zeros() as usize + b * 64) };
                let square_mask = state.square_mask(i);
                if square_mask == 0 {
                    return None;
                }
                let d = unsafe { Digit::new_unchecked(square_mask.trailing_zeros() as u8 + 1) };
                self.put(state, i, d);
            }
        }
        Some(true)
    }

    fn branch_on_pair(&mut self, mut state: State, pair: [(SquareIndex, Digit); 2]) {
        let mut clone = state.clone();
        self.put(&mut clone, pair[0].0, pair[0].1);
        self.solve_dfs(clone);

        self.put(&mut state, pair[1].0, pair[1].1);
        self.solve_dfs(state)
    }

    fn branch(&mut self, state: State) {
        if let Some(i) = find_naked_pair(&state) {
            let mask = state.square_mask(i);
            let d0 = unsafe { Digit::new_unchecked(mask.trailing_zeros() as u8 + 1) };
            let mask = mask & (mask - 1);
            let d1 = unsafe { Digit::new_unchecked(mask.trailing_zeros() as u8 + 1) };
            return self.branch_on_pair(state, [(i, d0), (i, d1)]);
        }

        if let Some(pair) = state.find_hidden_pair() {
            return self.branch_on_pair(state, pair);
        }

        eprintln!("No pairs found!");

        let mut min = (10, Digit::new(1), GridMask::default());

        let all_units = &MASK.all_units;

        'outer: for d in Digit::all() {
            let digit_mask = state.digit_mask(d);
            if digit_mask.is_zero() {
                continue;
            }
            for unit_mask in all_units.iter().cloned() {
                let mask = digit_mask & unit_mask;
                if mask.is_zero() {
                    continue;
                }
                let count = mask.count_ones();
                if count < min.0 {
                    min = (count, d, mask);
                    if count <= 2 {
                        break 'outer;
                    }
                }
            }
        }

        for i in iter_mask_indices(min.2) {
            let mut clone = state.clone();
            self.put(&mut clone, i, min.1);
            self.solve_dfs(clone);
        }
    }

    fn solve_dfs(&mut self, mut state: State) {
        super::REC_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        loop {
            match solution_state(&state) {
                SolutionState::Invalid => return,
                SolutionState::Solved => return self.add_solved(),
                SolutionState::Unsolved => {}
            }

            let naked_singles = find_naked_singles(&state);
            match self.apply_naked_singles(&mut state, naked_singles) {
                None => return,
                Some(true) => continue,
                Some(false) => {}
            }

            if let Some((i, d)) = state.find_hidden_single() {
                self.put(&mut state, i, d);
                continue;
            }

            return self.branch(state);
        }
    }

    pub fn solve(&mut self, state: State) -> Solution {
        self.solution = None;
        self.solution_count = 0;

        self.solve_dfs(state);

        match self.solution.take() {
            None => Solution::No,
            Some(sol) => match self.solution_count {
                0 => unreachable!(),
                1 => Solution::One(sol),
                n => Solution::Multiple(Some(n)),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Solution {
    No,
    One(Grid),
    Multiple(Option<u64>),
}

pub fn solve(problem: &Grid) -> Solution {
    let mut state = State::new();
    let mut solver = Solver::new();
    if !solver.apply_hints(&mut state, problem) {
        return Solution::No;
    }
    solver.solve(state)
}

#[cfg(test)]
mod test {
    use super::*;

    fn parse(s: &str) -> Grid {
        super::super::parse(s).unwrap()
    }

    #[test]
    fn solve_problems() {
        let problem =
            "000000000000000001001023040000500020002041600070000000004036702060050030800900060";
        let solution =
            "245617893638495271791823546416589327382741659579362418954136782167258934823974165";
        assert_eq!(solve(&parse(problem)), Solution::One(parse(solution)));
    }
}
