#![allow(clippy::needless_range_loop, clippy::identity_op)]

use super::{Digit, Grid, SquareIndex, DIM1, DIM2, DIM3, DIM4};
use lazy_static::lazy_static;
use packed_simd::u32x4;
use std::hint::unreachable_unchecked;
use std::ops::{BitAnd, BitOr, BitXor, Not, Shr};

type GridMask = u32x4;
const SQUARE_ALL: u32 = (1u32 << DIM2) - 1;
const BAND_ALL: u32 = (1u32 << DIM3) - 1;
const GRID_NONE: GridMask = GridMask::new(0, 0, 0, 0);
const GRID_ALL: GridMask = GridMask::new(BAND_ALL, BAND_ALL, BAND_ALL, 0);

const UNIT_TYPES: usize = 3;

struct Constants {
    singleton: [GridMask; DIM4],
    all_units: [GridMask; DIM2 * UNIT_TYPES],
    adj: [GridMask; DIM4],
    same_box: [[GridMask; DIM2]; DIM1],
    same_row: [[GridMask; DIM2]; DIM2],
}

fn check_bit(mask: GridMask, i: SquareIndex) -> bool {
    (mask & MASK.singleton[i.get()]) != GRID_NONE
}

fn count_ones(mask: GridMask) -> u32 {
    mask.extract(0).count_ones() + mask.extract(1).count_ones() + mask.extract(2).count_ones()
}

fn is_valid_mask(mask: GridMask) -> bool {
    mask.extract(0) & !BAND_ALL == 0
        && mask.extract(1) & !BAND_ALL == 0
        && mask.extract(2) & !BAND_ALL == 0
        && mask.extract(3) == 0
}

struct GridMaskIter {
    mask: GridMask,
    i: usize,
}

fn iter_mask_indices(mask: GridMask) -> GridMaskIter {
    GridMaskIter { mask, i: 0 }
}

impl Iterator for GridMaskIter {
    type Item = SquareIndex;
    fn next(&mut self) -> Option<Self::Item> {
        while self.i < 3 {
            let part = unsafe { self.mask.extract_unchecked(self.i) };
            if part != 0 {
                self.mask = unsafe { self.mask.replace_unchecked(self.i, part & (part - 1)) };
                let k = part.trailing_zeros() as usize;
                return Some(unsafe { SquareIndex::new_unchecked(DIM3 * self.i + k) });
            }
            self.i += 1;
        }
        None
    }
}

impl GridMaskIter {
    unsafe fn next_unchecked(&mut self) -> SquareIndex {
        debug_assert!(self.mask != GRID_NONE);
        self.next().unwrap_or_else(|| unreachable_unchecked())
    }
}

impl Constants {
    pub fn new() -> Self {
        let mut singleton = [GRID_NONE; DIM4];
        for i in 0..DIM4 {
            let band = i / DIM3;
            let bitpos = i % DIM3;
            singleton[i] = singleton[i].replace(band, 1u32 << bitpos);
        }

        let mut cols = [GRID_NONE; DIM2];
        let mut rows = [GRID_NONE; DIM2];
        let mut boxes = [[GRID_NONE; DIM1]; DIM1];

        for col in 0..DIM2 {
            cols[col] = (0..DIM2)
                .map(|row| singleton[row * DIM2 + col])
                .fold(GRID_NONE, BitOr::bitor);
        }
        for row in 0..DIM2 {
            rows[row] = (0..DIM2)
                .map(|col| singleton[row * DIM2 + col])
                .fold(GRID_NONE, BitOr::bitor);
        }
        for band in 0..DIM1 {
            for stack in 0..DIM1 {
                boxes[band][stack] = (0..DIM1)
                    .flat_map(|sr| {
                        (0..DIM1).map(move |sc| {
                            singleton[(band * DIM1 + sr) * DIM2 + (stack * DIM1 + sc)]
                        })
                    })
                    .fold(GRID_NONE, BitOr::bitor);
            }
        }

        let mut all_units = [GRID_NONE; DIM2 * UNIT_TYPES];
        for i in 0..DIM2 {
            all_units[i] = cols[i];
            all_units[DIM2 + i] = rows[i];
            all_units[DIM2 * 2 + i] = boxes[i / DIM1][i % DIM1];
        }

        let mut adj = [GRID_NONE; DIM4];
        for y in 0..DIM2 {
            for x in 0..DIM2 {
                adj[y * DIM2 + x] = rows[y] | cols[x] | boxes[y / 3][x / 3];
            }
        }

        let mut same_box = [[GRID_NONE; DIM2]; DIM1];
        let mut same_row = [[GRID_NONE; DIM2]; DIM2];

        for d in 0..DIM2 {
            for stack in 0..DIM1 {
                same_box[stack][d] = (0..DIM1)
                    .map(|sc| singleton[(stack * DIM1 + sc) * DIM2 + d])
                    .fold(GRID_NONE, BitOr::bitor);
            }
            let row_mask = (0..DIM1)
                .map(|s| same_box[s][d])
                .fold(GRID_NONE, BitOr::bitor);
            for col in 0..DIM2 {
                same_row[col][d] = row_mask
                    | (0..DIM2)
                        .map(|d2| singleton[col * DIM2 + d2])
                        .fold(GRID_NONE, BitOr::bitor);
            }
        }

        Constants {
            singleton,
            all_units,
            adj,
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
    let singleton = constants.singleton[i.get()];
    *unsolved_mask &= !singleton;
    for r in digit_masks.iter_mut() {
        *r &= !singleton;
    }
    digit_masks[d.get_index()] &= !constants.adj[i.get()];
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

    let same_col = *unsafe { constants.singleton.get_unchecked(col * DIM2 + d) };
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

trait BitParallelCounter: Sized + Copy {
    type T;
    fn new(x: Self::T, y: Self::T, z: Self::T) -> Self;
    fn compute(x: Self, y: Self, z: Self) -> Self::T;
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

impl<T: BitNum> BitParallelCounter for FindSingle<T> {
    type T = T;
    fn new(x: T, y: T, z: T) -> Self {
        FindSingle((x ^ y ^ z) & !(x & y & z), x | y | z)
    }

    fn compute(x: FindSingle<T>, y: FindSingle<T>, z: FindSingle<T>) -> T {
        x.0 & !(y.1 | z.1) | y.0 & !(x.1 | z.1) | z.0 & !(x.1 | y.1)
    }
}
impl<T: Shr<u32, Output = T>> Shr<u32> for FindSingle<T> {
    type Output = Self;
    fn shr(self, rhs: u32) -> Self::Output {
        FindSingle(self.0 >> rhs, self.1 >> rhs)
    }
}

#[derive(Debug, Clone, Copy)]
struct FindPair<T>(T, T, T);

impl<T: BitNum> BitParallelCounter for FindPair<T> {
    type T = T;
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
impl<T: Shr<u32, Output = T>> Shr<u32> for FindPair<T> {
    type Output = Self;
    fn shr(self, rhs: u32) -> Self::Output {
        FindPair(self.0 >> rhs, self.1 >> rhs, self.2 >> rhs)
    }
}

fn col_and_digit(i: SquareIndex) -> (usize, Digit) {
    (i.get() / DIM2, unsafe {
        Digit::new_unchecked((i.get() % DIM2) as u8 + 1)
    })
}

fn find_hidden_col<Find: BitParallelCounter<T = GridMask>>(
    square_masks: &[GridMask; DIM2],
    unsolved_cols: &GridMask,
) -> Option<(usize, Digit)> {
    let get = |x, y, z| Find::new(square_masks[x], square_masks[y], square_masks[z]);
    let mask = *unsolved_cols & Find::compute(get(0, 1, 2), get(3, 4, 5), get(6, 7, 8));
    if mask != GRID_NONE {
        return Some(col_and_digit(unsafe {
            iter_mask_indices(mask).next_unchecked()
        }));
    }
    None
}

fn find_hidden_row<Find: BitParallelCounter<T = u32>>(
    square_masks: &[GridMask; DIM2],
    unsolved_rows: &[u16; DIM2],
) -> Option<(usize, Digit)>
where
    Find: Shr<u32, Output = Find>,
{
    for row in 0..DIM2 {
        let row_mask = square_masks[row];
        let find = Find::new(
            row_mask.extract(0),
            row_mask.extract(1),
            row_mask.extract(2),
        );
        let mask = unsolved_rows[row]
            & Find::compute(find, find >> DIM2 as u32, find >> (2 * DIM2) as u32) as u16;
        if mask != 0 {
            return Some((DIM2 + row, unsafe {
                Digit::new_unchecked(mask.trailing_zeros() as u8 + 1)
            }));
        }
    }

    None
}

fn find_hidden_box<Find: BitParallelCounter<T = GridMask>>(
    square_masks: &[GridMask; DIM2],
    unsolved_boxes: &[u16; DIM2],
) -> Option<(usize, Digit)>
where
    Find: Shr<u32, Output = Find>,
{
    for band in 0..DIM1 {
        let x = Find::new(
            square_masks[band * DIM1 + 0],
            square_masks[band * DIM1 + 1],
            square_masks[band * DIM1 + 2],
        );
        let y = Find::compute(x, x >> DIM2 as u32, x >> (2 * DIM2) as u32);
        for stack in 0..DIM1 {
            let boxi = band * DIM1 + stack;
            let mask = unsolved_boxes[boxi] & (y.extract(stack) as u16);
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
        check_bit(self.digit_mask(d), i)
    }

    pub fn square_mask(&self, i: SquareIndex) -> u32 {
        let row = i.get() / DIM2;
        let col = i.get() % DIM2;
        let stack = col / DIM1;
        let subcol = col % DIM1;
        self.square_masks[row].extract(stack) >> (subcol * DIM2) & SQUARE_ALL
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

        debug_assert!(is_valid_mask(self.unsolved_mask));
        for mask in &self.digit_masks {
            debug_assert!(is_valid_mask(*mask));
        }
        for mask in &self.square_masks {
            debug_assert!(is_valid_mask(*mask));
        }
        #[cfg(debug)]
        for i in SquareIndex::all() {
            if !check_bit(self.unsolved_mask, i) {
                let mask = self.square_mask(i);
                debug_assert!(mask.count_ones() == 1);
            } else {
                for d in Digit::all() {
                    let a = check_bit(self.digit_mask(d), i);
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
                debug_assert!(count_ones(mask) == 1);
                let i = unsafe { iter_mask_indices(mask).next_unchecked() };
                (i, d)
            })
    }

    pub fn find_hidden_pair(&self) -> Option<[(SquareIndex, Digit); 2]> {
        find_hidden_col::<FindPair<_>>(&self.square_masks, &self.unsolved_cols)
            .or_else(|| find_hidden_row::<FindPair<_>>(&self.square_masks, &self.unsolved_rows))
            .or_else(|| find_hidden_box::<FindPair<_>>(&self.square_masks, &self.unsolved_boxes))
            .map(|(unit, d)| {
                let mask = self.digit_mask(d) & MASK.all_units[unit];
                debug_assert!(count_ones(mask) == 2);
                let mut iter = iter_mask_indices(mask);
                let i = unsafe { iter.next_unchecked() };
                let j = unsafe { iter.next_unchecked() };
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
    if b != GRID_NONE {
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

    if unsolved_squares == GRID_NONE {
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
        if mask == GRID_NONE {
            return Some(false);
        }
        for i in iter_mask_indices(mask) {
            let square_mask = state.square_mask(i);
            if square_mask == 0 {
                return None;
            }
            let d = unsafe { Digit::new_unchecked(square_mask.trailing_zeros() as u8 + 1) };
            self.put(state, i, d);
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
            if digit_mask == GRID_NONE {
                continue;
            }
            for unit_mask in all_units.iter().cloned() {
                let mask = digit_mask & unit_mask;
                if mask == GRID_NONE {
                    continue;
                }
                let count = count_ones(mask);
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
