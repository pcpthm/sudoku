use super::{Digit, Grid, SquareIndex, DIM1, DIM2, DIM4};
use lazy_static::lazy_static;

type GridMask = u128;

const MASK_ALL: GridMask = (1u128 << DIM4) - 1;

struct MaskIter(GridMask);

fn iter_mask_indices(mask: GridMask) -> impl Iterator<Item = SquareIndex> {
    MaskIter(mask & MASK_ALL)
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

struct MaskConstants {
    rows: [GridMask; DIM2],
    cols: [GridMask; DIM2],
    boxes: [[GridMask; DIM1]; DIM1],
    adj: [GridMask; DIM4],
}

impl MaskConstants {
    pub fn new() -> Self {
        let mut rows = [0u128; DIM2];
        let mut cols = [0u128; DIM2];
        let mut boxes = [[0u128; DIM1]; DIM1];
        let mut adj = [0u128; DIM4];

        let r = (1u128 << DIM2) - 1;
        let c = (0..DIM2)
            .map(|i| 1u128 << (i * DIM2))
            .fold(0u128, std::ops::BitOr::bitor);
        let b = (0..DIM1)
            .flat_map(|y| (0..DIM1).map(move |x| 1u128 << (y * DIM2 + x)))
            .fold(0u128, std::ops::BitOr::bitor);
        for i in 0..DIM2 {
            rows[i] = r << (i * DIM2);
            cols[i] = c << i;
        }
        #[allow(clippy::needless_range_loop)]
        for y in 0..DIM1 {
            for x in 0..DIM1 {
                boxes[y][x] = b << ((y * DIM2 + x) * DIM1);
            }
        }

        for y in 0..DIM2 {
            for x in 0..DIM2 {
                adj[y * DIM2 + x] = rows[y] | cols[x] | boxes[y / 3][x / 3];
            }
        }

        MaskConstants {
            rows,
            cols,
            boxes,
            adj,
        }
    }
}

lazy_static! {
    static ref MASK: MaskConstants = MaskConstants::new();
}

#[derive(Debug, Clone)]
struct State {
    // (digit_masks[d - 1] >> i & 1): Is digit d a candidate of square i?
    // If a square is solved, there are no remaining candidates for the square.
    digit_masks: [GridMask; DIM2],

    // mask of unsolved squares
    unsolved_mask: GridMask,
}

impl State {
    pub fn new() -> Self {
        State {
            digit_masks: [MASK_ALL; DIM2],
            unsolved_mask: MASK_ALL,
        }
    }

    pub fn digit_mask(&self, d: Digit) -> GridMask {
        self.digit_masks[d.get() as usize - 1]
    }

    pub fn digit_mask_mut(&mut self, d: Digit) -> &mut GridMask {
        &mut self.digit_masks[d.get() as usize - 1]
    }

    pub fn unsolved_mask(&self) -> GridMask {
        self.unsolved_mask
    }

    pub fn is_candidate(&self, i: SquareIndex, d: Digit) -> bool {
        self.digit_mask(d) >> i.get() & 1 != 0
    }

    pub fn assign(&mut self, i: SquareIndex, d: Digit) {
        debug_assert!(self.is_candidate(i, d));
        *self.digit_mask_mut(d) &= !MASK.adj[i.get()];
        let mask = !(1u128 << i.get());
        for m in &mut self.digit_masks[..] {
            *m &= mask;
        }
        self.unsolved_mask &= mask;
    }
}

fn unit_masks() -> impl Iterator<Item = &'static GridMask> {
    MASK.rows
        .iter()
        .chain(MASK.cols.iter())
        .chain(MASK.boxes.iter().flatten())
}

fn find_hidden_single(state: &State) -> Option<(SquareIndex, Digit)> {
    for unit_mask in unit_masks() {
        for d in Digit::all() {
            let mask = state.digit_mask(d) & unit_mask;
            if mask != 0 && mask & (mask - 1) == 0 {
                let i = iter_mask_indices(mask).next().unwrap();
                return Some((i, d));
            }
        }
    }
    None
}

#[derive(Debug)]
enum SolutionState {
    Invalid,
    Solved,
    Unsolved,
}

fn solution_state(state: &State) -> SolutionState {
    let span = Digit::all()
        .map(|d| state.digit_mask(d))
        .fold(0, std::ops::BitOr::bitor);
    let unsolved_mask = state.unsolved_mask();
    if span != unsolved_mask {
        return SolutionState::Invalid;
    }
    if unsolved_mask == 0 {
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

    fn apply_naked_singles(&mut self, state: &mut State) -> bool {
        let mask = {
            let get = |x, y, z| {
                let x = state.digit_mask(Digit::new(x));
                let y = state.digit_mask(Digit::new(y));
                let z = state.digit_mask(Digit::new(z));
                ((x ^ y ^ z) & !(x & y & z), x | y | z)
            };
            let x = get(1, 2, 3);
            let y = get(4, 5, 6);
            let z = get(7, 8, 9);
            x.0 & !(y.1 | z.1) | y.0 & !(x.1 | z.1) | z.0 & !(x.1 | y.1)
        };
        for i in SquareIndex::all() {
            let mut count = 0;
            for d in Digit::all() {
                if state.is_candidate(i, d) {
                    count += 1;
                }
            }
            let a = count == 1;
            let b = mask >> i.get() & 1 != 0;
            debug_assert!(a == b);
        }
        if mask == 0 {
            return false;
        }
        for i in iter_mask_indices(mask) {
            for d in Digit::all() {
                if state.is_candidate(i, d) {
                    self.put(state, i, d);
                    break;
                }
            }
        }
        true
    }

    fn branch_impl(
        &mut self,
        state: &State,
        tuple: impl IntoIterator<Item = (SquareIndex, Digit)>,
    ) {
        for (i, d) in tuple.into_iter() {
            let mut clone = state.clone();
            self.put(&mut clone, i, d);
            self.solve_dfs(clone);
        }
    }

    fn branch(&mut self, state: &State) {
        let mut min = (10, Digit::new(1), 0);

        'outer: for unit_mask in unit_masks() {
            for d in Digit::all() {
                let mask = state.digit_mask(d) & unit_mask;
                if mask == 0 {
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

        self.branch_impl(&state, iter_mask_indices(min.2).map(move |i| (i, min.1)));
    }

    fn solve_dfs(&mut self, mut state: State) {
        loop {
            let ns_applied = self.apply_naked_singles(&mut state);
            let hs_applied = match find_hidden_single(&state) {
                None => false,
                Some((i, d)) => {
                    self.put(&mut state, i, d);
                    true
                }
            };
            if !ns_applied && !hs_applied {
                break;
            }
        }
        match solution_state(&state) {
            SolutionState::Invalid => return,
            SolutionState::Solved => return self.add_solved(),
            SolutionState::Unsolved => {}
        };
        self.branch(&state);
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
