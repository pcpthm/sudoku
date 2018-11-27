extern crate criterion;
extern crate sudoku;

use criterion::{criterion_group, criterion_main, Criterion};
use std::{
    fs::File,
    io::{BufRead, BufReader},
};

fn bench(c: &mut Criterion) {
    let f = BufReader::new(File::open("sample25.txt").unwrap());
    for line in f.lines().take(5) {
        let line = line.unwrap();
        let problem = sudoku::parse(line.trim()).unwrap();
        c.bench_function(line.trim(), move |b| b.iter(|| sudoku::solve(&problem)));
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
