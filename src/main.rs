extern crate failure;
extern crate structopt;
extern crate sudoku;

use failure::{Fallible, ResultExt};
use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::PathBuf,
};
use structopt::StructOpt;

#[derive(StructOpt)]
struct Solve {
    #[structopt(parse(from_os_str))]
    input_path: Option<PathBuf>,
    #[structopt(parse(from_os_str), short = "o", long = "output")]
    output_path: Option<PathBuf>,
}

fn solve(args: &Solve) -> Fallible<()> {
    let stdin = std::io::stdin();
    let input: Box<dyn BufRead> = match &args.input_path {
        Some(path) => Box::new(BufReader::new(
            File::open(path).context("Failed to open input file")?,
        )),
        None => Box::new(stdin.lock()),
    };
    let stdout = std::io::stdout();
    let mut output: Box<dyn Write> = match &args.output_path {
        Some(path) => Box::new(BufWriter::new(
            File::create(path).context("Failed to create output file")?,
        )),
        None => Box::new(stdout.lock()),
    };

    for line in input.lines() {
        let line = line.context("Failed to read input")?;
        let problem = sudoku::parse(line.trim())?;
        let solution = sudoku::solve(&problem);
        match solution {
            sudoku::Solution::No => writeln!(&mut output, "no solution"),
            sudoku::Solution::One(g) => writeln!(&mut output, "{}", g),
            sudoku::Solution::Multiple(Some(n)) => writeln!(&mut output, "{} solutions", n),
            sudoku::Solution::Multiple(None) => writeln!(&mut output, "multiple solutions"),
        }
        .context("Failed to write output")?;
    }

    Ok(())
}

#[derive(StructOpt)]
enum App {
    #[structopt(name = "solve")]
    Solve(Solve),
}

fn main() {
    let instant = std::time::Instant::now();
    match App::from_args() {
        App::Solve(args) => solve(&args),
    }
    .unwrap_or_else(|err| {
        for e in err.iter_chain() {
            eprintln!("{}", e);
        }
    });
    let duration = instant.elapsed();
    println!("{}.{:03}s", duration.as_secs(), duration.subsec_millis())
}
