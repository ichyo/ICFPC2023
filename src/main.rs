mod io;

use io::{read_problem, Problem};
use rand::prelude::*;

type Point = (u32, u32);

fn within_or_equal(p: Point, q: Point, d: u32) -> bool {
    dist_sq(p, q) <= (d as u64).pow(2)
}

fn dist_sq(p: Point, q: Point) -> u64 {
    ((p.0 as i64 - q.0 as i64).pow(2) + (p.1 as i64 - q.1 as i64).pow(2)) as u64
}

const MUSICIAN_RADIUS: u32 = 10;

const SCORE_FACTOR: i64 = 1_000_000;

fn compute_score(problem: &Problem, placements: &[Point]) -> i64 {
    let mut score = 0;

    for attendee in &problem.attendees {
        let mut best = 0;
        let mut best_score = 0;

        for (i, &placement) in placements.iter().enumerate() {
            let dist_sq = dist_sq((attendee.x, attendee.y), placement);
            let score = attendee.tastes[i] as i64 * SCORE_FACTOR - dist_sq as i64;

            if score > best_score {
                best = i;
                best_score = score;
            }
        }

        score += best_score;
    }

    score
}

fn main() {
    for i in 1..=45 {
        eprintln!("reading {}", i);
        let problem = read_problem(i);
        eprintln!("solving {}", i);

        if problem.stage_width < MUSICIAN_RADIUS * 2 || problem.stage_height < MUSICIAN_RADIUS * 2 {
            eprintln!("skipping {}", i);
            continue;
        }

        eprintln!("# musicians: {}", problem.musicians.len());
        eprintln!("# attendees: {}", problem.attendees.len());
        eprintln!("M * A: {}", problem.musicians.len() * problem.attendees.len());
        eprintln!("M^2 * A: {}", problem.musicians.len().pow(2) * problem.attendees.len());

        eprintln!("room: {}x{}", problem.room_width, problem.room_height);
        eprintln!("stage: {}x{}", problem.stage_width, problem.stage_height);
        eprintln!(
            "stage/room: {:.2}%",
            100.0 * (problem.stage_width as f64 * problem.stage_height as f64)
                / (problem.room_width as f64 * problem.room_height as f64)
        );

        let mut rng = rand::thread_rng();

        let mut placements = Vec::new();

        for i in 0..problem.musicians.len() {
            loop {
                let x = problem.stage_bottom_left.0
                    + rng.gen_range(MUSICIAN_RADIUS..=problem.stage_width - MUSICIAN_RADIUS);
                let y = problem.stage_bottom_left.1
                    + rng.gen_range(MUSICIAN_RADIUS..=problem.stage_height - MUSICIAN_RADIUS);

                assert!(x <= problem.room_width);
                assert!(y <= problem.room_height);

                if placements
                    .iter()
                    .any(|&p| within_or_equal(p, (x, y), MUSICIAN_RADIUS))
                {
                    continue;
                }

                placements.push((x, y));
                break;
            }
        }

        eprintln!("done {}", i);
    }
}
