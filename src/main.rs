mod geo;
mod io;

use std::{collections::HashMap, fmt::Debug, time::Duration};

use rand::prelude::*;
use rayon::prelude::*;
use thousands::Separable;

type Point = (u32, u32);

fn within_or_equal(p: Point, q: Point, d: u32) -> bool {
    dist_sq(p, q) <= (d as u64).pow(2)
}

fn dist_sq(p: Point, q: Point) -> u64 {
    ((p.0 as i64 - q.0 as i64).pow(2) + (p.1 as i64 - q.1 as i64).pow(2)) as u64
}

const MAX_PROBLEM_ID: u32 = 55;

const BLOCK_RADIUS: u32 = 5;
const PLACEMENT_RADIUS: u32 = 10;

const SCORE_FACTOR: i64 = 1_000_000;

fn compute_score(problem: &io::Problem, placements: &[Point]) -> i64 {
    compute_score_debug(problem, placements, false)
}

fn compute_score_debug(problem: &io::Problem, placements: &[Point], debug: bool) -> i64 {
    let mut score = 0;

    let mut block_count = 0;
    let mut pass_count = 0;
    let mut block_score = 0;

    for a in &problem.attendees {
        for k in 0..problem.musicians.len() {
            let seg = (
                geo::Point::new(placements[k].0 as f64, placements[k].1 as f64),
                geo::Point::new(a.x as f64, a.y as f64),
            );

            let inst_type = problem.musicians[k];
            let taste_value = a.tastes[inst_type as usize] as i64;
            let d2 = dist_sq(placements[k], (a.x, a.y)) as i64;
            let add_score = (SCORE_FACTOR * taste_value + d2 - 1) / d2;

            if (0..problem.musicians.len()).any(|j| {
                j != k
                    && geo::dist_sq_segment_point(
                        seg,
                        geo::Point::new(placements[j].0 as f64, placements[j].1 as f64),
                    ) <= (BLOCK_RADIUS as f64).powi(2) + 2e-4
            }) {
                block_count += 1;
                block_score += add_score;
                continue;
            }
            pass_count += 1;

            score += add_score;
        }
    }
    if debug {
        eprintln!(
            "block: {} pass: {} block_score: {} score: {}",
            block_count.separate_with_commas(),
            pass_count.separate_with_commas(),
            block_score.separate_with_commas(),
            score.separate_with_commas()
        );
    }

    score
}

fn generate_random_placement(problem: &io::Problem) -> Vec<Point> {
    assert!(
        problem.stage_width >= PLACEMENT_RADIUS * 2 && problem.stage_height >= PLACEMENT_RADIUS * 2
    );

    let mut rng = rand::thread_rng();

    let mut placements = Vec::new();

    for i in 0..problem.musicians.len() {
        loop {
            let x = problem.stage_bottom_left.0
                + rng.gen_range(PLACEMENT_RADIUS..=problem.stage_width - PLACEMENT_RADIUS);
            let y = problem.stage_bottom_left.1
                + rng.gen_range(PLACEMENT_RADIUS..=problem.stage_height - PLACEMENT_RADIUS);

            assert!(x <= problem.room_width);
            assert!(y <= problem.room_height);
            assert!(problem.within_stage(x, y, PLACEMENT_RADIUS));

            if placements
                .iter()
                .any(|&p| within_or_equal(p, (x, y), PLACEMENT_RADIUS))
            {
                continue;
            }

            placements.push((x, y));
            break;
        }
    }

    placements
}

fn generate_without_block(
    problem: &io::Problem,
    duration: Duration,
    early_stop: Duration,
) -> Vec<Point> {
    let mut score_cache = HashMap::new();
    for sy in PLACEMENT_RADIUS..=problem.stage_height - PLACEMENT_RADIUS {
        let y = sy + problem.stage_bottom();
        for sx in PLACEMENT_RADIUS..=problem.stage_width - PLACEMENT_RADIUS {
            let x = sx + problem.stage_left();
            for inst_type in 0..=problem.max_inst() {
                let mut score = 0;
                for a in &problem.attendees {
                    let taste_value = a.tastes[inst_type as usize] as i64;
                    let d2 = dist_sq((x, y), (a.x, a.y)) as i64;
                    let add_score = (SCORE_FACTOR * taste_value + d2 - 1) / d2;
                    score += add_score;
                }
                score_cache.insert(((x, y), inst_type), score);
            }
        }
    }

    let mut rng = rand::thread_rng();
    let mut placements = generate_random_placement(problem);
    let start = std::time::Instant::now();
    let mut last_update = start;
    while start.elapsed() < duration && last_update.elapsed() < early_stop {
        let k = rng.gen_range(0..problem.musicians.len());
        let nx = problem.stage_bottom_left.0
            + rng.gen_range(PLACEMENT_RADIUS..=problem.stage_width - PLACEMENT_RADIUS);
        let ny = problem.stage_bottom_left.1
            + rng.gen_range(PLACEMENT_RADIUS..=problem.stage_height - PLACEMENT_RADIUS);
        let inst_type = problem.musicians[k];

        if (0..problem.musicians.len())
            .any(|j| j != k && within_or_equal((nx, ny), placements[j], PLACEMENT_RADIUS))
        {
            continue;
        }

        let mut score_diff = 0i64;
        score_diff -= score_cache[&(placements[k], inst_type)];
        score_diff += score_cache[&((nx, ny), inst_type)];

        if score_diff > 0 {
            placements[k] = (nx, ny);
            last_update = std::time::Instant::now();
            // eprintln!("{}s: {}", start.elapsed().as_secs(), score_diff);
        }
    }
    placements
}

fn annealing(
    problem: &io::Problem,
    init: &[Point],
    duration: Duration,
    start_temp: f64,
    end_temp: f64,
) -> Vec<Point> {
    let mut rng = rand::thread_rng();

    let mut placements = init.to_vec();
    let mut score = compute_score(problem, &placements);

    let start = std::time::Instant::now();
    let mut iterations = 0;
    loop {
        let time = start.elapsed().as_secs_f64() / duration.as_secs_f64();
        if time >= 1.0 {
            break;
        }

        let temp = start_temp + (end_temp - start_temp) * time;

        let k = rng.gen_range(0..problem.musicians.len());

        let current = placements[k];

        let x = problem.stage_bottom_left.0
            + rng.gen_range(PLACEMENT_RADIUS..=problem.stage_width - PLACEMENT_RADIUS);
        let y = problem.stage_bottom_left.1
            + rng.gen_range(PLACEMENT_RADIUS..=problem.stage_height - PLACEMENT_RADIUS);

        // if !problem.within_stage(x, y, PLACEMENT_RADIUS) {
        //     continue;
        // }

        if (0..problem.musicians.len())
            .any(|j| j != k && within_or_equal((x, y), placements[j], PLACEMENT_RADIUS))
        {
            continue;
        }

        placements[k] = (x, y);

        let new_score = compute_score(problem, &placements);
        let score_diff = new_score - score;
        let prob = (score_diff as f64 / temp).exp().clamp(0.0, 1.0);

        if rng.gen_bool(prob) {
            eprintln!(
                "{}s: {} -> {} ({})",
                start.elapsed().as_secs(),
                score,
                new_score,
                new_score - score
            );
            score = new_score;
        } else {
            placements[k] = current;
        }

        iterations += 1;
    }
    eprintln!("iterations: {}", iterations);

    placements
}

fn climbing(
    problem: &io::Problem,
    init: &[Point],
    timeout: Duration,
    early_stop: Duration,
) -> Vec<Point> {
    let mut rng = rand::thread_rng();

    let mut placements = init.to_vec();
    let mut score = compute_score(problem, &placements);

    let start = std::time::Instant::now();
    let mut last_update = start;
    while start.elapsed() < timeout && last_update.elapsed() < early_stop {
        let k = rng.gen_range(0..problem.musicians.len());

        let current = placements[k];

        let x = problem.stage_bottom_left.0
            + rng.gen_range(PLACEMENT_RADIUS..=problem.stage_width - PLACEMENT_RADIUS);
        let y = problem.stage_bottom_left.1
            + rng.gen_range(PLACEMENT_RADIUS..=problem.stage_height - PLACEMENT_RADIUS);

        if !problem.within_stage(x, y, PLACEMENT_RADIUS) {
            continue;
        }

        if (0..problem.musicians.len())
            .any(|j| j != k && within_or_equal((x, y), placements[j], PLACEMENT_RADIUS))
        {
            continue;
        }

        placements[k] = (x, y);

        let new_score = compute_score(problem, &placements);
        if new_score > score {
            //eprintln!("{}s: {} -> {}", start.elapsed().as_secs(), score, new_score);
            score = new_score;
            last_update = std::time::Instant::now();
        } else {
            placements[k] = current;
        }
    }

    placements
}

fn main() {
    let token = std::env::var("ICFPC_TOKEN").expect("ICFPC_TOKEN not set");

    (1..=MAX_PROBLEM_ID).into_par_iter().for_each(|id| {
        let problem = io::read_problem(id);
        let m = problem.musicians.len();
        let a = problem.attendees.len();

        // if !vec![8].into_iter().find(|&x| x == id).is_some() {
        //     return;
        // }

        /*
        if m * m * a > 10_000_000_000 {
            eprintln!(
                "Skipping problem {}: too many combinations ({})",
                id,
                (m * m * a).separate_with_commas()
            );
            continue;
        }
        */

        /*
        eprintln!("# musicians: {}", problem.musicians.len());
        eprintln!("# attendees: {}", problem.attendees.len());
        eprintln!(
            "M * A: {}",
            problem.musicians.len() * problem.attendees.len()
        );
        eprintln!(
            "M^2 * A: {}",
            problem.musicians.len().pow(2) * problem.attendees.len()
        );

        eprintln!("room: {}x{}", problem.room_width, problem.room_height);
        eprintln!("stage: {}x{}", problem.stage_width, problem.stage_height);
        eprintln!(
            "stage/room: {:.2}%",
            100.0 * (problem.stage_width as f64 * problem.stage_height as f64)
                / (problem.room_width as f64 * problem.room_height as f64)
        );
        */

        let placements =
            generate_without_block(&problem, Duration::from_secs(60), Duration::from_secs(10));

        eprintln!(
            "Start optimizing {} from score {}",
            id,
            compute_score(&problem, &placements)
        );

        let placements = climbing(
            &problem,
            &placements,
            Duration::from_secs(60),
            Duration::from_secs(10),
        );
        // let placements = annealing(&problem, &placements, Duration::from_secs(60), 1e6, 1e2);
        let score = compute_score_debug(&problem, &placements, true);
        eprintln!("Id: {} Score: {}", id, score);

        if score > 0 {
            eprintln!("Submitting problem {} with score {}", id, score);
            io::submit_placements(
                &token,
                problem.id,
                placements
                    .into_iter()
                    .map(|(x, y)| (x as f64, y as f64))
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
            .unwrap();
        }
    })
}
