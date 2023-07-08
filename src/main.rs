mod geo;
mod io;

use std::{collections::HashMap, f64::consts::PI, time::Duration};

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

struct ScoreDiffCalculator {
    attendees: Vec<Vec<(f64, i64)>>,
    blocked_count: Vec<Vec<i64>>,
    placements: Vec<Option<Point>>,
    score: i64,
}

impl ScoreDiffCalculator {
    fn new(problem: &io::Problem, placements: &[(u32, u32)]) -> ScoreDiffCalculator {
        let mut calculator = ScoreDiffCalculator {
            attendees: vec![vec![]; placements.len()],
            blocked_count: vec![vec![0; problem.attendees.len()]; placements.len()],
            placements: vec![None; placements.len()],
            score: 0,
        };
        for k in 0..placements.len() {
            calculator.place_musician(problem, k, placements[k]);
        }
        calculator
    }
    // O(|A| + |M| * (the size of range))
    fn remove_musician(&mut self, problem: &io::Problem, k: usize) -> i64 {
        assert!(!self.attendees[k].is_empty());
        assert!(self.placements[k].is_some());

        let mut score_diff = 0i64;

        for i in 0..problem.attendees.len() {
            if self.blocked_count[k][i] == 0 {
                score_diff -= self.attendees[k][i].1;
            }
        }

        let q = self.placements[k].unwrap();
        let q = geo::Point::new(q.0 as f64, q.1 as f64);
        self.placements[k] = None; // must be before self.placements iteration

        for j in 0..self.placements.len() {
            if let Some(p) = self.placements[j] {
                let p = geo::Point::new(p.0 as f64, p.1 as f64);
                let th = (q - p).arctan();
                let th_width = (BLOCK_RADIUS as f64 / (q - p).norm()).asin();
                if th - th_width <= -PI {
                    // (-inf, th + th_width)
                    let m = self.attendees[j].partition_point(|&x| x.0 < th + th_width);
                    for i in 0..m {
                        self.blocked_count[j][i] -= 1;
                        assert!(self.blocked_count[j][i] >= 0);
                        if self.blocked_count[j][i] == 0 {
                            score_diff += self.attendees[j][i].1;
                        }
                    }
                    // (th - th_width + 2PI, inf)
                    let m = self.attendees[j].partition_point(|&x| x.0 <= th - th_width + 2.0 * PI);
                    for i in m..self.attendees[j].len() {
                        self.blocked_count[j][i] -= 1;
                        assert!(self.blocked_count[j][i] >= 0);
                        if self.blocked_count[j][i] == 0 {
                            score_diff += self.attendees[j][i].1;
                        }
                    }
                } else if th + th_width > PI {
                    // (-inf, th + th_width - 2PI)
                    let m = self.attendees[j].partition_point(|&x| x.0 < th + th_width - 2.0 * PI);
                    for i in 0..m {
                        self.blocked_count[j][i] -= 1;
                        assert!(self.blocked_count[j][i] >= 0);
                        if self.blocked_count[j][i] == 0 {
                            score_diff += self.attendees[j][i].1;
                        }
                    }
                    // (th - th_width, inf)
                    let m = self.attendees[j].partition_point(|&x| x.0 <= th - th_width);
                    for i in m..self.attendees[j].len() {
                        self.blocked_count[j][i] -= 1;
                        assert!(self.blocked_count[j][i] >= 0);
                        if self.blocked_count[j][i] == 0 {
                            score_diff += self.attendees[j][i].1;
                        }
                    }
                } else {
                    // (th - th_width, th + th_width)
                    let l = self.attendees[j].partition_point(|&x| x.0 <= th - th_width);
                    let r = self.attendees[j].partition_point(|&x| x.0 < th + th_width);
                    for i in l..r {
                        self.blocked_count[j][i] -= 1;
                        assert!(self.blocked_count[j][i] >= 0);
                        if self.blocked_count[j][i] == 0 {
                            score_diff += self.attendees[j][i].1;
                        }
                    }
                }
            }
        }

        self.score += score_diff;
        self.attendees[k].clear();
        self.blocked_count[k].iter_mut().for_each(|x| *x = 0);

        score_diff
    }

    // O(|A| + |M| * (the size of range))
    fn place_musician(&mut self, problem: &io::Problem, k: usize, placement: Point) -> i64 {
        assert!(self.attendees[k].is_empty());
        assert!(self.blocked_count[k].iter().all(|&x| x == 0));
        assert!(self.placements[k].is_none());

        let p = geo::Point::new(placement.0 as f64, placement.1 as f64);
        let inst_type = problem.musicians[k];

        for i in 0..problem.attendees.len() {
            let a = &problem.attendees[i];
            let q = geo::Point::new(a.x as f64, a.y as f64);

            let th = (q - p).arctan();

            let taste_value = problem.attendees[i].tastes[inst_type as usize] as i64;
            let a = &problem.attendees[i];
            let d2 = dist_sq(placement, (a.x, a.y)) as i64;
            let add_score = (SCORE_FACTOR * taste_value + d2 - 1) / d2;
            self.attendees[k].push((th, add_score));
        }

        self.attendees[k].sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut score_diff = 0i64;

        for q in &self.placements {
            if let Some(q) = q {
                let q = geo::Point::new(q.0 as f64, q.1 as f64);
                let th = (q - p).arctan();
                let th_width = (BLOCK_RADIUS as f64 / (q - p).norm()).asin();
                if th - th_width <= -PI {
                    // (-inf, th + th_width)
                    let m = self.attendees[k].partition_point(|&x| x.0 < th + th_width);
                    for i in 0..m {
                        self.blocked_count[k][i] += 1;
                    }
                    // (th - th_width + 2PI, inf)
                    let m = self.attendees[k].partition_point(|&x| x.0 <= th - th_width + 2.0 * PI);
                    for i in m..self.attendees[k].len() {
                        self.blocked_count[k][i] += 1;
                    }
                } else if th + th_width > PI {
                    // (-inf, th + th_width - 2PI)
                    let m = self.attendees[k].partition_point(|&x| x.0 < th + th_width - 2.0 * PI);
                    for i in 0..m {
                        self.blocked_count[k][i] += 1;
                    }
                    // (th - th_width, inf)
                    let m = self.attendees[k].partition_point(|&x| x.0 <= th - th_width);
                    for i in m..self.attendees[k].len() {
                        self.blocked_count[k][i] += 1;
                    }
                } else {
                    // (th - th_width, th + th_width)
                    let l = self.attendees[k].partition_point(|&x| x.0 <= th - th_width);
                    let r = self.attendees[k].partition_point(|&x| x.0 < th + th_width);
                    for i in l..r {
                        self.blocked_count[k][i] += 1;
                    }
                }
            }
        }
        for i in 0..problem.attendees.len() {
            if self.blocked_count[k][i] == 0 {
                score_diff += self.attendees[k][i].1;
            }
        }

        for j in 0..self.placements.len() {
            if let Some(q) = self.placements[j] {
                let q = geo::Point::new(q.0 as f64, q.1 as f64);
                let th = (p - q).arctan(); // reverse
                let th_width = (BLOCK_RADIUS as f64 / (p - q).norm()).asin();
                if th - th_width <= -PI {
                    // (-inf, th + th_width)
                    let m = self.attendees[j].partition_point(|&x| x.0 < th + th_width);
                    for i in 0..m {
                        if self.blocked_count[j][i] == 0 {
                            score_diff -= self.attendees[j][i].1;
                        }
                        self.blocked_count[j][i] += 1;
                    }
                    // (th - th_width + 2PI, inf)
                    let m = self.attendees[j].partition_point(|&x| x.0 <= th - th_width + 2.0 * PI);
                    for i in m..self.attendees[j].len() {
                        if self.blocked_count[j][i] == 0 {
                            score_diff -= self.attendees[j][i].1;
                        }
                        self.blocked_count[j][i] += 1;
                    }
                } else if th + th_width > PI {
                    // (-inf, th + th_width - 2PI)
                    let m = self.attendees[j].partition_point(|&x| x.0 < th + th_width - 2.0 * PI);
                    for i in 0..m {
                        if self.blocked_count[j][i] == 0 {
                            score_diff -= self.attendees[j][i].1;
                        }
                        self.blocked_count[j][i] += 1;
                    }
                    // (th - th_width, inf)
                    let m = self.attendees[j].partition_point(|&x| x.0 <= th - th_width);
                    for i in m..self.attendees[j].len() {
                        if self.blocked_count[j][i] == 0 {
                            score_diff -= self.attendees[j][i].1;
                        }
                        self.blocked_count[j][i] += 1;
                    }
                } else {
                    // (th - th_width, th + th_width)
                    let l = self.attendees[j].partition_point(|&x| x.0 <= th - th_width);
                    let r = self.attendees[j].partition_point(|&x| x.0 < th + th_width);
                    for i in l..r {
                        if self.blocked_count[j][i] == 0 {
                            score_diff -= self.attendees[j][i].1;
                        }
                        self.blocked_count[j][i] += 1;
                    }
                }
            }
        }

        self.placements[k] = Some(placement); // must be after self.placements iteration
        self.score += score_diff;

        score_diff
    }
}

struct ScoreInfo {
    score: i64,
    blocked_score: i64,

    block_count: u32,
    pass_count: u32,
}

fn compute_score_fast(problem: &io::Problem, placements: &[Point]) -> i64 {
    let mut events = Vec::new();
    let mut score = 0i64;

    for k in 0..problem.musicians.len() {
        events.clear();
        let inst_type = problem.musicians[k];

        let p = geo::Point::new(placements[k].0 as f64, placements[k].1 as f64);
        for i in 0..problem.attendees.len() {
            let a = &problem.attendees[i];
            let q = geo::Point::new(a.x as f64, a.y as f64);
            let th = (q - p).arctan();
            events.push((th, 0, i));
        }

        for j in 0..problem.musicians.len() {
            if j == k {
                continue;
            }
            let q = geo::Point::new(placements[j].0 as f64, placements[j].1 as f64);
            let th = (q - p).arctan();

            let th_width = (BLOCK_RADIUS as f64 / (q - p).norm()).asin();
            if th - th_width <= -PI {
                events.push((-1e9, 1, 0));
                events.push((th + th_width, -1, 0));
                events.push((2.0 * PI + (th - th_width), 1, 0));
                events.push((1e9, -1, 0));
            } else if th + th_width > PI {
                events.push((th - th_width, 1, 0));
                events.push((1e9, -1, 0));
                events.push((-1e9, 1, 0));
                events.push((th + th_width - 2.0 * PI, -1, 0));
            } else {
                events.push((th - th_width, 1, 0));
                events.push((th + th_width, -1, 0));
            }
        }

        let mut counter = 0i64;

        events.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap()
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        });
        for &(_, t, i) in &events {
            if t == 0 {
                assert!(counter >= 0);
                if counter == 0 {
                    let taste_value = problem.attendees[i].tastes[inst_type as usize] as i64;
                    let a = &problem.attendees[i];
                    let d2 = dist_sq(placements[k], (a.x, a.y)) as i64;
                    let add_score = (SCORE_FACTOR * taste_value + d2 - 1) / d2;
                    score += add_score;
                }
            } else {
                counter += t;
            }
        }
    }

    score
}

fn compute_score(problem: &io::Problem, placements: &[Point]) -> ScoreInfo {
    let mut score = 0;

    let mut block_count = 0;
    let mut pass_count = 0;
    let mut blocked_score = 0;

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
                blocked_score += add_score;
                continue;
            }
            pass_count += 1;

            score += add_score;
        }
    }
    ScoreInfo {
        score,
        blocked_score,
        block_count,
        pass_count,
    }
}

fn generate_random_placement(problem: &io::Problem) -> Vec<Point> {
    let x_list = (problem.stage_left() + PLACEMENT_RADIUS
        ..=problem.stage_right() - PLACEMENT_RADIUS)
        .collect::<Vec<_>>();

    let y_list = (problem.stage_bottom() + PLACEMENT_RADIUS
        ..=problem.stage_top() - PLACEMENT_RADIUS)
        .collect::<Vec<_>>();
    generate_random_placement_from_list(problem, &x_list, &y_list)
}

fn generate_random_placement_from_list(
    problem: &io::Problem,
    x_list: &[u32],
    y_list: &[u32],
) -> Vec<Point> {
    assert!(
        problem.stage_width >= PLACEMENT_RADIUS * 2 && problem.stage_height >= PLACEMENT_RADIUS * 2
    );

    let mut rng = rand::thread_rng();

    let mut placements = Vec::new();

    for i in 0..problem.musicians.len() {
        let mut retry = 0;
        loop {
            let x = *x_list.choose(&mut rng).unwrap();
            let y = *y_list.choose(&mut rng).unwrap();

            assert!(x <= problem.room_width);
            assert!(y <= problem.room_height);
            assert!(problem.within_stage(x, y, PLACEMENT_RADIUS));

            if placements
                .iter()
                .any(|&p| within_or_equal(p, (x, y), PLACEMENT_RADIUS))
            {
                retry += 1;
                assert!(retry <= 1000);
                continue;
            }

            placements.push((x, y));
            break;
        }
    }

    placements
}

fn sample_by_unit(min_x: u32, max_x: u32, unit: u32) -> Vec<u32> {
    (min_x..=max_x).filter(|x| x % unit == 0).collect()
}

fn generate_without_block(
    problem: &io::Problem,
    duration: Duration,
    start_temp: f64,
    end_temp: f64,
) -> Vec<Point> {
    let unit = (1..=100)
        .into_iter()
        .filter(|u| {
            let cache_size = sample_by_unit(
                problem.stage_left() + PLACEMENT_RADIUS,
                problem.stage_right() - PLACEMENT_RADIUS,
                *u,
            )
            .len() as u64
                * sample_by_unit(
                    problem.stage_bottom() + PLACEMENT_RADIUS,
                    problem.stage_top() - PLACEMENT_RADIUS,
                    *u,
                )
                .len() as u64
                * (problem.max_inst() + 1) as u64;
            let combinations = cache_size * problem.attendees.len() as u64;
            cache_size <= 1e8 as u64 && combinations <= 1e10 as u64
        })
        .next()
        .unwrap();

    let x_list = sample_by_unit(
        problem.stage_left() + PLACEMENT_RADIUS,
        problem.stage_right() - PLACEMENT_RADIUS,
        unit,
    );
    let y_list = sample_by_unit(
        problem.stage_bottom() + PLACEMENT_RADIUS,
        problem.stage_top() - PLACEMENT_RADIUS,
        unit,
    );

    let mut score_cache = HashMap::new();
    for &y in &y_list {
        for &x in &x_list {
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

    let mut placements = generate_random_placement(problem);
    for k in 0..placements.len() {
        let inst_type = problem.musicians[k];
        let mut score = 0;
        for a in &problem.attendees {
            let taste_value = a.tastes[inst_type as usize] as i64;
            let d2 = dist_sq(placements[k], (a.x, a.y)) as i64;
            let add_score = (SCORE_FACTOR * taste_value + d2 - 1) / d2;
            score += add_score;
        }
        score_cache.insert((placements[k], inst_type), score);
    }

    let mut rng = rand::thread_rng();
    let start = std::time::Instant::now();
    let mut iterations = 0u64;
    loop {
        let time = start.elapsed().as_secs_f64() / duration.as_secs_f64();
        if time >= 1.0 {
            break;
        }

        let temp = start_temp + (end_temp - start_temp) * time;

        let k = rng.gen_range(0..problem.musicians.len());
        let inst_type = problem.musicians[k];

        let nx = *x_list.choose(&mut rng).unwrap();
        let ny = *y_list.choose(&mut rng).unwrap();

        if (0..problem.musicians.len())
            .any(|j| j != k && within_or_equal((nx, ny), placements[j], PLACEMENT_RADIUS))
        {
            continue;
        }

        let mut score_diff = 0i64;
        score_diff -= score_cache[&(placements[k], inst_type)];
        score_diff += score_cache[&((nx, ny), inst_type)];

        let prob = (score_diff as f64 / temp).exp().clamp(0.0, 1.0);

        if rng.gen_bool(prob) {
            // eprintln!(
            //     "{}s: {} (prob={}, temp={}, time={})",
            //     start.elapsed().as_secs(),
            //     score_diff,
            //     prob,
            //     temp,
            //     time
            // );
            placements[k] = (nx, ny);
        }
        iterations += 1;
    }

    // dbg!(iterations);

    placements
}

fn annealing(
    problem: &io::Problem,
    init: &[Point],
    duration: Duration,
    start_temp: f64,
    end_temp: f64,
) -> (Vec<Point>, u64) {
    let mut rng = rand::thread_rng();

    let mut placements = init.to_vec();
    let mut score = compute_score_fast(problem, &placements);
    let mut score_calculator = ScoreDiffCalculator::new(problem, &placements);

    let start = std::time::Instant::now();
    let mut iterations = 0u64;
    loop {
        let time = start.elapsed().as_secs_f64() / duration.as_secs_f64();
        if time >= 1.0 {
            break;
        }

        let temp = start_temp + (end_temp - start_temp) * time;

        let update_type = rng.gen_range(0..3);
        let mut undo: Box<dyn FnMut(&mut Vec<(u32, u32)>, &mut ScoreDiffCalculator)> =
            if update_type == 0 {
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
                score_calculator.remove_musician(problem, k);
                score_calculator.place_musician(problem, k, placements[k]);

                Box::new(move |p, c| {
                    p[k] = current;
                    c.remove_musician(problem, k);
                    c.place_musician(problem, k, p[k]);
                })
            } else if update_type == 1 {
                let k = rng.gen_range(0..problem.musicians.len());

                let current = placements[k];

                let x = rng.gen_range(
                    current.0.saturating_sub(PLACEMENT_RADIUS)..=current.0 + PLACEMENT_RADIUS,
                );
                let y = rng.gen_range(
                    current.1.saturating_sub(PLACEMENT_RADIUS)..=current.1 + PLACEMENT_RADIUS,
                );

                if !problem.within_stage(x, y, PLACEMENT_RADIUS) {
                    continue;
                }

                if (0..problem.musicians.len())
                    .any(|j| j != k && within_or_equal((x, y), placements[j], PLACEMENT_RADIUS))
                {
                    continue;
                }

                placements[k] = (x, y);
                score_calculator.remove_musician(problem, k);
                score_calculator.place_musician(problem, k, placements[k]);

                Box::new(move |p, c| {
                    p[k] = current;
                    c.remove_musician(problem, k);
                    c.place_musician(problem, k, p[k]);
                })
            } else {
                let (i, j) = {
                    let v = (0..problem.musicians.len()).choose_multiple(&mut rng, 2);
                    (v[0], v[1])
                };

                if problem.musicians[i] == problem.musicians[j] {
                    continue;
                }

                placements.swap(i, j);
                score_calculator.remove_musician(problem, i);
                score_calculator.remove_musician(problem, j);
                score_calculator.place_musician(problem, i, placements[i]);
                score_calculator.place_musician(problem, j, placements[j]);

                Box::new(move |p, c| {
                    p.swap(i, j);
                    c.remove_musician(problem, i);
                    c.remove_musician(problem, j);
                    c.place_musician(problem, i, p[i]);
                    c.place_musician(problem, j, p[j]);
                })
            };

        let new_score = score_calculator.score;

        let score_diff = new_score - score;
        let prob = (score_diff as f64 / temp).exp().clamp(0.0, 1.0);

        if rng.gen_bool(prob) {
            // eprintln!(
            //     "{}s: {} -> {} ({})",
            //     start.elapsed().as_secs(),
            //     score,
            //     new_score,
            //     new_score - score
            // );
            score = new_score;
        } else {
            undo(&mut placements, &mut score_calculator);
        }

        iterations += 1;
    }

    // dbg!(score_calculator.score);
    // dbg!(compute_score_fast(problem, &placements));
    // dbg!(compute_score(problem, &placements).score);

    (placements, iterations)
}

fn main() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(22)
        .build_global()
        .unwrap();

    let token = std::env::var("ICFPC_TOKEN").expect("ICFPC_TOKEN not set");

    let user_board = io::get_userboard(&token).unwrap();

    let mut best_ids: Vec<u32> = (1..=MAX_PROBLEM_ID).collect();
    best_ids.sort_by_key(|id| std::cmp::Reverse(user_board[*id as usize - 1].unwrap_or(0)));

    let results: Vec<_> = (1..=MAX_PROBLEM_ID)
        .into_par_iter()
        .filter(|&id| {
            // if best_ids.iter().take(10).find(|&x| x == &id).is_none() {
            //     // return false;
            // }
            // if !vec![8, 2, 1, 12].contains(&id) {
            //     return false;
            // }
            true
        })
        .map(|id| {
            let problem = io::read_problem(id);
            let m = problem.musicians.len();
            let a = problem.attendees.len();

            let placements = generate_without_block(&problem, Duration::from_secs(60), 1e5, 1e0);

            let score = compute_score(&problem, &placements);

            let score_ratio = score.score as f64 / (score.blocked_score + score.score) as f64;

            let (placements_anneal, anneal_iter) = annealing(
                &problem,
                &placements,
                Duration::from_secs(60 * 30),
                1e5,
                1e0,
            );
            let score_anneal = compute_score(&problem, &placements_anneal);

            eprintln!(
                "id: {:>2} block: {:>9} pass: {:>9} score: {:>14} score_without_block: {:>14} score_ratio: {:>3.0}% anneal_score: {:>14} anneal_score_ratio: {:>3.0}% anneal_iter: {:>10}",
                id,
                score.block_count.separate_with_commas(),
                score.pass_count.separate_with_commas(),
                score.score.separate_with_commas(),
                (score.blocked_score + score.score).separate_with_commas(),
                score_ratio * 100.0,
                score_anneal.score.separate_with_commas(),
                score_anneal.score as f64 / score.score as f64 * 100.0,
                anneal_iter.separate_with_commas(),
            );

            let max_score = user_board[id as usize - 1].unwrap_or(0);

            if score.score as f64 > max_score as f64 * 1.05 && score.score > score_anneal.score {
                eprintln!("Submitting problem {} with score {} ({}% increase)", id, score.score.separate_with_commas(), if max_score > 0 { (score.score as f64 / max_score as f64 * 100.0 - 100.0).round() } else { 1e9 });
                io::submit_placements(
                    &token,
                    problem.id,
                    placements
                        .iter()
                        .cloned()
                        .map(|(x, y)| (x as f64, y as f64))
                        .collect::<Vec<_>>()
                        .as_slice(),
                )
                .unwrap();
            }

            if score_anneal.score as f64 > max_score as f64 * 1.05 && score.score < score_anneal.score {
                eprintln!("Submitting problem {} with score {} ({}% increase) (anneal)", id, score_anneal.score.separate_with_commas(), if max_score > 0 { (score_anneal.score as f64 / max_score as f64 * 100.0 - 100.0).round() } else { 1e9 });
                io::submit_placements(
                    &token,
                    problem.id,
                   placements_anneal
                        .iter()
                        .cloned()
                        .map(|(x, y)| (x as f64, y as f64))
                        .collect::<Vec<_>>()
                        .as_slice(),
                )
                .unwrap();
            }

            (problem, placements, score, score_anneal)
        })
        .collect();

    let total_score = results
        .iter()
        .map(|(_, _, score, _)| score.score.max(0))
        .sum::<i64>();
    let score_without_block = results
        .iter()
        .map(|(_, _, score, _)| (score.blocked_score + score.score).max(score.score).max(0))
        .sum::<i64>();
    let total_anneal_score = results
        .iter()
        .map(|(_, _, _, score_anneal)| score_anneal.score.max(0))
        .sum::<i64>();

    dbg!(total_score.separate_with_commas());
    dbg!(score_without_block.separate_with_commas());
    dbg!(total_anneal_score.separate_with_commas());

    // eprintln!(
    //     "Start optimizing {} from score {}",
    //     id,
    //     compute_score_debug(&problem, &placements, true)
    // );

    // let placements = annealing(
    //     &problem,
    //     &placements,
    //     Duration::from_secs(50 * 60),
    //     1e6,
    //     1e0,
    // );
    // let score = compute_score_debug(&problem, &placements, true);
    // eprintln!("Id: {} Score: {}", id, score);

    // if score > 0 {
    //     eprintln!("Submitting problem {} with score {}", id, score);
    //     io::submit_placements(
    //         &token,
    //         problem.id,
    //         placements
    //             .into_iter()
    //             .map(|(x, y)| (x as f64, y as f64))
    //             .collect::<Vec<_>>()
    //             .as_slice(),
    //     )
    //     .unwrap();
    // }
}
