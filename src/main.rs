mod geo;
mod io;

use std::{
    collections::HashMap,
    f64::{consts::PI, INFINITY},
    time::Duration,
};

use bitvec::prelude::*;
use clap::Parser;
use log::{error, info};
use rand::prelude::*;
use rayon::prelude::*;
use thousands::Separable;

type Point = (u32, u32);

fn within(p: Point, q: Point, d: u32) -> bool {
    dist_sq(p, q) < (d as u64).pow(2)
}

fn dist_sq(p: Point, q: Point) -> u64 {
    ((p.0 as i64 - q.0 as i64).pow(2) + (p.1 as i64 - q.1 as i64).pow(2)) as u64
}

const MAX_PROBLEM_ID: u32 = 90;
const FIRST_PROBLEM_ID_PHASE_2: u32 = 56;

const BLOCK_RADIUS: u32 = 5;
const PLACEMENT_RADIUS: u32 = 10;

const SCORE_FACTOR: i64 = 1_000_000;

struct ScoreDiffCalculator {
    attendees: Vec<Vec<(f64, i64)>>,
    blocked_count: Vec<Vec<i64>>,
    placements: Vec<Option<Point>>,
    raw_score: Vec<i64>,
    bonus_factor: Vec<f64>,
    pillar_block_cache: HashMap<(u32, u32), BitVec>,
    volume: Vec<bool>,
}

impl ScoreDiffCalculator {
    fn new(
        problem: &io::Problem,
        placements: &[(u32, u32)],
        volume: &[bool],
    ) -> ScoreDiffCalculator {
        let mut calculator = ScoreDiffCalculator {
            attendees: vec![vec![]; placements.len()],
            blocked_count: vec![vec![0; problem.attendees.len()]; placements.len()],
            placements: vec![None; placements.len()],
            raw_score: vec![0; placements.len()],
            bonus_factor: vec![0.0; placements.len()],
            pillar_block_cache: HashMap::new(),
            volume: volume.to_vec(),
        };
        for k in 0..placements.len() {
            calculator.place_musician(problem, k, placements[k]);
        }
        calculator
    }

    fn compute_score(&self) -> i64 {
        10i64
            * self
                .raw_score
                .iter()
                .zip(self.bonus_factor.iter())
                .zip(self.volume.iter())
                .map(|((&raw_score, &bonus_factor), &volume)| {
                    if volume {
                        (raw_score as f64 * (1.0 + bonus_factor)).ceil() as i64
                    } else {
                        0
                    }
                })
                .sum::<i64>()
    }

    fn toggle_volume(&mut self, k: usize) -> i64 {
        let score = (self.raw_score[k] as f64 * (1.0 + self.bonus_factor[k])).ceil() as i64;
        self.volume[k] = !self.volume[k];
        if self.volume[k] {
            score
        } else {
            -score
        }
    }

    // O(|A| + |M| * (the size of range))
    fn remove_musician(&mut self, problem: &io::Problem, k: usize) {
        assert!(!self.attendees[k].is_empty());
        assert!(self.placements[k].is_some());

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
                            self.raw_score[j] += self.attendees[j][i].1;
                        }
                    }
                    // (th - th_width + 2PI, inf)
                    let m = self.attendees[j].partition_point(|&x| x.0 <= th - th_width + 2.0 * PI);
                    for i in m..self.attendees[j].len() {
                        self.blocked_count[j][i] -= 1;
                        assert!(self.blocked_count[j][i] >= 0);
                        if self.blocked_count[j][i] == 0 {
                            self.raw_score[j] += self.attendees[j][i].1;
                        }
                    }
                } else if th + th_width > PI {
                    // (-inf, th + th_width - 2PI)
                    let m = self.attendees[j].partition_point(|&x| x.0 < th + th_width - 2.0 * PI);
                    for i in 0..m {
                        self.blocked_count[j][i] -= 1;
                        assert!(self.blocked_count[j][i] >= 0);
                        if self.blocked_count[j][i] == 0 {
                            self.raw_score[j] += self.attendees[j][i].1;
                        }
                    }
                    // (th - th_width, inf)
                    let m = self.attendees[j].partition_point(|&x| x.0 <= th - th_width);
                    for i in m..self.attendees[j].len() {
                        self.blocked_count[j][i] -= 1;
                        assert!(self.blocked_count[j][i] >= 0);
                        if self.blocked_count[j][i] == 0 {
                            self.raw_score[j] += self.attendees[j][i].1;
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
                            self.raw_score[j] += self.attendees[j][i].1;
                        }
                    }
                }
            }
        }

        if problem.bonus_enabled() {
            for j in 0..self.placements.len() {
                if problem.musicians[j] != problem.musicians[k] {
                    continue;
                }
                if let Some(p) = self.placements[j] {
                    let p = geo::Point::new(p.0 as f64, p.1 as f64);
                    let d = (p - q).norm();
                    self.bonus_factor[j] -= 1.0 / d;
                    assert!(self.bonus_factor[j] >= 0.0);
                }
            }
        }

        self.attendees[k].clear();
        self.blocked_count[k].iter_mut().for_each(|x| *x = 0);
        self.raw_score[k] = 0;
        self.bonus_factor[k] = 0.0;
    }

    // O(|A| + |M| * (the size of range))
    fn place_musician(&mut self, problem: &io::Problem, k: usize, placement: Point) {
        assert!(self.attendees[k].is_empty());
        assert!(self.blocked_count[k].iter().all(|&x| x == 0));
        assert!(self.placements[k].is_none());
        assert!(self.raw_score[k] == 0);
        assert!(self.bonus_factor[k] == 0.0);

        let p = geo::Point::new(placement.0 as f64, placement.1 as f64);
        let inst_type = problem.musicians[k];

        if !self.pillar_block_cache.contains_key(&placement) && !problem.pillars.is_empty() {
            let mut blocked_vec = BitVec::with_capacity(problem.attendees.len());
            for a in &problem.attendees {
                let q = geo::Point::new(a.x as f64, a.y as f64);
                let seg = (p, q);
                let mut blocked = false;
                for pillar in &problem.pillars {
                    let center = geo::Point::new(pillar.center.0 as f64, pillar.center.1 as f64);
                    if geo::dist_sq_segment_point(seg, center) < (pillar.radius as f64).powi(2) {
                        blocked = true;
                        break;
                    }
                }
                blocked_vec.push(blocked);
            }
            self.pillar_block_cache.insert(placement, blocked_vec);
        }
        let blocked_vec = self.pillar_block_cache.get(&placement);

        for i in 0..problem.attendees.len() {
            if blocked_vec.map_or(false, |v| v[i]) {
                continue;
            }
            let a = &problem.attendees[i];
            let q = geo::Point::new(a.x as f64, a.y as f64);

            let th = (q - p).arctan();

            let taste_value = problem.attendees[i].tastes[inst_type as usize] as i64;
            let d2 = dist_sq(placement, (a.x, a.y)) as i64;
            let add_score = (SCORE_FACTOR * taste_value + d2 - 1) / d2;
            self.attendees[k].push((th, add_score));
        }

        self.attendees[k].sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

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
        for i in 0..self.attendees[k].len() {
            if self.blocked_count[k][i] == 0 {
                self.raw_score[k] += self.attendees[k][i].1;
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
                            self.raw_score[j] -= self.attendees[j][i].1;
                        }
                        self.blocked_count[j][i] += 1;
                    }
                    // (th - th_width + 2PI, inf)
                    let m = self.attendees[j].partition_point(|&x| x.0 <= th - th_width + 2.0 * PI);
                    for i in m..self.attendees[j].len() {
                        if self.blocked_count[j][i] == 0 {
                            self.raw_score[j] -= self.attendees[j][i].1;
                        }
                        self.blocked_count[j][i] += 1;
                    }
                } else if th + th_width > PI {
                    // (-inf, th + th_width - 2PI)
                    let m = self.attendees[j].partition_point(|&x| x.0 < th + th_width - 2.0 * PI);
                    for i in 0..m {
                        if self.blocked_count[j][i] == 0 {
                            self.raw_score[j] -= self.attendees[j][i].1;
                        }
                        self.blocked_count[j][i] += 1;
                    }
                    // (th - th_width, inf)
                    let m = self.attendees[j].partition_point(|&x| x.0 <= th - th_width);
                    for i in m..self.attendees[j].len() {
                        if self.blocked_count[j][i] == 0 {
                            self.raw_score[j] -= self.attendees[j][i].1;
                        }
                        self.blocked_count[j][i] += 1;
                    }
                } else {
                    // (th - th_width, th + th_width)
                    let l = self.attendees[j].partition_point(|&x| x.0 <= th - th_width);
                    let r = self.attendees[j].partition_point(|&x| x.0 < th + th_width);
                    for i in l..r {
                        if self.blocked_count[j][i] == 0 {
                            self.raw_score[j] -= self.attendees[j][i].1;
                        }
                        self.blocked_count[j][i] += 1;
                    }
                }
            }
        }

        if problem.bonus_enabled() {
            for j in 0..self.placements.len() {
                if problem.musicians[j] != problem.musicians[k] {
                    continue;
                }
                if let Some(q) = self.placements[j] {
                    let q = geo::Point::new(q.0 as f64, q.1 as f64);
                    let d = (p - q).norm();
                    self.bonus_factor[j] += 1.0 / d;
                    self.bonus_factor[k] += 1.0 / d;
                }
            }
        }

        self.placements[k] = Some(placement); // must be after self.placements iteration
    }
}

struct ScoreInfo {
    score: i64,
    blocked_score: i64,

    block_count: u32,
    pass_count: u32,
}

fn compute_score_fast(problem: &io::Problem, placements: &[Point], volume: &[bool]) -> i64 {
    ScoreDiffCalculator::new(problem, placements, volume).compute_score()
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
                .any(|&p| within(p, (x, y), PLACEMENT_RADIUS))
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
            let w = sample_by_unit(
                problem.stage_left() + PLACEMENT_RADIUS,
                problem.stage_right() - PLACEMENT_RADIUS,
                *u,
            )
            .len() as u64;
            let h = sample_by_unit(
                problem.stage_bottom() + PLACEMENT_RADIUS,
                problem.stage_top() - PLACEMENT_RADIUS,
                *u,
            )
            .len() as u64;
            let cache_size = h * w * (problem.max_inst() + 1) as u64;
            let combinations = cache_size * problem.attendees.len() as u64;
            let combinations2 =
                w * h * problem.attendees.len() as u64 * problem.pillars.len() as u64;
            cache_size <= 1e7 as u64 && combinations.max(combinations2) <= 1e9 as u64
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
            let p = geo::Point::new(x as f64, y as f64);
            for a in &problem.attendees {
                let q = geo::Point::new(a.x as f64, a.y as f64);
                let seg = (p, q);

                let blocked = problem.pillars.iter().any(|p| {
                    let center = geo::Point::new(p.center.0 as f64, p.center.1 as f64);
                    geo::dist_sq_segment_point(seg, center) < (p.radius as f64).powi(2)
                });

                if blocked {
                    continue;
                }

                let d2 = dist_sq((x, y), (a.x, a.y)) as i64;
                for inst_type in 0..=problem.max_inst() {
                    let taste_value = a.tastes[inst_type as usize] as i64;
                    let add_score = (SCORE_FACTOR * taste_value + d2 - 1) / d2;
                    *score_cache.entry(((x, y), inst_type)).or_default() += add_score;
                }
            }
        }
    }

    let mut placements = generate_random_placement(problem);
    for k in 0..placements.len() {
        for inst_type in 0..=problem.max_inst() {
            let mut score = 0;
            let p = geo::Point::new(placements[k].0 as f64, placements[k].1 as f64);
            for a in &problem.attendees {
                let q = geo::Point::new(a.x as f64, a.y as f64);
                let seg = (p, q);

                let blocked = problem.pillars.iter().any(|p| {
                    let center = geo::Point::new(p.center.0 as f64, p.center.1 as f64);
                    geo::dist_sq_segment_point(seg, center) < (p.radius as f64).powi(2)
                });

                if blocked {
                    continue;
                }

                let taste_value = a.tastes[inst_type as usize] as i64;
                let d2 = dist_sq(placements[k], (a.x, a.y)) as i64;
                let add_score = (SCORE_FACTOR * taste_value + d2 - 1) / d2;
                score += add_score;
            }
            score_cache.insert((placements[k], inst_type), score);
        }
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

        let update_type = rng.gen_range(0..2);
        if update_type == 0 {
            let k = rng.gen_range(0..problem.musicians.len());
            let inst_type = problem.musicians[k];

            let nx = *x_list.choose(&mut rng).unwrap();
            let ny = *y_list.choose(&mut rng).unwrap();

            if (0..problem.musicians.len())
                .any(|j| j != k && within((nx, ny), placements[j], PLACEMENT_RADIUS))
            {
                continue;
            }

            let mut score_diff = 0i64;
            score_diff -= score_cache[&(placements[k], inst_type)];
            score_diff += score_cache[&((nx, ny), inst_type)];

            let prob = (score_diff as f64 / temp).exp().clamp(0.0, 1.0);

            if rng.gen_bool(prob) {
                placements[k] = (nx, ny);
            }
        } else {
            let (i, j) = {
                let v = (0..problem.musicians.len()).choose_multiple(&mut rng, 2);
                (v[0], v[1])
            };
            if problem.musicians[i] == problem.musicians[j] {
                continue;
            }

            let mut score_diff = 0i64;
            score_diff -= score_cache[&(placements[i], problem.musicians[i])];
            score_diff -= score_cache[&(placements[j], problem.musicians[j])];
            score_diff += score_cache[&(placements[j], problem.musicians[i])];
            score_diff += score_cache[&(placements[i], problem.musicians[j])];

            let prob = (score_diff as f64 / temp).exp().clamp(0.0, 1.0);

            if rng.gen_bool(prob) {
                placements.swap(i, j);
            }
        }
        iterations += 1;
    }

    // dbg!(iterations);

    placements
}

fn annealing(
    problem: &io::Problem,
    init: &[Point],
    init_volumes: &[bool],
    duration: Duration,
    start_temp: f64,
    end_temp: f64,
) -> (Vec<Point>, Vec<bool>, u64) {
    let mut rng = rand::thread_rng();

    let mut placements = init.to_vec();
    let mut volumes = init_volumes.to_vec();
    let mut score_calculator = ScoreDiffCalculator::new(problem, &placements, &volumes);
    let mut score = score_calculator.compute_score();

    let start = std::time::Instant::now();
    let mut iterations = 0u64;

    let dx = [1, 0, -1, 0];
    let dy = [0, 1, 0, -1];

    //let mut update_counter: HashMap<i32, u64> = HashMap::new();
    loop {
        let time = start.elapsed().as_secs_f64() / duration.as_secs_f64();
        if time >= 1.0 {
            break;
        }

        let temp = start_temp + (end_temp - start_temp) * time;

        let update_type = rng.gen_range(0..4);

        if update_type == 3 {
            let k = rng.gen_range(0..problem.musicians.len());

            volumes[k] = !volumes[k];
            let score_diff = score_calculator.toggle_volume(k);

            let prob = (score_diff as f64 / temp).exp().clamp(0.0, 1.0);

            if rng.gen_bool(prob) {
                score += score_diff;
            } else {
                volumes[k] = !volumes[k];
                score_calculator.toggle_volume(k);
            }

            iterations += 1;
            continue;
        }

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
                    .any(|j| j != k && within((x, y), placements[j], PLACEMENT_RADIUS))
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

                let r = rng.gen_range(0..4);
                let len = rng.gen_range(1..=5);
                let x = current.0.saturating_add_signed(len * dx[r]);
                let y = current.1.saturating_add_signed(len * dy[r]);

                if (x, y) == current {
                    continue;
                }

                if !problem.within_stage(x, y, PLACEMENT_RADIUS) {
                    continue;
                }

                if (0..problem.musicians.len())
                    .any(|j| j != k && within((x, y), placements[j], PLACEMENT_RADIUS))
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
            } else if update_type == 2 {
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
            } else {
                unreachable!()
            };

        let new_score = score_calculator.compute_score();

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
            //*update_counter.entry(update_type).or_default() += 1;
        } else {
            undo(&mut placements, &mut score_calculator);
        }

        iterations += 1;
    }

    // dbg!(score_calculator.score);
    // dbg!(compute_score_fast(problem, &placements));
    // dbg!(compute_score(problem, &placements).score);
    // dbg!(update_counter);

    (placements, volumes, iterations)
}

struct ExecInfo {
    id: u32,
    score: i64,
    iterations: u64,
    best_score: i64,
}

#[derive(Parser, Debug)]
struct Args {
    #[clap(short, long)]
    threads: Option<usize>,
    #[clap(long)]
    top_n: Option<u32>,
    #[clap(long)]
    min_id: Option<u32>,
    #[clap(long)]
    max_id: Option<u32>,
    #[clap(short, long, required = true)]
    duration_mins: Vec<u64>,
    #[clap(long)]
    disable_init: bool,
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let token = std::env::var("ICFPC_TOKEN").expect("ICFPC_TOKEN not set");

    let args = Args::parse();

    info!("arguments: {:?}", args);

    let threads = match args.threads {
        Some(threads) => {
            info!("using {} threads", threads);
            threads
        }
        None => {
            let physical_cpus = num_cpus::get_physical();
            info!("-t not specified, using {} threads", physical_cpus);
            physical_cpus
        }
    };

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();

    let user_board = io::get_userboard(&token).unwrap();

    let mut best_ids: Vec<u32> = (1..=MAX_PROBLEM_ID).collect();
    best_ids.sort_by_key(|id| std::cmp::Reverse(user_board[*id as usize - 1].unwrap_or(0)));

    let run_ids = best_ids
        .into_iter()
        .filter(|&id| {
            if let Some(min_id) = args.min_id {
                if id < min_id {
                    return false;
                }
            }
            if let Some(max_id) = args.max_id {
                if id > max_id {
                    return false;
                }
            }
            true
        })
        .take(args.top_n.unwrap_or(u32::MAX) as usize)
        .collect::<Vec<_>>();

    info!("Running for {} ids: {:?}", run_ids.len(), run_ids);

    info!("Durations are {:?}", args.duration_mins);

    for &duration_min in &args.duration_mins {
        info!("Starting for duration {} min", duration_min);
        let results: Vec<_> = run_ids
            .clone()
            .into_par_iter()
            .map(|id| {
                let problem = io::read_problem(id);

                let duration = Duration::from_secs(duration_min * 60);
                let init_duration = (duration / 60).max(Duration::from_secs(60).min(duration));

                info!("Starting problem {} with duration {:?}", id, duration);
                let (placements, volumes, iterations) = annealing(
                    &problem,
                    &if !args.disable_init {
                        generate_without_block(&problem, init_duration, 1e5, 1e0)
                    } else {
                        generate_random_placement(&problem)
                    },
                    &vec![true; problem.musicians.len()],
                    duration,
                    1e5,
                    1e0,
                );

                let score = compute_score_fast(&problem, &placements, &volumes);
                let best_score = user_board[id as usize - 1].unwrap_or(0);
                info!(
                    "id: {:>2} score: {:>14} volume: {:3.0}% iter: {:>10} best: {:>14} ratio_to_best: {:4}%",
                    id,
                    score.separate_with_commas(),
                    volumes.iter().filter(|&&x| x).count() as f64 * 100.0 / volumes.len() as f64,
                    iterations.separate_with_commas(),
                    best_score.separate_with_commas(),
                    if best_score > 0 {
                        score as f64 * 100.0 / best_score as f64
                    } else {
                        INFINITY
                    }
                );
                if score as f64 > best_score as f64 + 1e6 {
                    info!(
                        "Submitting problem {} with score {} ({}% increase)",
                        id,
                        score.separate_with_commas(),
                        if best_score > 0 {
                            (score as f64 / best_score as f64 * 100.0 - 100.0).round()
                        } else {
                            INFINITY
                        }
                    );
                    if let Err(e) = io::submit_placements(
                        &token,
                        problem.id,
                        placements
                            .iter()
                            .cloned()
                            .map(|(x, y)| (x as f64, y as f64))
                            .collect::<Vec<_>>()
                            .as_slice(),
                        &volumes,
                    ) {
                        error!("Failed to submit: {}", e);
                    }
                }

                ExecInfo {
                    id,
                    score,
                    iterations,
                    best_score,
                }
            })
            .collect();

        info!("Showing all results");

        for result in &results {
            info!(
                "id: {:>2} score: {:>14} iter: {:>10} best: {:>14} ratio_to_best: {:4}%",
                result.id,
                result.score.separate_with_commas(),
                result.iterations.separate_with_commas(),
                result.best_score.separate_with_commas(),
                if result.best_score > 0 {
                    result.score as f64 * 100.0 / result.best_score as f64
                } else {
                    INFINITY
                }
            );
        }

        let total_score = results.iter().map(|x| x.score).sum::<i64>();

        info!("total_score: {}", total_score.separate_with_commas());
    }
}
