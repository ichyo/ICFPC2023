use std::fs::File;

use anyhow::Result;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
struct ProblemJson {
    room_width: f64,
    room_height: f64,
    stage_width: f64,
    stage_height: f64,
    stage_bottom_left: (f64, f64),
    musicians: Vec<f64>,
    attendees: Vec<AttendeeJson>,
}

#[derive(Deserialize, Debug)]
struct AttendeeJson {
    x: f64,
    y: f64,
    tastes: Vec<f64>,
}

#[derive(Debug)]
struct Problem {
    room_width: u32,
    room_height: u32,
    stage_width: u32,
    stage_height: u32,
    stage_bottom_left: (u32, u32),
    musicians: Vec<u32>,
    attendees: Vec<Attendee>,
}

#[derive(Debug)]
struct Attendee {
    x: u32,
    y: u32,
    tastes: Vec<i32>,
}

fn try_i32(x: f64) -> Result<i32> {
    if (x as i32) as f64 == x {
        Ok(x as i32)
    } else {
        Err(anyhow::anyhow!("Cannot convert {} to i32", x))
    }
}

fn try_u32(x: f64) -> Result<u32> {
    if (x as u32) as f64 == x {
        Ok(x as u32)
    } else {
        Err(anyhow::anyhow!("Cannot convert {} to u32", x))
    }
}

fn try_read_problem(id: u32) -> Result<Problem> {
    let path = format!("problems/{}.json", id);
    let file = File::open(path)?;
    let problem_json: ProblemJson = serde_json::from_reader(file)?;
    let problem = Problem {
        room_width: try_u32(problem_json.room_width)?,
        room_height: try_u32(problem_json.room_height)?,
        stage_width: try_u32(problem_json.stage_width)?,
        stage_height: try_u32(problem_json.stage_height)?,
        stage_bottom_left: (
            try_u32(problem_json.stage_bottom_left.0)?,
            try_u32(problem_json.stage_bottom_left.1)?,
        ),
        musicians: problem_json
            .musicians
            .iter()
            .map(|&x| try_u32(x))
            .collect::<Result<_>>()?,
        attendees: problem_json
            .attendees
            .iter()
            .map(|a| {
                Ok(Attendee {
                    x: try_u32(a.x)?,
                    y: try_u32(a.y)?,
                    tastes: a
                        .tastes
                        .iter()
                        .map(|&x| try_i32(x))
                        .collect::<Result<_>>()?,
                })
            })
            .collect::<Result<_>>()?,
    };
    Ok(problem)
}

fn read_problem(id: u32) -> Problem {
    try_read_problem(id).expect("Cannot read problem {}")
}

fn main() {
    for i in 1..=45 {
        let problem = read_problem(i);
        println!("{:?}", problem);
    }
}
