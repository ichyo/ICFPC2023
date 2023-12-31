use std::{fs::File, io::Write};

use anyhow::Result;
use log::info;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Deserialize, Debug)]
struct ProblemJson {
    room_width: f64,
    room_height: f64,
    stage_width: f64,
    stage_height: f64,
    stage_bottom_left: (f64, f64),
    musicians: Vec<f64>,
    attendees: Vec<AttendeeJson>,
    pillars: Vec<PillarJson>,
}

#[derive(Deserialize, Debug)]
struct AttendeeJson {
    x: f64,
    y: f64,
    tastes: Vec<f64>,
}

#[derive(Deserialize, Debug)]
struct PillarJson {
    center: (f64, f64),
    radius: f64,
}

#[derive(Debug)]
pub struct Problem {
    pub id: u32,
    pub room_width: u32,
    pub room_height: u32,
    pub stage_width: u32,
    pub stage_height: u32,
    pub stage_bottom_left: (u32, u32),
    pub musicians: Vec<u32>,
    pub attendees: Vec<Attendee>,
    pub pillars: Vec<Pillar>,
}

#[derive(Debug)]
pub struct Pillar {
    pub center: (u32, u32),
    pub radius: u32,
}

impl Problem {
    pub fn bonus_enabled(&self) -> bool {
        !self.pillars.is_empty()
    }

    pub fn max_inst(&self) -> u32 {
        self.musicians.iter().max().unwrap().clone()
    }

    pub fn within_stage(&self, x: u32, y: u32, r: u32) -> bool {
        x >= self.stage_left() + r
            && x + r <= self.stage_right()
            && y >= self.stage_bottom() + r
            && y + r <= self.stage_top()
    }

    pub fn stage_left(&self) -> u32 {
        self.stage_bottom_left.0
    }
    pub fn stage_right(&self) -> u32 {
        self.stage_bottom_left.0 + self.stage_width
    }
    pub fn stage_bottom(&self) -> u32 {
        self.stage_bottom_left.1
    }
    pub fn stage_top(&self) -> u32 {
        self.stage_bottom_left.1 + self.stage_height
    }
}

#[derive(Debug)]
pub struct Attendee {
    pub x: u32,
    pub y: u32,
    pub tastes: Vec<i32>,
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
        id: id,
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
        pillars: problem_json
            .pillars
            .iter()
            .map(|p| {
                Ok(Pillar {
                    center: (try_u32(p.center.0)?, try_u32(p.center.1)?),
                    radius: try_u32(p.radius)?,
                })
            })
            .collect::<Result<_>>()?,
    };
    Ok(problem)
}

pub fn read_problem(id: u32) -> Problem {
    try_read_problem(id).expect("Cannot read problem {}")
}

#[derive(Debug, Serialize)]
struct Placement {
    x: f64,
    y: f64,
}

#[derive(Debug, Serialize)]
struct Submission {
    placements: Vec<Placement>,
    volumes: Vec<f64>,
}

#[derive(Debug, Serialize)]
struct PostBody {
    problem_id: u32,
    contents: String,
}

pub fn submit_placements(
    token: &str,
    id: u32,
    placements: &[(f64, f64)],
    volumes: &[bool],
) -> Result<()> {
    let submission = Submission {
        placements: placements
            .iter()
            .map(|&(x, y)| Placement { x, y })
            .collect(),
        volumes: volumes
            .iter()
            .map(|&x| if x { 10.0 } else { 0.0 })
            .collect(),
    };
    let submission_str = serde_json::to_string(&submission).unwrap();
    let post_body = PostBody {
        problem_id: id,
        contents: submission_str,
    };

    let backup_file_path = format!(
        "submissions/{}_{}.json",
        id,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let mut backup_file = File::create(&backup_file_path)?;
    backup_file.write_all(serde_json::to_string(&post_body)?.as_bytes())?;
    eprintln!("Saved to {}", backup_file_path);

    reqwest::blocking::Client::builder()
        .build()?
        .post("https://api.icfpcontest.com/submission")
        .bearer_auth(token)
        .json(&post_body)
        .send()?;

    Ok(())
}

pub fn get_userboard(token: &str) -> Result<Vec<Option<i64>>> {
    let response = reqwest::blocking::Client::new()
        .get("https://api.icfpcontest.com/userboard")
        .bearer_auth(token)
        .send()?;
    let json: Value = response.json()?;
    let problems = json["Success"]["problems"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().map(|x| x as i64))
        .collect();
    Ok(problems)
}

pub fn get_pre_best_solution(token: &str, id: u32) -> Result<Option<Vec<(u32, u32)>>> {
    let response = reqwest::blocking::Client::new()
        .get(format!(
            "https://api.icfpcontest.com/submissions?offset=0&limit=1000&problem_id={}",
            id
        ))
        .bearer_auth(token)
        .send()?;
    let json: Value = response.json()?;
    let data = json["Success"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|x| {
            x["score"]
                .as_object()
                .map(|x| x.contains_key("Success"))
                .unwrap_or(false)
        })
        .filter(|x| x["submitted_at"].as_str().unwrap() < "2023-07-09T12:59:50.352656515Z")
        .max_by_key(|x| x["score"]["Success"].as_f64().unwrap() as i64);
    if let Some(data) = data {
        let id = data["_id"].as_str().unwrap();
        let url = format!(
            "https://api.icfpcontest.com/submission?submission_id={}",
            id
        );
        let response = reqwest::blocking::Client::new()
            .get(url)
            .bearer_auth(token)
            .send()?;
        let json: Value = response.json()?;
        let contens_str = json["Success"].as_object().unwrap()["contents"]
            .as_str()
            .unwrap();
        let contents: Value = serde_json::from_str(contens_str)?;
        assert!(!contents.as_object().unwrap().contains_key("volumes"));
        let placements = contents["placements"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| {
                Ok((
                    try_u32(x["x"].as_f64().unwrap())?,
                    try_u32(x["y"].as_f64().unwrap())?,
                ))
            })
            .collect::<Result<_>>()?;
        Ok(Some(placements))
    } else {
        info!("no best for {}", id);
        Ok(None)
    }
}
