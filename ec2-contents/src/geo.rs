/*
Real distance_sp(const Segment &s, const Point &p) {
  Point r = projection(s, p);
  if(is_intersect_sp(s, r)) return abs(r - p);
  return min(abs(s.a - p), abs(s.b - p));
}
*/

use std::ops::Sub;

#[derive(Clone, Copy)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    #[inline]
    pub fn new(x: f64, y: f64) -> Point {
        Point { x, y }
    }

    #[inline]
    pub fn arctan(&self) -> f64 {
        self.y.atan2(self.x)
    }

    #[inline]
    pub fn norm(&self) -> f64 {
        self.x.hypot(self.y)
    }
}

impl Sub for Point {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

pub type Segment = (Point, Point);

fn dist_sq(p: Point, q: Point) -> f64 {
    (p.x - q.x).powi(2) + (p.y - q.y).powi(2)
}
pub fn dist_sq_segment_point(s: Segment, p: Point) -> f64 {
    let l2 = dist_sq(s.0, s.1);
    if l2 == 0.0 {
        return dist_sq(p, s.0);
    }
    let t = ((p.x - s.0.x) * (s.1.x - s.0.x) + (p.y - s.0.y) * (s.1.y - s.0.y)) / l2;
    let t = t.clamp(0.0, 1.0);
    dist_sq(
        p,
        Point {
            x: s.0.x + t * (s.1.x - s.0.x),
            y: s.0.y + t * (s.1.y - s.0.y),
        },
    )
}
