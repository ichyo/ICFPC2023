mod io;

use io::read_problem;

fn main() {
    for i in 1..=45 {
        let problem = read_problem(i);
    }
}
