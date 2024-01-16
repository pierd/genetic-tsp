use lazy_static::lazy_static;
use rand::{seq::SliceRandom, Rng};
use regex::Regex;
use std::{
    collections::{HashMap, HashSet},
    env::args,
    fs::OpenOptions,
    io::{Read, Write},
    mem,
    path::Path,
    thread,
    time::{Duration, Instant},
};

const TIME_LIMIT: Duration = Duration::from_secs(60 * 5);
const FINAL_PROCESSING_TIME: Duration = Duration::from_millis(200);

#[derive(Clone, Debug)]
struct Solution {
    path: Vec<usize>,
    cost: usize,
}
type NamedSolution = (&'static str, Solution);

lazy_static! {
    static ref CITY_REGEX: Regex = Regex::new(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s*$").unwrap();
}

struct Cities {
    cities: HashMap<usize, (usize, usize)>,
}
impl Cities {
    fn read(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let mut cities = HashMap::new();

        let mut buf = String::new();
        OpenOptions::new()
            .read(true)
            .open(path.as_ref())?
            .read_to_string(&mut buf)?;

        for line in buf.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(captures) = CITY_REGEX.captures(line) {
                let id = captures.get(1).expect("id").as_str().parse::<usize>()?;
                let x = captures.get(2).expect("x").as_str().parse::<usize>()?;
                let y = captures.get(3).expect("y").as_str().parse::<usize>()?;
                cities.insert(id, (x, y));
            }
        }

        Ok(Self { cities })
    }

    fn cost(&self, path: &[usize]) -> usize {
        let mut cost = 0;
        for (&id1, &id2) in path.iter().zip(path.iter().skip(1)) {
            let city1 = self.cities.get(&id1).expect("id1");
            let city2 = self.cities.get(&id2).expect("id2");
            cost += distance(city1, city2);
        }
        cost += distance(
            self.cities.get(path.last().unwrap()).expect("id1"),
            self.cities.get(path.first().unwrap()).expect("id2"),
        );
        cost
    }

    fn dummy_solution(&self) -> Solution {
        let mut path = Vec::with_capacity(self.cities.len());
        for id in self.cities.keys() {
            path.push(*id);
        }
        Solution {
            cost: self.cost(&path),
            path,
        }
    }

    fn random_solution<R>(&self, rng: &mut R) -> Solution
    where
        R: rand::Rng + ?Sized,
    {
        let mut path = Vec::with_capacity(self.cities.len());
        for id in self.cities.keys() {
            path.push(*id);
        }
        path.shuffle(rng);
        Solution {
            cost: self.cost(&path),
            path,
        }
    }
}

fn write_solution(path: impl AsRef<Path>, solution: &Solution) -> anyhow::Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path.as_ref())?;
    let Solution { path, cost } = solution;
    for id in path {
        write!(file, "{} ", *id)?;
    }
    writeln!(file, "{}", *cost)?;
    Ok(())
}

fn distance((x1, y1): &(usize, usize), (x2, y2): &(usize, usize)) -> usize {
    x1.abs_diff(*x2) + y1.abs_diff(*y2)
}

trait Solver {
    fn find_better(&mut self) -> Option<bool>;
    fn solution(&self) -> &Solution;
}

struct RandomSolver<'a> {
    cities: &'a Cities,
    best_solution: Solution,
    path: Vec<usize>,
}
impl<'a> RandomSolver<'a> {
    fn new(cities: &'a Cities) -> Self {
        let best_solution = cities.dummy_solution();
        let path = best_solution.path.clone();
        Self {
            cities,
            best_solution,
            path,
        }
    }
}
impl<'a> Solver for RandomSolver<'a> {
    fn find_better(&mut self) -> Option<bool> {
        self.path.shuffle(&mut rand::thread_rng());
        let cost = self.cities.cost(&self.path);
        if cost < self.best_solution.cost {
            mem::swap(&mut self.path, &mut self.best_solution.path);
            self.best_solution.cost = cost;
            Some(true)
        } else {
            Some(false)
        }
    }

    fn solution(&self) -> &Solution {
        &self.best_solution
    }
}

struct GreedyClosestSolver<'a> {
    cities: &'a Cities,
    best_solution: Solution,
    starting_cities_to_try: Vec<usize>,
}
impl<'a> GreedyClosestSolver<'a> {
    fn new(cities: &'a Cities) -> Self {
        let best_solution = cities.dummy_solution();
        let starting_cities_to_try = best_solution.path.clone();
        Self {
            cities,
            best_solution,
            starting_cities_to_try,
        }
    }
}
impl<'a> Solver for GreedyClosestSolver<'a> {
    fn find_better(&mut self) -> Option<bool> {
        // let Some(mut current_city) = self.starting_cities_to_try.pop() else {
        //     return None;
        // };
        let mut current_city = *self
            .starting_cities_to_try
            .choose(&mut rand::thread_rng())
            .unwrap();
        let mut path = Vec::with_capacity(self.cities.cities.len());
        let mut remaining_cities = self
            .cities
            .cities
            .keys()
            .copied()
            .filter(|c| *c != current_city)
            .collect::<HashSet<_>>();
        path.push(current_city);
        while !remaining_cities.is_empty() {
            let current_city_coords = self.cities.cities.get(&current_city).expect("current city");
            let next_city = *remaining_cities
                .iter()
                .min_by_key(|id| {
                    distance(
                        current_city_coords,
                        self.cities.cities.get(id).expect("next city"),
                    )
                })
                .expect("next city");
            path.push(next_city);
            remaining_cities.remove(&next_city);
            current_city = next_city;
        }
        let cost = self.cities.cost(&path);
        if cost < self.best_solution.cost {
            mem::swap(&mut self.best_solution.path, &mut path);
            self.best_solution.cost = cost;
            Some(true)
        } else {
            Some(false)
        }
    }

    fn solution(&self) -> &Solution {
        &self.best_solution
    }
}

fn cross(cities: &Cities, solution1: &Solution, solution2: &Solution) -> Solution {
    let mut rng = rand::thread_rng();
    let mut path = Vec::with_capacity(cities.cities.len());

    // add cities from solution1 in order
    path.extend(
        solution1
            .path
            .iter()
            .take(rng.gen_range(1..solution1.path.len())),
    );

    // find idx of last city in solution2
    let last_city = *path.last().unwrap();
    let mut idx = solution2
        .path
        .iter()
        .enumerate()
        .find_map(|(idx, city)| (*city == last_city).then_some(idx))
        .unwrap();
    let mut used_cities: HashSet<usize> = path.iter().copied().collect();

    // add cities from solution2 in order
    while path.len() < cities.cities.len() {
        idx = (idx + 1) % solution2.path.len();
        let city = solution2.path[idx];
        if !used_cities.contains(&city) {
            path.push(city);
            used_cities.insert(city);
        }
    }

    let cost = cities.cost(&path);
    Solution { path, cost }
}

fn opt2mutate(cities: &Cities, solution: &Solution) -> Solution {
    let mut rng = rand::thread_rng();
    let mut path = solution.path.clone();
    let i = rng.gen_range(0..(path.len() - 1));
    let j = rng.gen_range(i + 1..path.len());
    path[i..=j].reverse();
    // FIXME: we could calculate the new cost in constant time
    let cost = cities.cost(&path);
    Solution { path, cost }
}

struct GeneticSolver<'a, Crosser, Mutator> {
    cities: &'a Cities,
    population: Vec<Solution>,
    size: usize,
    crossings_per_generation: usize,
    mutations_per_generation: usize,
    crosser: Crosser,
    mutator: Mutator,
}
impl<'a, C, M> GeneticSolver<'a, C, M> {
    fn new(
        cities: &'a Cities,
        size: usize,
        crossings_per_generation: usize,
        crosser: C,
        mutations_per_generation: usize,
        mutator: M,
    ) -> Self {
        let mut population = vec![cities.random_solution(&mut rand::thread_rng()); size];
        population.sort_unstable_by_key(|s| s.cost);
        Self {
            cities,
            population,
            size,
            crossings_per_generation,
            crosser,
            mutations_per_generation,
            mutator,
        }
    }

    fn insert(&mut self, solution: Solution) {
        let idx = self
            .population
            .binary_search_by_key(&solution.cost, |s| s.cost)
            .unwrap_or_else(|idx| idx);
        self.population.insert(idx, solution);
    }
}
impl<'a, C, M> Solver for GeneticSolver<'a, C, M>
where
    C: Fn(&Cities, &Solution, &Solution) -> Solution,
    M: Fn(&Cities, &Solution) -> Solution,
{
    fn find_better(&mut self) -> Option<bool> {
        let best_cost = self.solution().cost;
        let rng = &mut rand::thread_rng();

        for _ in 0..self.crossings_per_generation {
            let mut chosen = self.population.choose_multiple(rng, 2);
            let solution1 = chosen.next().expect("solution1");
            let solution2 = chosen.next().expect("solution2");
            assert!(chosen.next().is_none());
            let new_solution = (self.crosser)(self.cities, solution1, solution2);
            self.population.push(new_solution);
        }

        for _ in 0..self.mutations_per_generation {
            let solution = self.population.choose(&mut rand::thread_rng()).unwrap();
            let mutated = (self.mutator)(self.cities, solution);
            self.population.push(mutated);
        }
        self.population.sort_unstable_by_key(|s| s.cost);
        self.population.truncate(self.size);

        let new_best_cost = self.solution().cost;
        Some(new_best_cost < best_cost)
    }

    fn solution(&self) -> &Solution {
        self.population.first().expect("population")
    }
}

fn keep_solving_until<S: Solver>(
    deadline: Instant,
    name: &'static str,
    tx: flume::Sender<NamedSolution>,
    solver: &mut S,
) {
    while Instant::now() < deadline {
        let Some(better) = solver.find_better() else {
            println!("no more solutions for {}", name);
            break;
        };
        if better {
            let solution = solver.solution();
            if tx.send((name, solution.clone())).is_err() {
                return;
            }
        }
    }
    let solution = solver.solution();
    tx.send((name, solution.clone())).ok();
}

fn main() {
    let start_time = Instant::now();
    let mut paths = args().skip(1);
    let input_path = paths.next().expect("input path");
    let output_path = paths.next().expect("output path");

    let cities = Cities::read(input_path).expect("read cities");
    let (tx, rx) = flume::unbounded::<NamedSolution>();
    let (sneaky_tx, sneaky_rx) = flume::unbounded::<Solution>();
    let mut best_solution = cities.dummy_solution();

    thread::scope(|s| {
        // solution collector
        let best_solution = &mut best_solution;
        s.spawn(move || {
            while start_time.elapsed() < TIME_LIMIT - FINAL_PROCESSING_TIME / 2 {
                if let Ok((name, solution)) = rx.recv_timeout(Duration::from_millis(1000)) {
                    println!("{:?} {} {}", start_time.elapsed(), name, solution.cost);
                    if solution.cost < best_solution.cost {
                        *best_solution = solution;
                        println!("best so far!");
                        if name != "genetic" {
                            sneaky_tx.send(best_solution.clone()).ok();
                        }
                    }
                }
            }
            // this would happen normally but putting it here for clarity
            // dropping rx will cause the other threads to exit
            drop(rx);
        });

        // random
        s.spawn(|| {
            let mut solver = RandomSolver::new(&cities);
            keep_solving_until(
                start_time + TIME_LIMIT - FINAL_PROCESSING_TIME,
                "random",
                tx.clone(),
                &mut solver,
            );
        });

        // greedy closest
        s.spawn(|| {
            let mut solver = GreedyClosestSolver::new(&cities);
            keep_solving_until(
                start_time + TIME_LIMIT - FINAL_PROCESSING_TIME,
                "greedy closest",
                tx.clone(),
                &mut solver,
            );
        });

        // genetic
        s.spawn(|| {
            let mut solver = GeneticSolver::new(&cities, 100, 20, cross, 5, opt2mutate);
            while Instant::now() < start_time + TIME_LIMIT - FINAL_PROCESSING_TIME {
                if let Ok(solution) = sneaky_rx.try_recv() {
                    solver.insert(solution);
                }
                let Some(better) = solver.find_better() else {
                    println!("no more solutions for genetic");
                    break;
                };
                if better {
                    let solution = solver.solution();
                    if tx.send(("genetic", solution.clone())).is_err() {
                        return;
                    }
                }
            }
            let solution = solver.solution();
            tx.send(("genetic", solution.clone())).ok();
        });
    });

    write_solution(output_path, &best_solution).expect("write solution");
}
