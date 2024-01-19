#![feature(test)]

extern crate test;

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
impl Solution {
    fn last(&self) -> usize {
        *self.path.last().unwrap()
    }

    #[allow(dead_code)]
    fn rotate(&mut self, k: usize) {
        self.path.rotate_right(k);
    }

    fn distance_to_city(&self, city_idx: usize, cities: &Cities) -> usize {
        if city_idx == 0 {
            cities.distances[self.last()][self.path[city_idx]]
        } else {
            cities.distances[self.path[city_idx - 1]][self.path[city_idx]]
        }
    }

    fn distance_from_city(&self, city_idx: usize, cities: &Cities) -> usize {
        if city_idx == self.path.len() - 1 {
            cities.distances[self.path[city_idx]][self.path[0]]
        } else {
            cities.distances[self.path[city_idx]][self.path[city_idx + 1]]
        }
    }

    fn distance_to_and_from_city(&self, city_idx: usize, cities: &Cities) -> usize {
        if city_idx == 0 {
            cities.distances[self.last()][self.path[city_idx]]
                + cities.distances[self.path[city_idx]][self.path[city_idx + 1]]
        } else if city_idx == self.path.len() - 1 {
            cities.distances[self.path[city_idx - 1]][self.path[city_idx]]
                + cities.distances[self.path[city_idx]][self.path[0]]
        } else {
            cities.distances[self.path[city_idx - 1]][self.path[city_idx]]
                + cities.distances[self.path[city_idx]][self.path[city_idx + 1]]
        }
    }

    fn swap(&mut self, i: usize, j: usize, cities: &Cities) {
        let removed_cost =
            self.distance_to_and_from_city(i, cities) + self.distance_to_and_from_city(j, cities);
        self.path.swap(i, j);
        let added_cost =
            self.distance_to_and_from_city(i, cities) + self.distance_to_and_from_city(j, cities);
        self.cost = self.cost - removed_cost + added_cost;
    }

    fn reverse(&mut self, i: usize, j: usize, cities: &Cities) {
        let removed_cost = self.distance_to_city(i, cities) + self.distance_from_city(j, cities);
        self.path[i..=j].reverse();
        let added_cost = self.distance_to_city(i, cities) + self.distance_from_city(j, cities);
        self.cost = self.cost - removed_cost + added_cost;
    }
}

#[test]
fn test_swap() {
    let cities = Cities::read("inputs/bier127.tsp").unwrap();
    for i in 0..cities.len() {
        for j in 0..cities.len() {
            let mut solution = cities.dummy_solution();
            solution.swap(i, j, &cities);
            assert_eq!(solution.cost, cities.cost(&solution.path));
        }
    }
}

#[test]
fn test_reverse() {
    let cities = Cities::read("inputs/bier127.tsp").unwrap();
    for i in 0..cities.len() - 1 {
        for j in i + 1..cities.len() {
            let mut solution = cities.dummy_solution();
            solution.reverse(i, j, &cities);
            assert_eq!(solution.cost, cities.cost(&solution.path));
        }
    }
}

#[bench]
fn bench_swap(b: &mut test::Bencher) {
    let cities = Cities::read("inputs/bier127.tsp").unwrap();
    let mut solution = cities.dummy_solution();
    b.iter(|| {
        solution.swap(3, 8, &cities);
    });
}

#[bench]
fn bench_reverse(b: &mut test::Bencher) {
    let cities = Cities::read("inputs/bier127.tsp").unwrap();
    let mut solution = cities.dummy_solution();
    b.iter(|| {
        solution.reverse(3, 8, &cities);
    });
}

type NamedSolution = (&'static str, Solution);

lazy_static! {
    static ref CITY_REGEX: Regex = Regex::new(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s*$").unwrap();
}

const MAX_CITIES: usize = 200;

struct Cities {
    cities: HashMap<usize, (usize, usize)>,
    distances: [[usize; MAX_CITIES]; MAX_CITIES],
}
impl Cities {
    fn len(&self) -> usize {
        self.cities.len()
    }

    fn read(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let mut cities = HashMap::with_capacity(MAX_CITIES);

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

        let mut distances = [[0; MAX_CITIES]; MAX_CITIES];
        for (&id1, &city1) in cities.iter() {
            for (&id2, &city2) in cities.iter() {
                distances[id1][id2] = distance(&city1, &city2);
            }
        }

        Ok(Self { cities, distances })
    }

    fn cost(&self, path: &[usize]) -> usize {
        let mut cost = 0;
        for (&id1, &id2) in path.iter().zip(path.iter().skip(1)) {
            cost += self.distances[id1][id2];
        }
        cost += self.distances[*path.last().unwrap()][*path.first().unwrap()];
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
        .create(true)
        .append(true)
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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

struct GreedyInsertSolver<'a> {
    cities: &'a Cities,
    best_solution: Solution,
}
impl<'a> GreedyInsertSolver<'a> {
    #[allow(dead_code)]
    fn new(cities: &'a Cities) -> Self {
        let best_solution = cities.dummy_solution();
        Self {
            cities,
            best_solution,
        }
    }
}
impl<'a> Solver for GreedyInsertSolver<'a> {
    fn find_better(&mut self) -> Option<bool> {
        let mut cities_to_insert = self.cities.cities.keys().copied().collect::<Vec<_>>();
        let mut path = Vec::with_capacity(self.cities.len());
        // insert the first 2 cities
        path.push(cities_to_insert.pop().unwrap());
        path.push(cities_to_insert.pop().unwrap());
        let mut cost = self.cities.cost(&path);

        while let Some(city_to_insert) = cities_to_insert.pop() {
            let (idx, cost_increase) = (0..path.len())
                .map(|idx| {
                    let removed_cost = if idx == 0 {
                        self.cities.distances[*path.last().unwrap()][path[idx]]
                    } else {
                        self.cities.distances[path[idx - 1]][path[idx]]
                    };
                    let added_cost = if idx == 0 {
                        self.cities.distances[*path.last().unwrap()][city_to_insert]
                            + self.cities.distances[city_to_insert][path[idx]]
                    } else {
                        self.cities.distances[path[idx - 1]][city_to_insert]
                            + self.cities.distances[city_to_insert][path[idx]]
                    };
                    (idx, added_cost - removed_cost)
                })
                .min_by_key(|(_, cost_increase)| *cost_increase)
                .unwrap();
            path.insert(idx, city_to_insert);
            cost += cost_increase;
        }
        debug_assert!(path.len() == self.cities.len());
        debug_assert!(cost == self.cities.cost(&path));
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

    // add cities from solution1 in order but starting from a random one
    path.extend(
        solution1
            .path
            .iter()
            .cycle()
            .skip(rng.gen_range(0..solution1.path.len()))
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
    let mut used_cities = [false; MAX_CITIES];
    for &city in &path {
        used_cities[city] = true;
    }

    // add cities from solution2 in order
    while path.len() < cities.len() {
        idx = (idx + 1) % solution2.path.len();
        let city = solution2.path[idx];
        if !used_cities[city] {
            path.push(city);
            used_cities[city] = true;
        }
    }

    let cost = cities.cost(&path);
    Solution { path, cost }
}

#[bench]
fn bench_cross(b: &mut test::Bencher) {
    let cities = Cities::read("inputs/bier127.tsp").unwrap();
    let solution1 = cities.dummy_solution();
    let solution2 = cities.dummy_solution();
    b.iter(|| {
        cross(&cities, &solution1, &solution2);
    });
}

fn opt2mutate(cities: &Cities, solution: &mut Solution) {
    let mut rng = rand::thread_rng();
    let i = rng.gen_range(0..(solution.path.len() - 1));
    let j = rng.gen_range(i + 1..solution.path.len());
    solution.reverse(i, j, cities);
}

fn swap2mutate(cities: &Cities, solution: &mut Solution) {
    let mut rng = rand::thread_rng();
    let i = rng.gen_range(0..(solution.path.len() - 1));
    let j = rng.gen_range(i + 1..solution.path.len());
    solution.swap(i, j, cities);
}

struct GeneticSolver<'a, Crosser, Mutator> {
    cities: &'a Cities,
    population: Vec<Solution>,
    size: usize,
    elite_size: usize,
    crossings_per_generation: usize,
    mutation_probability: f64,
    crosser: Crosser,
    mutator: Mutator,
}
impl<'a, C, M> GeneticSolver<'a, C, M> {
    fn new(
        cities: &'a Cities,
        size: usize,
        elite_size: usize,
        crossings_per_generation: usize,
        crosser: C,
        mutation_probability: f64,
        mutator: M,
    ) -> Self {
        let mut population = vec![cities.random_solution(&mut rand::thread_rng()); size];
        population.sort_unstable_by_key(|s| s.cost);
        Self {
            cities,
            population,
            size,
            elite_size,
            crossings_per_generation,
            crosser,
            mutation_probability,
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
    M: Fn(&'a Cities, &'_ mut Solution),
{
    fn find_better(&mut self) -> Option<bool> {
        let best_cost = self.solution().cost;
        let rng = &mut rand::thread_rng();

        let mut new_population = Vec::with_capacity(self.size);
        new_population.extend(self.population.iter().take(self.elite_size).cloned());
        for _ in 0..self.crossings_per_generation {
            let mut chosen = self.population.choose_multiple(rng, 2);
            let solution1 = chosen.next().expect("solution1");
            let solution2 = chosen.next().expect("solution2");
            debug_assert!(chosen.next().is_none());
            let mut new_solution = (self.crosser)(self.cities, solution1, solution2);
            if rng.gen_bool(self.mutation_probability) {
                (self.mutator)(self.cities, &mut new_solution);
            }
            new_population.push(new_solution);
        }

        // make sure we have enough solutions
        while new_population.len() < self.size {
            let mut chosen = self.population.choose(rng).cloned().unwrap();
            (self.mutator)(self.cities, &mut chosen);
            new_population.push(chosen);
        }

        self.population = new_population;
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

#[allow(clippy::too_many_arguments)]
fn run_genetic_in_context<'a>(
    deadline: Instant,
    name: &'static str,
    tx: flume::Sender<NamedSolution>,
    polinate_rx: flume::Receiver<Solution>,
    cities: &'a Cities,
    size: usize,
    crosser: fn(&Cities, &Solution, &Solution) -> Solution,
    mutator: fn(&'a Cities, &'_ mut Solution),
) {
    let mut solver = GeneticSolver::new(
        cities,
        size,
        size / 4,
        size - size / 4,
        crosser,
        0.2,
        mutator,
    );
    let mut generation = 0;
    while Instant::now() < deadline {
        while let Ok(solution) = polinate_rx.try_recv() {
            solver.insert(solution);
        }
        let Some(better) = solver.find_better() else {
            println!("no more solutions for {}", name);
            break;
        };
        generation += 1;
        if better {
            let solution = solver.solution();
            if tx.send((name, solution.clone())).is_err() {
                return;
            }
        }
    }
    println!("{} ran for {} generations", name, generation);
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
    let mut best_solution = cities.dummy_solution();

    thread::scope(|s| {
        // random
        // s.spawn(|| {
        //     let mut solver = RandomSolver::new(&cities);
        //     keep_solving_until(
        //         start_time + TIME_LIMIT - FINAL_PROCESSING_TIME,
        //         "random",
        //         tx.clone(),
        //         &mut solver,
        //     );
        // });

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

        // greedy insert
        s.spawn(|| {
            let mut solver = GreedyInsertSolver::new(&cities);
            keep_solving_until(
                start_time + TIME_LIMIT - FINAL_PROCESSING_TIME,
                "greedy insert",
                tx.clone(),
                &mut solver,
            );
        });

        // genetic
        let mut polinate_txs = Vec::new();
        for id in 0..5 {
            let tx = tx.clone();
            let cities = &cities;
            // channel for crosspolination of different genetic solvers
            let (polinate_tx, polinate_rx) = flume::unbounded::<Solution>();
            polinate_txs.push(polinate_tx);
            let name = format!("genetic{}o", id).leak(); // hack but we don't leak too much
            let size = cities.len() * (id / 2 + 1);
            s.spawn(move || {
                run_genetic_in_context(
                    start_time + TIME_LIMIT - FINAL_PROCESSING_TIME,
                    name,
                    tx,
                    polinate_rx,
                    cities,
                    size,
                    cross,
                    opt2mutate,
                );
            });
        }
        for id in 0..5 {
            let tx = tx.clone();
            let cities = &cities;
            // channel for crosspolination of different genetic solvers
            let (polinate_tx, polinate_rx) = flume::unbounded::<Solution>();
            polinate_txs.push(polinate_tx);
            let name = format!("genetic{}s", id).leak(); // hack but we don't leak too much
            let size = cities.len() * (id / 2 + 1);
            s.spawn(move || {
                run_genetic_in_context(
                    start_time + TIME_LIMIT - FINAL_PROCESSING_TIME,
                    name,
                    tx,
                    polinate_rx,
                    cities,
                    size,
                    cross,
                    swap2mutate,
                );
            });
        }

        // solution collector
        let best_solution = &mut best_solution;
        let output_path = &output_path;
        let cities = &cities;
        s.spawn(move || {
            while start_time.elapsed() < TIME_LIMIT - FINAL_PROCESSING_TIME / 2 {
                if let Ok((name, solution)) = rx.recv_timeout(Duration::from_millis(1000)) {
                    if solution.cost < best_solution.cost {
                        println!("{:?} {} {}", start_time.elapsed(), name, solution.cost);
                        *best_solution = solution;
                        assert!(best_solution.path.len() == cities.len());
                        assert!(best_solution.cost == cities.cost(&best_solution.path));
                        write_solution(output_path, best_solution).expect("write solution");
                        // send the solution to all the genetic solvers
                        for polinate_tx in &polinate_txs {
                            polinate_tx.send(best_solution.clone()).ok();
                        }
                    }
                }
            }
            // this would happen normally but putting it here for clarity
            // dropping rx will cause the other threads to exit
            drop(rx);
        });
    });

    assert!(best_solution.path.len() == cities.len());
    assert!(best_solution.cost == cities.cost(&best_solution.path));
    write_solution(output_path, &best_solution).expect("write solution");
}
