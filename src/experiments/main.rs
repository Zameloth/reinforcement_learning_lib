use reinforcement_learning_lib::environments::line_world::line_world_dp;

fn main() {
    println!("Hello, world!");

    let lw = line_world_dp();

    println!("{:?}", lw)
}
