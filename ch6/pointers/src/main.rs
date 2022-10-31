static B: [u8; 10] = [99, 97, 114, 114, 121, 116, 111, 119, 101, 108];
fn main() {
    let one = 42;
    let two = &one;
    println!("\none:{}, two: {:p}, B: {:p}", one, two, &B)
}
