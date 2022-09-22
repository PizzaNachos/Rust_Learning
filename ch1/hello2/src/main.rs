fn greet_world(){
    println!("Hello, world!");
    let southern_germany = "Grüß Dich";
    let japan = "ハローワールド";
    let countries = [southern_germany, japan];
    for country in countries.iter(){
        println!("{}", &country);
    }
}

fn main() {
    greet_world();
}
