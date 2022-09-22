use std::convert::TryInto;

fn main() {
    let integer:i32 = 100;
    let sixteen:i16 = 200;
    let float:f32 = 2.0;


    let tried = sixteen.try_into();

    match tried {
        Ok(val) => {
            if integer < val {
                print!("{}",val)
            }
        }
        Err(e) => {
            print!("{}", e)
        }
    }
}
