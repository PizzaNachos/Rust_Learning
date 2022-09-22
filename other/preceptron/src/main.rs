extern crate sdl2;

use rand::prelude::*;

use sdl2::pixels::Color;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::rect::Point;
use std::time::Duration;
use sdl2::pixels;
// use sdl2::gfx::primitives::DrawRenderer;
struct Neuron{
    weights: Vec<f64>,
    bias: f64,
    activation_function: Box<dyn Fn(f64) -> f64>
}
impl Neuron {
    fn new(number_of_weights : usize) -> Neuron {
        let random_bias = rand::thread_rng().gen::<f64>()  - 0.5_f64;
        let weights_vector = vec![rand::thread_rng().gen::<f64>() - 0.5_f64; number_of_weights];

        fn sigmoid(input: f64) -> f64{
            1.0 / ( 1.0 + 2.71828_f64.powf(-input))
        }
        Neuron{
            weights: weights_vector,
            bias: random_bias,
            activation_function: Box::new(&sigmoid)
        }
    }

    fn guess(&self, inputs: &Vec<f64>) -> f64{
        let mut sum : f64 = 0.0;
        for (i, weight) in self.weights.iter().enumerate(){
            sum += weight * inputs[i];
        }
        sum += self.bias;
        (self.activation_function)(sum)
    }
}

fn main()  -> Result<(), String> {
    let n = Neuron::new(2);

    let mut n_inputs = vec![];
    for i in 0..10 {
        n_inputs.push(vec![]);
        for _ in 0..2 {
            n_inputs[i].push(rand::thread_rng().gen::<f64>() - 0.5);
        }
    }
    let mut guesses = vec![];
    for input in n_inputs {
        guesses.push(n.guess(&input))
    }
    println!("{:?}", guesses);


    let sdl_context = sdl2::init()?;
    let video_subsys = sdl_context.video()?;
    let window = video_subsys
        .window(
            "rust-sdl2_gfx: draw line & FPSManager",
            800,
            800,
        )
        .position_centered()
        .opengl()
        .build()
        .map_err(|e| e.to_string())?;

    let mut canvas = window.into_canvas().build().map_err(|e| e.to_string())?;

    canvas.set_draw_color(pixels::Color::RGB(0, 0, 0));
    canvas.clear();
    canvas.present();

    let mut lastx = 0;
    let mut lasty = 0;

    let mut events = sdl_context.event_pump()?;

    let point : Point = Point::new(100, 100);
    canvas.draw_point(point);
    // canvas.pixel(10 as i16, 10 as i16, 0xFF000FFu32)?;

    Ok(())
}
