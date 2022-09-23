
use std::vec;

use rand::prelude::*;
struct Neuron{
    weights: Vec<f64>,
    bias: f64,
    activation_function: Box<dyn Fn(f64) -> f64>
}
impl Neuron {
    fn new(number_of_weights : usize) -> Neuron {
        let random_bias = rand::thread_rng().gen::<f64>()  - 0.5_f64;
        let mut random_weights = Vec::with_capacity(number_of_weights);
        for i in 0..number_of_weights{
            random_weights.push(rand::thread_rng().gen::<f64>() - 0.5_f64);
        }
        let weights_vector = random_weights;

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

    fn tweak_weights(&mut self, tweak_ammount: f64){
        for weight in self.weights.iter_mut(){
            *weight = *weight + (tweak_ammount * rand::thread_rng().gen::<f64>());
        }
    }
}

struct Layer{
    nuerons: Vec<Neuron>
}
impl Layer {
    fn new(input_size : usize, layer_size : usize) -> Layer{
        let mut n = vec![];
        for i in 0..layer_size{
            n.push(Neuron::new(input_size))
        }
        Layer { nuerons: n }
    }

    fn feed_forward(&self, inputs : &Vec<f64>) -> Vec<f64>{
        let mut outputs : Vec<f64> = Vec::with_capacity(self.nuerons.len());
        for neuron in self.nuerons.iter() {
            outputs.push(neuron.guess(inputs));
        }
        outputs
    }
    fn tweak_weights(&mut self, tweak_ammount:f64){
        for n in self.nuerons.iter_mut(){
            n.tweak_weights(tweak_ammount);
        }
    }
}

struct Network {
    layers : Vec<Layer>
}
impl Network {  
    fn new(layers: Vec<usize>) -> Network{
        let mut l = Vec::with_capacity(layers.len());
        for i in 1..layers.len(){
            l.push(Layer::new(layers[i - 1], layers[i])); 
        }
        Network { layers: l }
    }
    fn feed_forward(&self, inputs : Vec<f64>) -> Vec<f64> {
        let mut tmp : Vec<f64> = inputs;
        for layer in self.layers.iter(){
            tmp = layer.feed_forward(&tmp);
        }
        tmp
    }
    fn tweak_weights(&mut self, tweak_ammount:f64){
        for layer in self.layers.iter_mut() {
            layer.tweak_weights(tweak_ammount);
        }
    }
}

fn main(){
    let mut n = Network::new(vec![2,100,1]);
    let mut tweak_ammount : f64 = 0.0005;

    loop{
        tweak_ammount += 0.001 * (rand::thread_rng().gen::<f64>() - 0.5);
        n.tweak_weights(tweak_ammount);
        let mut outputs = vec![];
        let x_size : i32 = 100;
        let y_size : i32 = 50;
        for i in 0..y_size{
            outputs.push(vec![]);
            for j in 0..x_size{
                outputs[i as usize].push(n.feed_forward(vec![(i - y_size / 2).into(),(j - y_size / 2).into()]));
            }
        }
        render_vectors(outputs);
    }
}

fn render_vectors(inputs: Vec<Vec<Vec<f64>>>){
    print!("{esc}[2J{esc}[1;1H", esc = 27 as char);
    for row in inputs {
        for col in row {
            let num = col[0] * 100.0;
            // print!("{}",num.round());
            match  num as i32 {
                0..=10 => print!(" "),
                11..=20 => print!("."),
                21..=40 => print!(","),
                41..=50 => print!("o"),
                51..=60 => print!("x"),
                61..=80 => print!("%"),
                81..=90 => print!("$"),
                91..=100 => print!("#"),
                _ => print!(" ")
            }   
        }
        println!("");
    }
}

fn sigmoid(input: f64) -> f64{
    1.0 / ( 1.0 + 2.71828_f64.powf(-input))
}