
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
}

fn main(){
    let n = Network::new(vec![2,10,5]);
    let n_inputs = vec![rand::thread_rng().gen::<f64>() - 0.5,rand::thread_rng().gen::<f64>() - 0.5];
    let guesses = n.feed_forward(n_inputs);
    println!("{:?}", guesses);
}
