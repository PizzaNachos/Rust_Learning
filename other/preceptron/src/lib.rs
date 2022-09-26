
use std::vec;
use getrandom::getrandom;
use wasm_bindgen::prelude::*;
struct Neuron{
    weights: Vec<f64>,
    bias: f64,
    activation_function: Box<dyn Fn(f64) -> f64>
}
impl Neuron {
    fn new(number_of_weights : usize) -> Neuron {
                // TODO! Replace with random number generator
        let random_bias = (random_f64_0_1() - 0.5) * 2.0;
        let mut random_weights = Vec::with_capacity(number_of_weights);
        for _ in 0..number_of_weights{
            // TODO! Replace with random number generator
            random_weights.push(random_f64_0_1() - 0.5);
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
            *weight = *weight + (tweak_ammount * (random_f64_0_1() - 0.5));
        }
        self.bias += tweak_ammount * (random_f64_0_1() - 0.5) * 4.0;
    }
    fn tweak_random_weight(&mut self, tweak_ammount:f64){
        // TODO! Replace with random number generator
        // let l = 
        let i : usize = (((random_f64_0_1() * self.weights.len() as f64).floor()) % self.weights.len() as f64) as usize;
        self.weights[i] += tweak_ammount * (random_f64_0_1() - 0.5);
        self.bias += tweak_ammount * (random_f64_0_1() - 0.5);
    }
}

struct Layer{
    nuerons: Vec<Neuron>
}
impl Layer {
    fn new(input_size : usize, layer_size : usize) -> Layer{
        let mut n = vec![];
        for _ in 0..layer_size{
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
    fn tweak_random_weight(&mut self, tweak_ammount:f64){
        // TODO! Replace with random number generator
        let i : usize = (((random_f64_0_1() * self.nuerons.len() as f64).floor()) % self.nuerons.len() as f64) as usize;

        // let i = 0;
        self.nuerons[i].tweak_random_weight(tweak_ammount);
    }
}

#[wasm_bindgen]
pub struct Network {
    layers : Vec<Layer>
}

#[wasm_bindgen]
impl Network {  
    pub fn new(layers: Vec<usize>) -> Network{
        let mut l = Vec::with_capacity(layers.len());
        for i in 1..layers.len(){
            l.push(Layer::new(layers[i - 1], layers[i])); 
        }
        Network { layers: l }
    }
    pub fn feed_forward(&self, inputs : Vec<f64>) -> Vec<f64> {
        let mut tmp : Vec<f64> = inputs;
        for layer in self.layers.iter(){
            tmp = layer.feed_forward(&tmp);
        }
        tmp
    }
    pub fn tweak_weights(&mut self, tweak_ammount:f64){
        for layer in self.layers.iter_mut() {
            layer.tweak_weights(tweak_ammount);
        }
    }
    pub fn tweak_random_weight(&mut self, tweak_ammount:f64){
        // TODO! Replace with random number generator
        let i : usize = (((random_f64_0_1() * self.layers.len() as f64).floor()) % self.layers.len() as f64) as usize;
        self.layers[i].tweak_random_weight(tweak_ammount);
    }
}


fn sigmoid(input: f64) -> f64{
    1.0 / ( 1.0 + 2.71828_f64.powf(-input))
}

#[wasm_bindgen]
pub fn random_f64_0_1() -> f64{
    let mut byts = [0;8];
    match getrandom(&mut byts){
        Ok(_) => {
            // We have byts filled with random bytes and we need to 
            // Normalize it to [0,1) by changing the exponent term 
            // 0b0_01111111110_00000000000000000000000
            byts[0] = 0b0_0111111;
            byts[1] = (byts[1] | 0b1110_0000) & (byts[1] & 0b1110_1111);
            // print!("{:?}", byts);
            let l = f64::from_be_bytes(byts) * 4.0;
            l
        },
        Err(_) => 0.5
    }
}

#[wasm_bindgen]
pub struct FunctionTuple{
    red: Box<dyn Fn(f64, f64) -> bool>,
    green: Box<dyn Fn(f64, f64) -> bool>,
    blue: Box<dyn Fn(f64, f64) -> bool>,
}

#[wasm_bindgen]
impl FunctionTuple{
    pub fn new() -> FunctionTuple{
        let r = generate_poly();
        let g = generate_poly();
        let b = generate_poly();
        FunctionTuple { red: r, green: g, blue: b }
    }
    pub fn red(&self, x:f64, y:f64) -> bool{
        (self.red)(x, y)
    }
    pub fn green(&self, x:f64, y:f64) -> bool{
        (self.green)(x, y)
    }    
    pub fn blue(&self, x:f64, y:f64) -> bool{
        (self.blue)(x, y)
    }
}

fn generate_poly() -> Box<dyn Fn(f64, f64) -> bool>{
    let i = 10.0;
    // fn create(x:f64, y:f64) -> f64{
    //     i
    // }
    Box::new(move|x,y| -> bool{
        return x > y;
    })
}

#[wasm_bindgen(start)]
pub fn main() -> Result<(), JsValue>{
    Ok(())
}

