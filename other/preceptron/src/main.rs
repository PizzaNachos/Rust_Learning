
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

fn main(){
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
}
