#![feature(generic_const_exprs)]
#![allow(incomplete_features, unused_variables)]
#![feature(associated_type_bounds)]

use std::{vec, usize, any::Any, error::Error, f32::consts::E};
use getrandom::getrandom;
use wasm_bindgen::{prelude::*, closure};

struct Neuron{
    weights: Vec<f64>,
    bias: f64,
    activation_function: Box<dyn Fn(f64) -> f64>
}
impl Neuron {
    fn new(number_of_weights : usize) -> Neuron {
                // TODO! Replace with random number generator
        let random_bias = 0.0;//(random_f64_0_1() - 0.5) * 2.0;
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
    fn bump_bias(&mut self, bump_ammount:f64){
        self.bias += bump_ammount;
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


// #[wasm_bindgen] 
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

    // Set up function to get around weird JS bindings
    pub fn back_propigate_js(&mut self, x_input: Vec<f64>, y_input: Vec<f64>, r_out: Vec<f64>, g_out : Vec<f64>, b_out: Vec<f64>, learning_rate:f64) -> (){
        assert!(x_input.len() == y_input.len(), "X and Y must be same length");
        let mut inputs: Vec<Vec<f64>> = Vec::with_capacity(x_input.len());
        for (i,x) in x_input.into_iter().enumerate() {
            inputs.push(vec![x,y_input[i]]);
        }
        let mut outputs: Vec<Vec<f64>> = Vec::with_capacity(r_out.len());
        for (i,x) in r_out.into_iter().enumerate() {
            outputs.push(vec![x,g_out[i], b_out[i]]);
        }
        assert!(outputs.len() == inputs.len(), "Inputs and Outputs must be same length");
        self.back_propigate(inputs, outputs, learning_rate)
    }

    pub fn back_propigate(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, learning_rate:f64) -> (){
        // Step through each input, feed forward through the network
        // Store the output or "state" from each layer.
        // Then compute cost based on desired outputs
        // make note of desired change for this layers weights and biases
        // then propigate backwards and make note of desired change (scaled to something?)
        // to each Layers weights and biases
        
        // After doing this for each input and output pair then we actually
        // apply the desired changes 

        // Layers[Neurons[Changes to be made to each weight]]
        let mut agregate_weight_changes : Vec<Vec<Vec<f64>>> = vec![vec![]; self.layers.len()];
        for (i,changes) in agregate_weight_changes.iter_mut().enumerate(){
            for j in 0..self.layers[i].nuerons.len() {
                changes.push(vec![0.0;self.layers[i].nuerons[j].weights.len()]);
            }
        }

        let mut agregate_bias_changes : Vec<Vec<f64>> = vec![vec![]; self.layers.len()];
        for (i,changes) in agregate_bias_changes.iter_mut().enumerate(){
            let mut asd = 0.0;
            for _ in 0..self.layers[i].nuerons.len() {
                changes.push(asd);
                asd += 1.0;
            }
        }

        // Vector of each layers error
        // let mut c = &vec![];
        for (top_level_input, input) in inputs.into_iter().enumerate(){
            let mut layer_outputs : Vec<Vec<f64>> = vec![];
            let mut total_error : Vec<Vec<f64>> = vec![vec![]; self.layers.len()];

            let mut tmp : Vec<f64> = input;
            for layer in self.layers.iter(){
                tmp = layer.feed_forward(&tmp);
                layer_outputs.push(tmp.clone());
            }

            // Error array is targets - layer_outputs
            let mut output_error : Vec<f64> = Vec::new();
            for (index,something) in layer_outputs[layer_outputs.len() - 1].iter().enumerate() {
                output_error.push(something - targets[top_level_input][index])
            }
            // TODO: cant hard code 3
            let mut hidden_error : Vec<f64> = vec![0.0;1];
            let my_index = 0;
            for (index,o_error) in output_error.iter().enumerate() {
                hidden_error[my_index] += self.layers[self.layers.len() - 1].nuerons[0].weights[0] * o_error * -1.0;
                hidden_error[my_index] += self.layers[self.layers.len() - 1].nuerons[1].weights[0] * o_error * -1.0;
                break;
                // hidden_error[index] += self.layers[self.layers.len() - 1].nuerons[index].weights[2] * o_error;

                // hidden_error.push(o_error - targets[top_level_input][index])
            }
            agregate_weight_changes[0][0] = hidden_error.clone();
            agregate_weight_changes[1][0] = output_error.clone();
            agregate_weight_changes[1][1] = output_error.clone();
            agregate_weight_changes[1][2] = output_error.clone();

            // agregate_weight_changes[1][0][1] = output_error[1];
            // agregate_weight_changes[1][0][2] = output_error[2];

            // agregate_weight_changes[1][1] = output_error[1];
            // agregate_weight_changes[1][2] = output_error[2];

            // for (i,l) in self.layers.iter().enumerate(){
            //     for (j,n) in (*l).nuerons.iter().enumerate(){
            //         println!("L:{}, N: {} \t{:?}", i, j , (*n).weights);
            //     }
            // }

            // for (i,l) in layer_outputs.iter().enumerate(){
            //     println!("Layer:{}, output: {:?}", i, *l);
            // }

            println!("oe: {:?}", output_error);
            println!("he: {:?}", hidden_error);



            // let mut layer_errors : 
            // for (i,output) in layer_outputs.iter().enumerate().rev(){
            //     let mut this_target : Vec<f64> = vec![];
            //     // Cacluate the targest for this layer 
            //     // (First time its the network targets, then we calculate based off of the layer errors)
            //     if i == (layer_outputs.len() - 1){
            //         this_target = targets[top_level_input].clone();
            //     } else {
            //         this_target = layer_errors[i + 1].clone();
            //     }
            //     // Remove the mut
            //     let this_target = this_target;

            //     let mut this_layer_error : Vec<f64> = Vec::new();
            //     for (t,l_t) in this_target.iter().enumerate(){
            //         this_layer_error.push(l_t )
            //     }

            //     // final_error.push(target - layer_outputs[layer_outputs.len() - 1][i]);
            // }

        };

        for (i,layer_to_change) in agregate_weight_changes.iter().enumerate(){
            for (j,weights) in layer_to_change.iter().enumerate(){
                for(k, weight) in weights.iter().enumerate(){
                    println!("changing l{},n{},w{} by {}",i,j,k,weight * learning_rate);
                    self.layers[i].nuerons[j].weights[k] += weight * learning_rate;
                }
            }
        }
        
        // for (i,layer_to_change) in agregate_bias_changes.iter().enumerate(){
        //     for (j,change) in layer_to_change.iter().enumerate(){
        //         self.layers[i].nuerons[j].bias += change;
        //     }
        // }
    }

    pub fn get_last_weights(&self) -> Vec<f64>{
        return self.layers[self.layers.len() - 1].nuerons.iter().map(|n| -> Vec<f64> {n.weights.clone()}).flatten().collect();
    }
    pub fn get_last_bias(&self) -> Vec<f64>{
        let mut biases : Vec<f64> = vec![];
        for l in self.layers.iter(){
            for n in (*l).nuerons.iter(){
                biases.push(n.bias)
            }
        }
        return biases;
    }
}

// ArrayType : [ Type ; Expression ]
pub struct Matrix<T, const WIDTH: usize, const HEIGHT: usize>{
    // Columns first, then row
    matrix: Vec<Vec<T>>,
}
impl<const WIDTH: usize, const HEIGHT: usize> Matrix<f64, WIDTH, HEIGHT> {
    fn new() -> Matrix<f64, WIDTH, HEIGHT>{
        Matrix { matrix: vec![vec![0.0;WIDTH];HEIGHT]}
    }
    fn new_closure<F:Fn(usize,usize)->f64> (closure:F) -> Matrix<f64,WIDTH,HEIGHT>{
        let mut wrapper = Vec::with_capacity(HEIGHT);
        for i in 0..HEIGHT{
            let mut inner = Vec::with_capacity(WIDTH);
            for j in 0..WIDTH{
                inner.push(closure(i,j));
            }
            wrapper.push(inner);
        }
        return Matrix{matrix:wrapper};
    }
    fn transpose(&self) -> Matrix<f64, HEIGHT, WIDTH>{
        let mut new_columns:Vec<Vec<f64>> = Vec::with_capacity(WIDTH);
        for _ in 0..WIDTH{
            new_columns.push(vec![0.0;HEIGHT])
        }
        for (i,row) in self.matrix.iter().enumerate(){
            for (j,number) in row.iter().enumerate(){
                new_columns[j][i] = *number;
            }
        }
        Matrix { matrix: new_columns}
    }

    fn add(&self, rhs: &Matrix<f64, WIDTH,HEIGHT>) -> Matrix<f64, WIDTH, HEIGHT>{
        let mut m: Matrix<f64, WIDTH, HEIGHT> = Matrix::new();
        for (i,row) in m.matrix.iter_mut().enumerate(){
            for (j,space) in row.iter_mut().enumerate(){
                *space = self.matrix[i][j] + rhs.matrix[i][j];
            }
        }
        return m;
    }

    fn dot_product<const WIDTH_RHS: usize> 
        (&self, rhs: &Matrix<f64, WIDTH_RHS, WIDTH>) 
            -> Matrix<f64, WIDTH_RHS, HEIGHT> 
    {
        let mut m : Matrix<f64,WIDTH_RHS, HEIGHT> =  Matrix::new();
        let mut running : f64;
        for i in 0..HEIGHT{
            for k in 0..WIDTH_RHS{
                running = 0.0;
                for j in 0..WIDTH{
                    running += self.matrix[i][j] * rhs.matrix[j][k];
                }
                m.matrix[i][k] = running;
            }
        }
        return m;
    }

}

impl<const WIDTH: usize, const HEIGHT: usize> std::fmt::Display for Matrix<f64,WIDTH,HEIGHT>{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f,"{}x{}",WIDTH, HEIGHT)?;
        for column in self.matrix.iter(){
            writeln!(f,"{:?}", column)?;
        }
        Ok(())
    }
}
#[derive(Debug)]
struct DotError{}

impl Error for DotError{}
impl std::fmt::Display for DotError{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}
struct SimpleMatrix{
    m:Vec<Vec<f64>>,
    width: usize,
    height: usize
}
impl SimpleMatrix{
    fn new(width: usize, height:usize) -> SimpleMatrix{
        SimpleMatrix { m: vec![vec![0.0;width];height], width, height}
    }
    fn new_closure<F:Fn(usize,usize)->f64> (width:usize,height:usize,closure:F) -> SimpleMatrix{
        let mut wrapper = Vec::with_capacity(height);
        for i in 0..height{
            let mut inner = Vec::with_capacity(width);
            for j in 0..width{
                inner.push(closure(i,j));
            }
            wrapper.push(inner);
        }
        return SimpleMatrix{m:wrapper, width, height};
    }
    fn transpose(&self) -> SimpleMatrix{
        let mut new_columns:Vec<Vec<f64>> = Vec::with_capacity(self.width);
        for _ in 0..self.width{
            new_columns.push(vec![0.0;self.height])
        }
        for (i,row) in self.m.iter().enumerate(){
            for (j,number) in row.iter().enumerate(){
                new_columns[j][i] = *number;
            }
        }
        SimpleMatrix { m: new_columns, width:self.height, height:self.width}
    }

    fn add(&self, rhs: &SimpleMatrix) -> Result<SimpleMatrix, DotError>{
        if self.width != rhs.width || self.height != rhs.height{
            return Err(DotError{})
        }

        let mut m = SimpleMatrix::new(self.width, self.height);
        for (i,row) in m.m.iter_mut().enumerate(){
            for (j,space) in row.iter_mut().enumerate(){
                *space = self.m[i][j] + rhs.m[i][j];
            }
        }
        return Ok(m);
    }

    fn dot_product (&self, rhs: &SimpleMatrix) -> Result<SimpleMatrix, DotError> {
        if self.width != rhs.height{
            return Err(DotError{})
        }
        let mut m = SimpleMatrix::new(rhs.width, self.height);
        let mut running : f64;
        for i in 0..self.height{
            for k in 0..rhs.width{
                running = 0.0;
                for j in 0..self.width{
                    running += self.m[i][j] * rhs.m[j][k];
                }
                m.m[i][k] = running;
            }
        }
        return Ok(m);
    }
}
impl std::fmt::Display for SimpleMatrix{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f,"{}x{}",self.width, self.height)?;
        for column in self.m.iter(){
            writeln!(f,"{:?}", column)?;
        }
        Ok(())
    }
}


struct SimpleTensor{
    weights: Vec<SimpleMatrix>,
    biases: Vec<Vec<f64>>,
}
impl SimpleTensor{
    fn new(layers: Vec<usize>) -> SimpleTensor{
        let mut l = Vec::with_capacity(layers.len());
        for i in 1..layers.len(){
            l.push(SimpleMatrix::new(layers[i - 1], layers[i]))
        }
        SimpleTensor { weights: l, biases: vec![vec![0.0];10]}
    }
    fn from_closure<F:Fn(usize,usize)->f64>(layers: Vec<usize>, closure:F) -> SimpleTensor{
        let mut l = Vec::with_capacity(layers.len());
        for i in 1..layers.len(){
            l.push(SimpleMatrix::new_closure(layers[i - 1], layers[i], &closure))
        }
        SimpleTensor { weights: l, biases: vec![vec![0.0];10]}
    }
    // fn feed_forward(inputs: Vec<f64>) -> Vec<f64>{

    // }
}

impl std::fmt::Display for SimpleTensor{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for m in self.weights.iter(){
            writeln!(f,"{}", m)?;
        }
        Ok(())
    }
}


fn sigmoid(input: f64) -> f64{
    1.0 / ( 1.0 + 2.71828_f64.powf(-input))
}

// #[wasm_bindgen]
pub fn random_f64_0_1() -> f64{
    let mut byts = [0;8];
    match getrandom(&mut byts){
        Ok(_) => {
            // println!("{:?}", byts);
            // let base: u32 = ;
            // let large_n = (n as u32) << 15;
            // let f32_bits = base | large_n;
            // let m = f32::from_bits(f32_bits);
            // 2.0 * ( m - 0.5 )
            // We have byts filled with random bytes and we need to 
            // Normalize it to [0,1) by changing the exponent term 
            // 0b0_01111111110_00000000000000000000000
            //   0_01111111111
            byts[0] = 0b0_0111111;
            byts[1] = (byts[1] | 0b1111_0000) & (byts[1] & 0b1111_1111);
            // Times 4 for some reason, experminets show we need to do this to normalize it
            // let or_mask:u64 = 0b0_01111111111_0000000000000000000000000000000000000000000000000000;
            // let and_mask:u64 = 0b0_01111111111_1111111111111111111111111111111111111111111111111111;

            // let num = u64::from_be_bytes(byts) | or_mask;
            // let num = num & and_mask;
            f64::from_be_bytes(byts) * 1.0
        },
        Err(_) => 0.5
    }
}
// #[wasm_bindgen]
pub fn random_i32(size: i32) -> i32{
    let mut byts = [0;4];
    match getrandom(&mut byts){
        Ok(_) => {
            i32::from_be_bytes(byts) % size
        },
        Err(_) => 0
    }
}
// #[wasm_bindgen]
pub struct FunctionTuple{
    red: Box<dyn Fn(f64, f64) -> f64>,
    green: Box<dyn Fn(f64, f64) -> f64>,
    blue: Box<dyn Fn(f64, f64) -> f64>,
}

// #[wasm_bindgen]
impl FunctionTuple{
    pub fn new() -> FunctionTuple{
        let r = generate_poly();
        let g = generate_poly();
        let b = generate_poly();
        FunctionTuple { red: r, green: g, blue: b }
    }
    pub fn red(&self, x:f64, y:f64) -> f64{
        (self.red)(x, y)
    }
    pub fn green(&self, x:f64, y:f64) -> f64{
        (self.green)(x, y)
    }    
    pub fn blue(&self, x:f64, y:f64) -> f64{
        (self.blue)(x, y)
    }
}

fn generate_poly() -> Box<dyn Fn(f64, f64) -> f64>{
    let num_of_terms = random_i32(4) + 2;
    let mut constants : Vec<f64> = vec![];
    for _ in 0..num_of_terms{
        constants.push(random_f64_0_1() / 4.0);
    }
    let b = random_i32(5);
    let c = random_i32(3);
    Box::new(move|x,y| -> f64 {
        let mut ouput : f64 = 0.0;
        for (i,constant) in constants.iter().enumerate(){
            ouput += constant * x.powf((i as i32 + 1).into());
        }
        ouput += b as f64;
        return ((ouput > (y + c as f64)) as i32).into();
    })
}
// #[wasm_bindgen(start)]
pub fn main() {
    let input = vec![vec![1.1],vec![1.2]];
    let i1: Matrix<f64, 1, 2> = Matrix::new_closure(|x,y| -> f64{
        // println!("{} {}",x,y);
        if x == 0 {
            1.1
        } else {
            1.2
        }
    });

    // let l1 : Matrix<f64, 4, 2> = Matrix::new_closure(|x,y| -> f64{random_f64_0_1()});
    // let l2 : Matrix<f64, 3 ,4> = Matrix::new_closure(|x,y| -> f64{random_f64_0_1()});

    let simple = SimpleTensor::from_closure(vec![2,4,3], |x,y| -> f64{random_f64_0_1()});
    println!("{}", simple);

    
    let simple = SimpleTensor::new(vec![2,4,3]);
    println!("{}", simple);

}

