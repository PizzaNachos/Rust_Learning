
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

    fn back_propigate(&mut self, inputs: Vec<Vec<f64>>, outputs: Vec<Vec<f64>>,learning_rate:f64) -> (){
        // Step through each input, feed forward through the network
        // Store the output or "state" from each layer.
        // Then compute cost based on desired outputs
        // make note of desired change for this layers weights and biases
        // then propigate backwards and make note of desired change (scaled to something?)
        // to each Layers weights and biases
        
        // After doing this for each input and output pair then we actually
        // apply the desired changes 

        // Layers[Neurons[Changes to be made to each weight]]
        let weight_const =learning_rate;
        let mut agregate_weight_changes : Vec<Vec<Vec<f64>>> = vec![vec![]; self.layers.len()];
        for (i,changes) in agregate_weight_changes.iter_mut().enumerate(){
            for j in 0..self.layers[i].nuerons.len() {
                changes.push(vec![0.0;self.layers[i].nuerons[j].weights.len()]);
            }
        }

        let bias_const = learning_rate;
        let mut agregate_bias_changes : Vec<Vec<f64>> = vec![vec![]; self.layers.len()];
        for (i,changes) in agregate_bias_changes.iter_mut().enumerate(){
            let mut asd = 0.0;
            for _ in 0..self.layers[i].nuerons.len() {
                changes.push(asd);
                // c.push(asd);
                asd += 1.0;
            }
        }


        for (top_level_input, input) in inputs.into_iter().enumerate(){
            let mut layer_outputs : Vec<Vec<f64>> = vec![];
            
            let mut tmp : Vec<f64> = input;
            for layer in self.layers.iter(){
                tmp = layer.feed_forward(&tmp);
                layer_outputs.push(tmp.clone());
            }


            // Iterate backwards over the layers
            for layer_output_index in (0..layer_outputs.len()).rev(){
                let layer_output = &layer_outputs[layer_output_index];
                let mut cost : Vec<f64> = vec![0.0;layer_output.len()];
                for (j,output) in outputs[top_level_input].iter().enumerate()  {
                    cost[j] = *output - layer_output[j];
                }
 
                // Diff the cost and note what changes we want to make
                for (j, neural_cost) in cost.iter().enumerate(){
                    // Bump bias
                    agregate_bias_changes[layer_output_index][j] += neural_cost * random_f64_0_1() * bias_const;

                    // This one crashes now
                    let mut strong_activation_indexes_previous_layer : Vec<usize>  = vec![];
                    let mut average_activation = 0.0;
                    // if (layer_output_index - 1) > 0{
                        for activation in layer_outputs[layer_output_index].iter(){
                            average_activation += activation;
                        }
                        average_activation /= layer_outputs[layer_output_index].len() as f64;

                        for (act_ind,activation) in layer_outputs[layer_output_index].iter().enumerate(){
                            if *activation > average_activation {
                                strong_activation_indexes_previous_layer.push(act_ind);
                            }
                        }
                    // }

                    for (w_ind,weight) in agregate_weight_changes[layer_output_index][j].iter_mut().enumerate(){
                        if strong_activation_indexes_previous_layer.contains(&w_ind){
                            *weight += neural_cost * weight_const;
                        }
                    }


                    // agregate_weight_changes[layer_output_index][j] += 
                }
            }

        };
        // return c;

        for (i,layer_to_change) in agregate_weight_changes.iter().enumerate(){
            for (j,weights) in layer_to_change.iter().enumerate(){
                for(k, weight) in weights.iter().enumerate(){
                    self.layers[i].nuerons[j].weights[k] += weight;
                }
            }
        }
        
        for (i,layer_to_change) in agregate_bias_changes.iter().enumerate(){
            for (j,change) in layer_to_change.iter().enumerate(){
                self.layers[i].nuerons[j].bias += change;
            }
        }
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
pub fn random_i32(size: i32) -> i32{
    let mut byts = [0;4];
    match getrandom(&mut byts){
        Ok(_) => {
            i32::from_be_bytes(byts) % size
        },
        Err(_) => 0
    }
}
#[wasm_bindgen]
pub struct FunctionTuple{
    red: Box<dyn Fn(f64, f64) -> f64>,
    green: Box<dyn Fn(f64, f64) -> f64>,
    blue: Box<dyn Fn(f64, f64) -> f64>,
}

#[wasm_bindgen]
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
    let m = random_f64_0_1() / 4.0;
    let n = random_i32(4) + 1;
    let b : f64 = random_i32(10).into();
    Box::new(move|x,y| -> f64 {
        return (((m* x.powf(n.into()) + b) > y) as i32).into();
    })
    // let m = random_f64_0_1() * 20.0;
    // Box::new(move|x,y| -> f64 {
    //     return (((m) > y) as i32).into();
    // })
}

#[wasm_bindgen(start)]
pub fn main() -> Result<(), JsValue>{
    Ok(())
}

