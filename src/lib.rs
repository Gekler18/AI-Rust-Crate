use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{self, Read};

mod learn;

#[derive(Debug, Serialize, Deserialize)]
#[derive(Clone)]
pub enum ActivationFunction {
    Sigmoid,
    ReLU,
    Linear,
    Tanh,
    Softmax
}

#[derive(Debug, Serialize, Deserialize)]
#[derive(Clone)]
pub struct Neuron {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub activation_function: ActivationFunction,
    pub inputs: Vec<usize>
}

impl Neuron {
    pub fn new(activation_function: ActivationFunction, bias: f64, weights: Vec<f64>, inputs: Vec<usize>) -> Neuron {
        return Neuron {
            activation_function,
            bias,
            weights,
            inputs
        }
    }

    pub fn clone(&self) -> Neuron {
        return Neuron {
            weights: self.weights.clone(),
            bias: self.bias.clone(),
            activation_function: self.activation_function.clone(),
            inputs: self.inputs.clone()
        }
    }

    pub fn calculate_output(&self, previous_layer_outputs: &[f64]) -> f64 {
        // Ensure the number of weights matches the number of neurons in the previous layer
        assert_eq!(self.weights.len(), previous_layer_outputs.len());

        let weighted_sum: f64 = self.weights.iter().zip(previous_layer_outputs.iter()).map(|(w, x)| w * x).sum();

        let input: f64 = weighted_sum + self.bias;

        let output: f64 = match self.activation_function {
            ActivationFunction::Sigmoid => 1.0 / (1.0 + (-input).exp()),
            ActivationFunction::ReLU => if input > 0.0 { input } else { 0.0 },
            ActivationFunction::Linear => input,
            ActivationFunction::Tanh => (input.exp() - (-input).exp()) / (input.exp() + (-input).exp()),
            ActivationFunction::Softmax => input.exp() / (1.0 + input.exp())
        };

        return output;
    }

    pub fn activation_derivative(self: &Self, output: f64) -> f64 {
        match self.activation_function {
            ActivationFunction::Sigmoid => {
                let sigmoid_output: f64 = 1.0 / (1.0 + (-output).exp());
                return sigmoid_output * (1.0 - sigmoid_output);
            }
            ActivationFunction::ReLU => {
                if output < 0.0 {
                    return 0.0;
                } else {
                    return 1.0;
                }
            }   
            _ => {
                println!("More activation derivatives have to be implemented");
                return 1.0;
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[derive(Clone)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
    pub inputs: Vec<f64>,
    pub position: usize
}

impl Layer {
    pub fn new(amount_of_neurons: i32, position: usize) -> Layer {
        let mut neurons: Vec<Neuron> = Vec::with_capacity(amount_of_neurons as usize);

        for _ in 0..amount_of_neurons {
            neurons.push(Neuron::new(ActivationFunction::Sigmoid, 0.0, vec![], vec![]));
        }

        return Layer {
            neurons,
            inputs: vec![],
            position
        }
    }

    fn calculate_layer_outputs(&mut self) -> Vec<f64> {
        let mut outputs: Vec<f64> = vec![];
        for neuron in &self.neurons {
            let mut neuron_inputs: Vec<f64> = vec![];

            for input_index in &neuron.inputs {
                neuron_inputs.push(self.inputs[*input_index]);
            }
            outputs.push(neuron.calculate_output(&neuron_inputs));
        }

        return outputs;
    }

    pub fn clone(&self) -> Layer {
        return Layer {
            neurons: self.neurons.clone(),
            inputs: self.inputs.clone(),
            position: self.position
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
}

impl NeuralNetwork {
    // 'learn' function in 'learn.rs'

    pub fn new(amount_of_layers: i32) -> NeuralNetwork {
        let mut neural_network: NeuralNetwork = NeuralNetwork {
            layers: vec![]
        };

        let mut layers: Vec<Layer> = vec![];

        for position in 0..amount_of_layers {
            layers.push(Layer::new(0, position as usize));
        }

        neural_network.layers = layers;

        return neural_network;
    }

    pub fn calculate_outputs(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        self.layers[0].inputs = inputs;

        for (index, _layer) in &mut self.layers.clone().iter().enumerate() {
            let output: Vec<f64> = self.layers[index].calculate_layer_outputs();

            if index + 1 != self.layers.len() {
                self.layers[index + 1].inputs = output;
            } else {
                return output;
            }
            
        }

        return vec![1.0];
    }

    pub fn cost(self: &Self, outputs: &Vec<Vec<f64>>, perfect_outputs: &Vec<Vec<f64>>) -> f64 {
        // Calculates the Cost(Loss function) of the Neural Network

        let mut costs: Vec<f64> = vec![];

        for (output_index, output) in outputs.iter().enumerate() {
            let mut cost: f64 = 0.0;

            for (index, perfect_output) in perfect_outputs[output_index].iter().enumerate() {
                cost += (output[index] - perfect_output).powf(2.0);
            }

            costs.push(cost);
        }

        let mut average_cost: f64 = 0.0;

        for cost in &costs {
            average_cost += cost;
        }

        average_cost = average_cost / costs.len() as f64;

        return average_cost;
    }

    pub fn save_neural_network_to_json(filename: &str, neural_network: &NeuralNetwork) -> io::Result<()> {
        // Creates a JSON file with the Neural Network architecture with weights and biases

        let file: File = OpenOptions::new().write(true).create(true).truncate(true).open(filename)?;
        serde_json::to_writer_pretty(file, neural_network)?;
        Ok(())
    }
    
    pub fn load_neural_network_from_json(filename: &str) -> io::Result<NeuralNetwork> {
        // Loads a Neural Network with it's weights and biases from a JSON file

        let file: File = File::open(filename).unwrap();
        
        let mut reader: io::BufReader<File> = io::BufReader::new(file);
        let mut contents: String = String::new();
        reader.read_to_string(&mut contents)?;
        let neural_network: NeuralNetwork = serde_json::from_str(&contents)?;
        Ok(neural_network)
    }
}