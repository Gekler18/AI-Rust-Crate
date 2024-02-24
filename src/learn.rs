use crate::{NeuralNetwork, Neuron};
use rand;
use rand::prelude::SliceRandom;

impl NeuralNetwork {
    pub fn learn(self: &mut Self, learn_rate: f64, mut inputs: Vec<Vec<f64>>, mut perfect_outputs: Vec<Vec<f64>>, amount_of_rounds: i128) {
        // The main function for the learning algorithm, which is meant to be called
        // by the user. Runs the backpropagation algorithm as many times as specified

        let cost_gradient: Vec<f64> = self.backpropagate(&inputs, &perfect_outputs);

        println!("Cost Gradient:\n{:?}", cost_gradient);

        for _ in 0..amount_of_rounds {
            let cost_gradient: Vec<f64> = self.backpropagate(&inputs, &perfect_outputs);
            let negative_cost_gradient: Vec<f64> = cost_gradient.iter().map(|&x| -x).collect();

            // Apply the Cost Gradient onto the weights and biases
            let mut index: usize = 0;
            for layer in self.layers.iter_mut() {
                for neuron in layer.neurons.iter_mut() {
                    for weight in neuron.weights.iter_mut() {
                        *weight += learn_rate * negative_cost_gradient[index];
                        index += 1;
                    }

                    neuron.bias += learn_rate * negative_cost_gradient[index];
                }
            }
            let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
            perfect_outputs.shuffle(&mut rng);
            inputs.shuffle(&mut rng);
        }
    }

    fn backpropagate(self: &Self, inputs: &Vec<Vec<f64>>, perfect_outputs: &Vec<Vec<f64>>) -> Vec<f64> {
        // The Cost Gradient will contain the gradients for weights first,
        // then for biases. The negative Cost Gradient shall be used to change
        // the weights and biases in the 'learn' function.

        let mut cost_gradients: Vec<Vec<f64>> = vec![];

        for (input_index, inputs) in inputs.iter().enumerate() {
            let perfect_outputs: &Vec<f64> = &perfect_outputs[input_index];            
            let mut cost_gradient: Vec<f64> = vec![];

            let output_layer: &crate::Layer = self.layers.last().expect("No output layer found");
            let mut delta: Vec<f64> = Vec::new();

            // Compute the delta for the output layer
            for (neuron_index, neuron) in output_layer.neurons.iter().enumerate() {
                let output: f64 = self.calculate_output(neuron, &inputs, self.layers.len() - 1);
                let error: f64 = output - perfect_outputs[neuron_index];
                delta.push(error * neuron.activation_derivative(output));
            } 

            // Backpropagate delta to previous layers
            for (layer_index, layer) in self.layers.iter().enumerate().rev().skip(1) {
                let next_layer = &self.layers[self.layers.len() - layer_index - 1];

                for (neuron_index, neuron) in layer.neurons.iter().enumerate() {
                    let mut error_sum: f64 = 0.0;

                    for (next_neuron_index, next_neuron) in next_layer.neurons.iter().enumerate() {
                        error_sum += delta[next_neuron_index] * next_neuron.weights[neuron_index];
                    }

                    let output: f64 = self.calculate_output(neuron, &inputs, layer_index);
                    delta.push(error_sum * neuron.activation_derivative(output));
                }
            }

            // Compute gradients for weights and biases
            for layer in self.layers.iter().rev() {
                for neuron_index in 0..layer.neurons.len() {
                    let neuron: &Neuron = &layer.neurons[neuron_index];
                    let delta: f64 = delta.pop().unwrap();

                    for _weight_index in 0..neuron.weights.len() {
                        // Derivative of the cost with respect to the weight
                        let weight_gradient: f64 = delta.clone();
                        cost_gradient.push(weight_gradient);
                    }

                    // Derivative of the cost with respect to the bias
                    let bias_gradient: f64 = delta.clone();

                    cost_gradient.push(bias_gradient);
                }
            }

            // Reverse the cost_gradient vector to match the original order of weights and biases
            cost_gradient.reverse();

            cost_gradients.push(cost_gradient);
        }

        // Calculate the average of gradients
        let num_columns: usize = cost_gradients[0].len();
        let num_rows: usize = cost_gradients.len();

        let mut column_sums: Vec<f64> = vec![0.0; num_columns];

        for row in cost_gradients {
            for (col_index, &value) in row.iter().enumerate() {
                column_sums[col_index] += value;
            }
        }

        let mut cost_gradient_averages: Vec<f64> = column_sums.iter().map(|&sum| sum / num_rows as f64).collect();

        for cost_gradient in &mut cost_gradient_averages {
            if *cost_gradient > 1.0 {
                *cost_gradient = 1.0
            }
        }
        
        return cost_gradient_averages;
    }

    fn calculate_output(self: &Self, neuron: &Neuron, inputs: &Vec<f64>, layer_index: usize) -> f64 {
        if layer_index == 0 {
            neuron.calculate_output(&inputs)
        } else {
            neuron.calculate_output(&self.layers[layer_index].inputs)
        }
    }
}