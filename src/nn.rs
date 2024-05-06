#![allow(dead_code)]

use crate::val::{Val, val, Relu};

#[derive(Debug)]
struct Neuron {
    weights: Vec<Val>,
    bias: Val,
    nonlin: bool
}

impl Neuron {
    fn new(size: usize, nonlin: bool) -> Self {
        Neuron {
            weights: Val::uniform(size),
            bias: val(0.),
            nonlin
        }
    }

    fn forward(&self, xs: &Vec<Val>) -> Val {
        let act = &self.bias + &xs
            .iter()
            .zip(self.weights.iter())
            .map(|(v, w)| v * w)
            .sum::<Val>();

        if self.nonlin { act.relu() } else { act }
    }

    fn parameters(&self) -> Vec<Val> {
        let mut ret = self.weights.clone();
        ret.push(self.bias.clone());
        ret
    }
}

struct Layer {
    neurons: Vec<Neuron>
}

impl Layer {

    fn new(nin: usize, nout: usize, nonlin: bool) -> Self {
        Layer {
          neurons: (0..nout).map(|_| Neuron::new(nin, nonlin)).collect()
        }
    }

    fn forward(&self, xs: &Vec<Val>) -> Vec<Val> {
        self.neurons
            .iter()
            .map(|neuron| neuron.forward(xs))
            .collect()
    }

    fn parameters(&self) -> Vec<Val> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}

pub struct Mlp {
    layers: Vec<Layer>
}

impl Mlp {
    pub fn new(nin: usize, mut nout: Vec<usize>) -> Self {
        let n = nout.len();
        let mut sz = vec![nin];
        sz.append(&mut nout);

        let layers = (0..n)
              .map(|i| Layer::new(sz[i], sz[i+1], i != n-1))
              .collect();

        Mlp { layers }
    }

    pub fn forward(&self, xs: Vec<Val>) -> Vec<Val> {
        self.layers
            .iter()
            .fold(xs, |xs, layer| layer.forward(&xs))
    }

    pub fn parameters(&self) -> Vec<Val> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    pub fn zero_grad(&self) {
        for parameter in self.parameters() {
            *parameter.grad_mut() = 0.;
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_0() {
        let deps = Val::uniform(10);
        let nn = Neuron::new(deps.len(), true);

        let _res = nn.forward(&deps);

        assert_eq!(10, deps.len())
    }
}
