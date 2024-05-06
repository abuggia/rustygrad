#![allow(dead_code)]

use std::ops::{Add, Mul, Sub, Div};
use std::rc::Rc;
use std::cell::{Ref, RefMut, RefCell};
use std::iter::Sum;
use std::fmt;

use rand::distributions::{Distribution, Uniform};

#[derive(Debug)]
enum Op {
    Scalar,
    Add(Val, Val),
    Sub(Val, Val),
    Mul(Val, Val),
    Div(Val, Val),
    Pow(Val, f64),
    Relu(Val)
}

#[derive(Debug)]
struct Data {
    pub data: RefCell<f64>,
    pub grad: RefCell<f64>,
    pub op: Op
}

#[derive(Debug, Clone)]
pub struct Val(Rc<Data>);

impl Val {
    fn new(data: f64, op: Op) -> Self {
        Val(Rc::new(Data {
            data: RefCell::new(data),
            grad: RefCell::new(0.0.into()),
            op
        }))
    }

    pub fn uniform(size: usize) -> Vec<Val> {
        let between = Uniform::<f64>::new(-1., 1.);
        let mut rng = rand::thread_rng();
        (0..size)
            .map(|_| val(between.sample(&mut rng)))
            .collect()
    }

    pub fn back(&self) {
        *self.grad_mut() = 1.;
        self._back();
    }

    fn _back(&self) {
        match &self.0.op {
            Op::Add(v1, v2) => {
                 *v1.grad_mut() += *self.grad();
                 *v2.grad_mut() += *self.grad();
                 
                 v1._back();
                 v2._back();
            },

            Op::Sub(v1, v2) => {
                 *v1.grad_mut() += *self.grad();
                 *v2.grad_mut() -= *self.grad();
                 
                 v1._back();
                 v2._back();
            },

            Op::Mul(v1, v2) => {
                 *v1.grad_mut() += *v2.data() * *self.grad();
                 *v2.grad_mut() += *v1.data() * *self.grad();
                 
                 v1._back();
                 v2._back();
            },

            Op::Div(v1, v2) => {
                let high = *v1.data();
                let low = *v2.data();

                 *v1.grad_mut() += (1. / low) * *self.grad();
                 *v2.grad_mut() += ((-1. * high) / low.powf(2.)) * *self.grad();
                 
                 v1._back();
                 v2._back();
            },

            Op::Pow(v1, power) => {
                let base = *v1.data();
                 *v1.grad_mut() += (power * base.powf(power - 1.)) * *self.grad();
                 
                 v1._back();
            },

            Op::Relu(val) => {
                if *val.data() > 0. {
                    *val.grad_mut() += *self.grad();
                }
                val._back();
            },

            Op::Scalar => return,
        }
    }

    pub fn data(&self) -> Ref<'_, f64> {
        self.0.data.borrow()
    }

    pub fn data_mut(&self) -> RefMut<'_, f64> {
        self.0.data.borrow_mut()
    }

    pub fn grad(&self) -> Ref<'_, f64> {
        self.0.grad.borrow()
    }

    pub fn grad_mut(&self) -> RefMut<'_, f64> {
        self.0.grad.borrow_mut()
    }

    /*
    pub fn op(&self) -> String {
        self.0.op.to_string();
    }
    */
}

impl Add for &Val {
    type Output = Val;

    fn add(self, rhs: Self) -> Self::Output {
        Val::new(
            (*self.data() + *rhs.data()).into(),
            Op::Add(Val(Rc::clone(&self.0)), Val(Rc::clone(&rhs.0)))
        )
    }
}

impl Sub for &Val {
    type Output = Val;

    fn sub(self, rhs: Self) -> Self::Output {
        Val::new(
            (*self.data() - *rhs.data()).into(),
            Op::Sub(Val(Rc::clone(&self.0)), Val(Rc::clone(&rhs.0)))
        )
    }
}

impl Mul for &Val {
    type Output = Val;

    fn mul(self, rhs: Self) -> Self::Output {
        Val::new(
            (*self.data() * *rhs.data()).into(),
            Op::Mul(Val(Rc::clone(&self.0)), Val(Rc::clone(&rhs.0)))
        )
    }
}

impl Div for &Val {
    type Output = Val;

    fn div(self, rhs: Self) -> Self::Output {
        Val::new(
            (*self.data() / *rhs.data()).into(),
            Op::Div(Val(Rc::clone(&self.0)), Val(Rc::clone(&rhs.0)))
        )
    }
}

impl fmt::Display for Val {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.data())
    }
}

pub trait Relu {
    fn relu(self) -> Val;
}

impl Relu for &Val {
    fn relu(self) -> Val {
        let data = if *self.data() > 0. { *self.data() } else { 0. };
        Val::new(data, Op::Relu(Val(Rc::clone(&self.0))))
    }
}

pub trait Pow {
    fn pow(self, power: f64) -> Val;
}

impl Pow for &Val {
    fn pow(self, power: f64) -> Val {
        let data = self.data().powf(power);
        Val::new(data, Op::Pow(Val(Rc::clone(&self.0)), power))
    }
}


pub fn val(x: f64) -> Val {
    Val::new(x, Op::Scalar)
}

pub fn vals(xs: Vec<f64>) -> Vec<Val> {
    xs.into_iter().map(|x| val(x)).collect()
}

impl Sum for Val {
    fn sum<I: Iterator<Item = Val>>(mut iter: I) -> Val {
        let start = iter.next().expect("must be empty");
        iter.fold(start, |acc, val| &acc + &val)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_mul() {
        let x = val(3.);
        let y = val(4.);
        let z = &x * &y;

        assert_eq!(12., *z.data());

        z.back();
        assert_eq!(4., *x.grad());
        assert_eq!(3., *y.grad());
    }

    #[test]
    fn test_add() {
        let x = val(3.);
        let y = val(4.);
        let z = &x + &y;

        assert_eq!(7., *z.data());

        z.back();
        assert_eq!(1., *x.grad());
        assert_eq!(1., *y.grad());
    }

    #[test]
    fn test_div() {
        let x = val(2.);
        let y = val(4.);
        let z = &x / &y;

        assert_eq!(0.5, *z.data());

        z.back();
        assert_eq!(0.25, *x.grad());
        assert_eq!(-1./8., *y.grad());
    }

    #[test]
    fn test_pow() {
        let x = val(2.);
        let y = &x.pow(3.);

        assert_eq!(8., *y.data());

        y.back();
        assert_eq!(12., *x.grad());
    }

    #[test]
    fn test_relu() {
        let x = val(2.);
        let y = &x.relu();

        assert_eq!(2., *y.data());

        y.back();
        assert_eq!(1., *x.grad());

        let x = val(-2.);
        let y = &x.relu();

        assert_eq!(0., *y.data());

        y.back();
        assert_eq!(0., *x.grad());
    }
}


