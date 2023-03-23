#[macro_use]
extern crate impl_ops;

use std::{
    cell::RefCell,
    collections::HashSet,
    fmt::{self, Debug},
    hash::{Hash, Hasher},
    ops,
    rc::Rc,
};
use uuid::Uuid;

#[derive(Clone)]
struct Value(Rc<RefCell<ValueData>>);

struct ValueData {
    data: f64,
    grad: f64,
    uuid: Uuid,
    _prev: Vec<Value>,
    _backward: Option<fn(value: &ValueData)>,
}

impl ValueData {
    fn new(data: f64) -> ValueData {
        ValueData {
            data,
            grad: 0.0,
            uuid: Uuid::new_v4(),
            _prev: Vec::new(),
            _backward: None,
        }
    }
}

impl_op_ex!(+ |a: &Value, b: &Value| -> Value {
    let out = Value::from(a.borrow().data + b.borrow().data);
    out.borrow_mut()._prev = vec![a.clone(), b.clone()];
    out.borrow_mut()._backward = Some(|value: &ValueData| {
        value._prev[0].borrow_mut().grad += value.grad;
        value._prev[1].borrow_mut().grad += value.grad;
    });
    out
});

impl_op_ex!(*|a: &Value, b: &Value| -> Value {
    let out = Value::from(a.borrow().data * b.borrow().data);
    out.borrow_mut()._prev = vec![a.clone(), b.clone()];
    out.borrow_mut()._backward = Some(|value: &ValueData| {
        let a_data = value._prev[0].borrow().data;
        let b_data = value._prev[1].borrow().data;
        value._prev[0].borrow_mut().grad += b_data * value.grad;
        value._prev[1].borrow_mut().grad += a_data * value.grad;
    });
    out
});

impl_op_ex_commutative!(+ |a: &Value, b: f64| -> Value { a + Value::from(b) });
impl_op_ex_commutative!(* |a: &Value, b: f64| -> Value { a * Value::from(b) });

impl_op_ex!(- |a: &Value| -> Value { a * (-1.0) });
impl_op_ex!(- |a: &Value, b: &Value| -> Value { a + (-b) });
impl_op_ex!(/ |a: &Value, b: &Value| -> Value { a * b.pow(-1.0) });
impl_op_ex!(+= |a: &mut Value, b: &Value| { *a = &*a + b });
impl_op_ex!(*= |a: &mut Value, b: &Value| { *a = &*a * b });
impl_op_ex!(/ |a: &Value, b: f64| -> Value { a / Value::from(b) });
impl_op_ex!(/ |a: f64, b: &Value| -> Value { Value::from(a) / b });

impl Value {
    fn new(value: ValueData) -> Value {
        Value(Rc::new(RefCell::new(value)))
    }

    fn relu(&self) -> Value {
        let out = Value::from(self.borrow().data.max(0.0));
        out.borrow_mut()._prev = vec![self.clone()];
        out.borrow_mut()._backward = Some(|value: &ValueData| {
            value._prev[0].borrow_mut().grad += if value.data > 0.0 { value.grad } else { 0.0 };
        });
        out
    }

    fn pow(&self, power: f64) -> Value {
        let out = Value::from(self.borrow().data.powf(power));
        out.borrow_mut()._prev = vec![self.clone(), Value::from(power)];
        out.borrow_mut()._backward = Some(|value: &ValueData| {
            let base = value._prev[0].borrow().data;
            let p = value._prev[1].borrow().data;
            value._prev[0].borrow_mut().grad += p * base.powf(p - 1.0) * value.grad;
        });
        out
    }

    fn backward(&self) {
        let mut topo: Vec<Value> = vec![];
        let mut visited: HashSet<Value> = HashSet::new();
        self._build_topo(&mut topo, &mut visited);
        topo.reverse();

        self.borrow_mut().grad = 1.0;
        for v in topo {
            if let Some(backprop) = v.borrow()._backward {
                backprop(&v.borrow());
            }
        }
    }

    fn _build_topo(&self, topo: &mut Vec<Value>, visited: &mut HashSet<Value>) {
        if visited.insert(self.clone()) {
            self.borrow()._prev.iter().for_each(|child| {
                child._build_topo(topo, visited);
            });
            topo.push(self.clone());
        }
    }
}

impl<T: Into<f64>> From<T> for Value {
    fn from(t: T) -> Value {
        Value::new(ValueData::new(t.into()))
    }
}

impl ops::Deref for Value {
    type Target = Rc<RefCell<ValueData>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let v = &self.borrow();
        write!(f, "data={} grad={}", v.data, v.grad)
    }
}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.borrow().uuid.hash(state);
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.borrow().uuid == other.borrow().uuid
    }
}

impl Eq for Value {}

fn main() {
    let a = Value::from(-4.0);
    let b = Value::from(2.0);

    let mut c = &a + &b;
    let mut d = &a * &b + &b.pow(3.0);

    c += &c + 1.0;
    c += 1.0 + &c + (-&a);
    d += &d * 2.0 + (&b + &a).relu();
    d += 3.0 * &d + (&b - &a).relu();

    let e = &c - &d;
    let f = e.pow(2.0);
    let mut g = &f / 2.0;
    g += 10.0 / &f;

    println!("{:.4}", g.borrow().data); // 24.7041

    g.backward();
    println!("{:.4}", a.borrow().grad); // 138.8338
    println!("{:.4}", b.borrow().grad); // 645.5773
}
