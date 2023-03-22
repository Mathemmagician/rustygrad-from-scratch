use std::{
    cell::RefCell,
    fmt::{self, Debug},
    ops,
    rc::Rc,
};

#[derive(Clone)]
struct Value(Rc<RefCell<ValueData>>);

struct ValueData {
    data: f64,
    grad: f64,
    _prev: Vec<Value>,
    _backward: Option<fn(value: &ValueData)>,
}

impl ValueData {
    fn new(data: f64) -> ValueData {
        ValueData {
            data,
            grad: 0.0,
            _prev: Vec::new(),
            _backward: None,
        }
    }
}

impl Value {
    fn new(value: ValueData) -> Value {
        Value(Rc::new(RefCell::new(value)))
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

impl ops::Add for Value {
    type Output = Value;

    fn add(self, rhs: Self) -> Self::Output {
        let out = Value::from(self.borrow().data + rhs.borrow().data);
        out.borrow_mut()._prev = vec![self.clone(), rhs.clone()];
        out.borrow_mut()._backward = Some(|value: &ValueData| {
            value._prev[0].borrow_mut().grad += value.grad;
            value._prev[1].borrow_mut().grad += value.grad;
        });
        out
    }
}

fn main() {
    let a = Value::from(1.0);
    let b = Value::from(-4.0);
    let c = a + b;

    println!("{:?}", c);  // data=-3 grad=0
}
