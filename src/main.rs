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

fn main() {
    let a = Value::from(-4.0);
    println!("{:?}", a); // data=-4 grad=0

    a.borrow_mut().data = 7.5;
    println!("{:?}", a); // data=7.5 grad=0
}
