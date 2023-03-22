
struct Value {
    data: f64,
    grad: f64,
    _prev: Vec<Value>,
    _backward: Option<fn(value: &Value)>,
}

impl Value {
    fn new(data: f64) -> Value {
        Value {
            data,
            grad: 0.0,
            _prev: Vec::new(),
            _backward: None,
        }
    }
}

fn main() {

}
