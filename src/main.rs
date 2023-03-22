
struct Value {
    data: f64,
    grad: f64,
    _prev: Vec<Value>,
    _backward: Option<fn(value: &Value)>,
}

fn main() {

}
