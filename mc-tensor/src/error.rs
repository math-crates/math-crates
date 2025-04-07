pub type MathResult<T> = Result<T, MathError>;

#[derive(Debug)]
pub enum MathErrorKind {
    WrongTensorRank,
}

#[derive(Debug)]
pub struct MathError {
    pub kind: MathErrorKind,
    pub desc: String,
}

impl MathError {
    pub fn new<T: ToString>(kind: MathErrorKind, desc: T) -> Self {
        MathError {
            kind,
            desc: desc.to_string(),
        }
    }
}
