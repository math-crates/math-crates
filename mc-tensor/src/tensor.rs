use std::{fmt::Debug, ops::Range};

use crate::error::*;

/// Tensor index range
pub type Idx = Range<usize>;

/// Data type used for indexes declaration in tensors.
/// `[4, 4, 4]` declares three indexes each of dimention 4.
pub type Indexes<const R: usize> = [Idx; R];

/// It declares an index of dimention `dim`.
pub fn idx(dim: usize) -> Idx {
    0..dim
}

pub fn const_idx(val: usize) -> Idx {
    val..val
}

/// Tensor structure.
/// Generic type `CV` represents a number of covarianе indeces (bottom in math notation) and
/// `CR` represents a number of contravariant (top in math notation) indeces.
#[derive(Debug)]
pub struct Tensor<T, const R: usize, const CV: usize, const CR: usize> {
    inner: [T; R],

    // covariant indexes lengths
    cov: Indexes<CV>,

    // contravariant indexes lengths
    contr: Indexes<CR>,
}

impl<T: Debug, const R: usize, const CV: usize, const CR: usize> Tensor<T, R, CV, CR> {
    pub fn try_new((cov, contr): (Indexes<CV>, Indexes<CR>), inner: [T; R]) -> MathResult<Self> {
        let cv_prod = Self::indexes_prod(&cov);
        let cr_prod = Self::indexes_prod(&contr);
        let cv_cr_prod = cv_prod * cr_prod;

        if inner.len() != cv_cr_prod {
            return Err(MathError::new(
                MathErrorKind::WrongTensorRank,
                format!("not enough element to create a tensor - {cv_cr_prod} elements required",),
            ));
        }

        Ok(Tensor { inner, cov, contr })
    }

    /// Description
    ///
    /// ## Panic
    /// Panics if `cvr + crr != R`.
    pub fn new(ranks: (Indexes<CV>, Indexes<CR>), inner: [T; R]) -> Self {
        match Self::try_new(ranks, inner) {
            Ok(t) => t,
            Err(err) => panic!("{err:?}"),
        }
    }

    /// Returns tensor rank.
    pub fn rank(&self) -> usize {
        self.cov
            .iter()
            .chain(self.contr.iter())
            .filter(|r| r.len() > 1)
            .count()
    }

    /// Returns `Option` with a tensor element by indexes
    /// where `idx.0` are covariant indexes and `idx.1` are
    /// are contravariant indexes.
    pub fn el(&self, idx: ([usize; CV], [usize; CR])) -> Option<&T> {
        let mut current_slice = self.inner.as_slice();
        let mut iter = idx
            .0
            .iter()
            .chain(idx.1.iter())
            .zip(self.cov.iter().chain(self.contr.iter()));

        for (i, rn) in &mut iter {
            let dim = rn.len();
            let len = current_slice.len() / dim;
            let start = len * i;
            let end = len * (i + 1);
            if start >= current_slice.len() || end > current_slice.len() {
                return None;
            }
            current_slice = &current_slice[start..end];
        }

        current_slice.first()
    }

    fn indexes_prod<const U: usize>(indexes: &Indexes<U>) -> usize {
        indexes
            .iter()
            // filter out constant indexes (len 0)
            .filter(|i| i.len() > 1)
            .fold(1, |acc, range| acc * range.len())
    }
}

#[cfg(test)]
mod test {
    use crate::tensor::{const_idx, idx};

    use super::Tensor;

    #[test]
    fn test_try_new() {
        let test_tensor_res = Tensor::try_new(([idx(3)], []), [1.0, 1.0, 1.0]);
        assert!(test_tensor_res.is_ok());
    }

    #[test]
    fn test_rank() {
        let test_tensor = Tensor::try_new(([idx(3)], []), [1.0, 1.0, 1.0]).unwrap();
        assert_eq!(test_tensor.rank(), 1);
        // fixed fixed index reduces the rank by one
        let test_tensor = Tensor::try_new(([idx(3)], [const_idx(2)]), [1.0, 1.0, 1.0]).unwrap();
        assert_eq!(test_tensor.rank(), 1);
    }

    #[test]
    fn test_el() {
        let test_tensor = Tensor::new(([idx(4)], []), [1, 2, 3, 4]);
        assert_eq!(test_tensor.el(([2], [])), Some(&3));

        let test_tensor = Tensor::new(
            ([idx(2), idx(2)], [idx(2), idx(2)]),
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        );
        assert_eq!(test_tensor.el(([1, 1], [2, 5])), None);
        assert_eq!(test_tensor.el(([0, 0], [0, 0])), Some(&1));
        assert_eq!(test_tensor.el(([1, 1], [1, 1])), Some(&16));
        assert_eq!(test_tensor.el(([0, 1], [1, 0])), Some(&7));
    }
}
