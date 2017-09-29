//! Memory layout of matrices

use lapack::c;
use ndarray::*;

use super::error::*;

pub type LDA = i32;
pub type LEN = i32;
pub type Col = i32;
pub type Row = i32;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MatrixLayout {
    C((Row, LDA)),
    F((Col, LDA)),
}

impl MatrixLayout {
    pub fn size(&self) -> (Row, Col) {
        match *self {
            MatrixLayout::C((row, lda)) => (row, lda),
            MatrixLayout::F((col, lda)) => (lda, col),
        }
    }

    pub fn resized(&self, row: Row, col: Col) -> MatrixLayout {
        match *self {
            MatrixLayout::C(_) => MatrixLayout::C((row, col)),
            MatrixLayout::F(_) => MatrixLayout::F((col, row)),
        }
    }

    pub fn lda(&self) -> LDA {
        match *self {
            MatrixLayout::C((_, lda)) => lda,
            MatrixLayout::F((_, lda)) => lda,
        }
    }

    pub fn len(&self) -> LEN {
        match *self {
            MatrixLayout::C((row, _)) => row,
            MatrixLayout::F((col, _)) => col,
        }
    }

    pub fn lapacke_layout(&self) -> c::Layout {
        match *self {
            MatrixLayout::C(_) => c::Layout::RowMajor,
            MatrixLayout::F(_) => c::Layout::ColumnMajor,
        }
    }

    pub fn same_order(&self, other: &MatrixLayout) -> bool {
        self.lapacke_layout() == other.lapacke_layout()
    }

    pub fn as_shape(&self) -> Shape<Ix2> {
        match *self {
            MatrixLayout::C((row, col)) => (row as usize, col as usize).into_shape(),
            MatrixLayout::F((col, row)) => (row as usize, col as usize).f().into_shape(),
        }
    }

    pub fn toggle_order(&self) -> Self {
        match *self {
            MatrixLayout::C((row, col)) => MatrixLayout::F((col, row)),
            MatrixLayout::F((col, row)) => MatrixLayout::C((row, col)),
        }
    }
}

pub trait AllocatedArray {
    type Elem;
    fn layout(&self) -> Result<MatrixLayout>;
    fn square_layout(&self) -> Result<MatrixLayout>;
    /// Returns Ok iff the matrix is square (without computing the layout).
    fn ensure_square(&self) -> Result<()>;
    fn as_allocated(&self) -> Result<&[Self::Elem]>;
}

pub trait AllocatedArrayMut: AllocatedArray {
    fn as_allocated_mut(&mut self) -> Result<&mut [Self::Elem]>;
}

impl<A, S> AllocatedArray for ArrayBase<S, Ix2>
where
    S: Data<Elem = A>,
{
    type Elem = A;

    fn layout(&self) -> Result<MatrixLayout> {
        let shape = self.shape();
        let strides = self.strides();
        if shape[0] == strides[1] as usize {
            return Ok(MatrixLayout::F((self.cols() as i32, self.rows() as i32)));
        }
        if shape[1] == strides[0] as usize {
            return Ok(MatrixLayout::C((self.rows() as i32, self.cols() as i32)));
        }
        Err(StrideError::new(strides[0], strides[1]).into())
    }

    fn square_layout(&self) -> Result<MatrixLayout> {
        let l = self.layout()?;
        let (n, m) = l.size();
        if n == m {
            Ok(l)
        } else {
            Err(NotSquareError::new(n, m).into())
        }
    }

    fn ensure_square(&self) -> Result<()> {
        if self.is_square() {
            Ok(())
        } else {
            Err(NotSquareError::new(self.rows() as i32, self.cols() as i32).into())
        }
    }

    fn as_allocated(&self) -> Result<&[A]> {
        Ok(self.as_slice_memory_order().ok_or(MemoryContError::new())?)
    }
}

impl<A, S> AllocatedArrayMut for ArrayBase<S, Ix2>
where
    S: DataMut<Elem = A>,
{
    fn as_allocated_mut(&mut self) -> Result<&mut [A]> {
        Ok(self.as_slice_memory_order_mut().ok_or(
            MemoryContError::new(),
        )?)
    }
}

// trait LapackLayout {
//     fn is_lapack_layout(&self) -> bool;
//     fn can_inplace_lapack_layout(&self) -> bool;
//     fn copy_to_lapack_layout(&self) -> ??;
//     fn inplace_lapack_layout(&mut self);
// }

trait ToLapackClone<A, D: Dimension> {
    fn to_lapack_clone<So>(&self) -> ArrayBase<So, D> where So: DataOwned<Elem = A>;
}

impl<'a, A, S, D> ToLapackClone<A, D> for ArrayBase<S, D>
where
    A: Clone,
    S: Data<Elem = A>,
    D: Dimension,
{
    fn to_lapack_clone<So>(&self) -> ArrayBase<So, D> where So: DataOwned<Elem = A> {
        unimplemented!()
    }
}

enum LapackView<T> {
    NoChange,
    New(T),
}

trait ToLapackView: Sized {
    fn to_lapack_view(&self) -> LapackView<Self>;
}

impl<S, D> ToLapackView for ArrayBase<S, D>
where
    S: Data,
    D: Dimension,
{
    fn to_lapack_view(&self) -> LapackView<Self> {
        unimplemented!()
    }
}

enum LapackInplace<T> {
    Inplace,
    New(T),
}

trait ToLapackInplace: Sized {
    fn to_lapack_inplace(&mut self) -> LapackInplace<Self>;
}

impl<S, D> ToLapackInplace for ArrayBase<S, D>
where
    S: DataMut,
    D: Dimension,
{
    fn to_lapack_inplace(&mut self) -> LapackInplace<Self> {
        unimplemented!()
    }
}

pub(crate) fn with_lapack_into<A, S, D, F, O>(mut a: ArrayBase<S, D>, body: F) -> (ArrayBase<S, D>, O)
where
    A: Clone,
    S: DataMut<Elem = A> + DataOwned,
    D: Dimension,
    F: FnOnce(&mut ArrayBase<S, D>) -> O,
{
    let mut a_lapack = match a.to_lapack_inplace() {
        LapackInplace::Inplace => a,
        LapackInplace::New(new) => new,
    };
    let out = body(&mut a_lapack);
    (a_lapack, out)
}

pub(crate) fn with_lapack_inplace<A, S, D, F, O>(a: &mut ArrayBase<S, D>, body: F) -> (&mut ArrayBase<S, D>, O)
where
    A: Clone,
    S: DataMut<Elem = A>,
    D: Dimension,
    F: FnOnce(&mut ArrayBase<S, D>) -> O,
{
    let out = match a.to_lapack_inplace() {
        LapackInplace::Inplace => body(a),
        LapackInplace::New(mut new) => {
            let out = body(&mut new);
            a.assign(&new);
            out
        }
    };
    (a, out)
}

pub(crate) fn with_lapack_clone<A, Sa, So, D, F, O>(a: &ArrayBase<Sa, D>, body: F) -> (ArrayBase<So, D>, O)
where
    A: Clone,
    Sa: Data<Elem = A>,
    So: DataOwned<Elem = A>,
    D: Dimension,
    F: FnOnce(&mut ArrayBase<So, D>) -> O,
{
    let mut a_lapack = a.to_lapack_clone();
    let out = body(&mut a_lapack);
    (a_lapack, out)
}

pub(crate) fn with_lapack_readonly<A, S, D, F, O>(a: &ArrayBase<S, D>, body: F) -> O
where
    A: Clone,
    S: Data<Elem = A>,
    D: Dimension,
    F: FnOnce(&ArrayBase<S, D>) -> O,
{
    match a.to_lapack_view() {
        LapackView::NoChange => body(a),
        LapackView::New(new) => body(&new),
    }
}

use types::*;

fn solve_into_1<A, Sa, Sb>(a: &ArrayBase<Sa, Ix2>, b: ArrayBase<Sb, Ix1>) -> ArrayBase<Sb, Ix1>
where
    A: Scalar,
    Sa: Data<Elem = A>,
    Sb: DataOwned + DataMut<Elem = A>
{
    let (x, info) = with_lapack_into(b, |b: &mut ArrayBase<Sb, Ix1>| {
        with_lapack_readonly(a, |a: &ArrayBase<Sa, Ix2>) {
            unimplemented!()
        }
    });
    x
}

fn solve_into_2<A, Sa, Sb>(a: &ArrayBase<Sa, Ix2>, b: ArrayBase<Sb, Ix2>) -> ArrayBase<Sb, Ix2>
where
    A: Scalar,
    Sa: Data<Elem = A>,
    Sb: DataOwned + DataMut<Elem = A>
{
    let (x, info) = with_lapack_into(b, |b: &mut ArrayBase<Sb, Ix2>| {
        with_lapack_readonly(a, |a: &ArrayBase<Sa, Ix2>) {
            unimplemented!()
        }
    });
    x
}

fn solve_1<A, Sa, Sb, So>(a: &ArrayBase<Sa, Ix2>, b: &ArrayBase<Sb, Ix1>) -> ArrayBase<So, Ix1>
where
    A: Scalar,
    Sa: Data<Elem = A>,
    Sb: Data<Elem = A>,
    So: DataOwned<Elem = A>,
{
    let (x, info) = with_lapack_clone(b, |b: &mut ArrayBase<So, Ix1>| {
        with_lapack_readonly(a, |a: &ArrayBase<Sa, Ix2>) {
            unimplemented!()
        }
    });
    x
}

fn solve_inplace_1<'a, A, Sa, Sb>(a: &ArrayBase<Sa, Ix2>, b: &'a mut ArrayBase<Sb, Ix1>) -> &'a mut ArrayBase<Sb, Ix1>
where
    A: Scalar,
    Sa: Data<Elem = A>,
    Sb: DataMut<Elem = A>,
{
    let (x, info) = with_lapack_inplace(b, |b: &mut ArrayBase<Sb, Ix1>| {
        with_lapack_readonly(a, |a: &ArrayBase<Sa, Ix2>) {
            unimplemented!()
        }
    });
    x
}

fn inverse<A, Sa, So>(a: &ArrayBase<Sa, Ix2>) -> ArrayBase<So, Ix2>
where
    A: Scalar,
    Sa: Data<Elem = A>,
    So: DataOwned<Elem = A>
{
    let (inv, info) = with_lapack_clone(a, |a: &mut ArrayBase<So, Ix2>| {
        unimplemented!()
    });
    inv
}

fn inverse_inplace<A, S>(a: &mut ArrayBase<S, Ix2>) -> &mut ArrayBase<S, Ix2>
where
    A: Scalar,
    S: DataMut<Elem = A>,
{
    let (inv, info) = with_lapack_inplace(a, |a: &mut ArrayBase<S, Ix2>| {
        unimplemented!()
    });
    inv
}

fn inverse_into<A, S>(a: ArrayBase<S, Ix2>) -> ArrayBase<S, Ix2>
where
    A: Scalar,
    S: DataMut<Elem = A> + DataOwned,
{
    let (inv, info) = with_lapack_into(a, |a: &mut ArrayBase<S, Ix2>| {
        unimplemented!()
    });
    inv
}

// TODO: NaN checking?
