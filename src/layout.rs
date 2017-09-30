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
        Ok(self.as_slice_memory_order_mut()
            .ok_or(MemoryContError::new())?)
    }
}

// TODO: handle triangular arrays more efficiently
// TODO: add NaN checking

/// An owned array with a memory representation that has contiguous columns.
pub struct LapackArrayOwned<S: DataOwned, D: Dimension>(ArrayBase<S, D>);

impl<A, S: DataOwned<Elem = A>, D: Dimension> LapackArrayOwned<S, D> {
    pub fn into_array(self) -> ArrayBase<S, D> {
        self.0
    }

    pub fn view(&self) -> LapackArrayView<A, D> {
        LapackArrayView(self.0.view())
    }
}

impl<A, S: DataOwned<Elem = A> + DataMut, D: Dimension> LapackArrayOwned<S, D> {
    pub fn view_mut(&mut self) -> LapackArrayViewMut<A, D> {
        LapackArrayViewMut(self.0.view_mut())
    }
}

impl<S, D> Into<ArrayBase<S, D>> for LapackArrayOwned<S, D>
where
    S: DataOwned,
    D: Dimension,
{
    fn into(self) -> ArrayBase<S, D> {
        self.0
    }
}

/// A mutable array view with a memory representation that has contiguous columns.
pub struct LapackArrayViewMut<'a, A: 'a, D>(ArrayViewMut<'a, A, D>);

impl<'a, A: 'a, D: Dimension> LapackArrayViewMut<'a, A, D> {
    pub fn view(&self) -> LapackArrayView<A, D> {
        LapackArrayView(self.0.view())
    }
}

/// An array view with a memory representation that has contiguous columns.
pub struct LapackArrayView<'a, A: 'a, D>(ArrayView<'a, A, D>);

impl<'a, A> LapackArrayView<'a, A, Ix1> {
    pub fn rows(&self) -> usize {
        self.0.len()
    }

    pub fn cols(&self) -> usize {
        1
    }

    pub fn column_stride(&self) -> isize {
        ::std::cmp::max(1, self.rows() as isize)
    }

    /// Returns a slice to the data that has the minimum length needed to
    /// access all the elements of the array.
    pub fn as_data_slice(&self) -> &[A] {
        unsafe { ::std::slice::from_raw_parts(self.0.as_ptr(), self.rows()) }
    }
}

impl<'a, A> LapackArrayView<'a, A, Ix2> {
    pub fn rows(&self) -> usize {
        self.0.rows()
    }

    pub fn cols(&self) -> usize {
        self.0.cols()
    }

    pub fn column_stride(&self) -> isize {
        ::std::cmp::max(1, self.0.strides()[1])
    }

    /// Returns a slice to the data that has the minimum length needed to
    /// access all the elements of the array.
    pub fn as_data_slice(&self) -> &[A] {
        let len = (self.cols() - 1) * (self.column_stride() as usize) + self.rows();
        unsafe { ::std::slice::from_raw_parts(self.0.as_ptr(), len) }
    }
}

impl<'a, A> LapackArrayViewMut<'a, A, Ix1> {
    pub fn rows(&self) -> usize {
        self.0.len()
    }

    pub fn cols(&self) -> usize {
        1
    }

    pub fn column_stride(&self) -> isize {
        ::std::cmp::max(1, self.rows() as isize)
    }

    /// Returns a slice to the data that has the minimum length needed to
    /// access all the elements of the array.
    pub fn as_data_slice_mut(&mut self) -> &mut [A] {
        unsafe { ::std::slice::from_raw_parts_mut(self.0.as_mut_ptr(), self.rows()) }
    }
}

impl<'a, A> LapackArrayViewMut<'a, A, Ix2> {
    pub fn rows(&self) -> usize {
        self.0.rows()
    }

    pub fn cols(&self) -> usize {
        self.0.cols()
    }

    pub fn column_stride(&self) -> isize {
        ::std::cmp::max(1, self.0.strides()[1])
    }

    /// Returns a slice to the data that has the minimum length needed to
    /// access all the elements of the array.
    pub fn as_data_slice_mut(&mut self) -> &mut [A] {
        let len = (self.cols() - 1) * (self.column_stride() as usize) + self.rows();
        unsafe { ::std::slice::from_raw_parts_mut(self.0.as_mut_ptr(), len) }
    }
}

pub enum LapackLayoutStatus {
    /// The array is aready in Lapack layout.
    IsLapack,
    /// The array is square and it's possible to transpose elements in-place to achieve Lapack layout.
    CanTransposeSquare,
    /// It's possible to transpose elements in-place to achieve Lapack layout.
    CanTransposeRect,
    /// The contents must be cloned to a new array to achieve Lapack layout.
    MustClone,
}

pub trait CheckLapackLayout {
    fn check_lapack_layout(&self) -> LapackLayoutStatus;
}

impl<S: Data> CheckLapackLayout for ArrayBase<S, Ix1> {
    fn check_lapack_layout(&self) -> LapackLayoutStatus {
        if self.strides() == &[1] {
            LapackLayoutStatus::IsLapack
        } else {
            LapackLayoutStatus::MustClone
        }
    }
}

impl<S: Data> CheckLapackLayout for ArrayBase<S, Ix2> {
    fn check_lapack_layout(&self) -> LapackLayoutStatus {
        let shape = self.shape();
        let strides = self.strides();
        if (strides[0] == 1 || shape[0] == 0 || shape[0] == 1)
            && (strides[1] >= shape[0] as isize || shape[1] == 0 || shape[1] == 1)
        {
            LapackLayoutStatus::IsLapack
        } else if shape[0] == shape[1] && strides[0] >= shape[1] as isize && strides[1] == 1 {
            LapackLayoutStatus::CanTransposeSquare
        } else if strides[0] == shape[1] as isize && strides[1] == 1 {
            LapackLayoutStatus::CanTransposeRect
        } else {
            LapackLayoutStatus::MustClone
        }
    }
}

pub trait IntoLapack<S: DataOwned, D: Dimension> {
    fn into_lapack(self) -> LapackArrayOwned<S, D>;
}

impl<A, S> IntoLapack<S, Ix1> for ArrayBase<S, Ix1>
where
    A: Clone,
    S: DataOwned<Elem = A>,
{
    fn into_lapack(self) -> LapackArrayOwned<S, Ix1> {
        match self.check_lapack_layout() {
            LapackLayoutStatus::IsLapack => LapackArrayOwned(self),
            _ => self.to_lapack_clone(),
        }
    }
}

impl<A, S> IntoLapack<S, Ix2> for ArrayBase<S, Ix2>
where
    A: Clone,
    S: DataOwned<Elem = A> + DataMut,
{
    fn into_lapack(mut self) -> LapackArrayOwned<S, Ix2> {
        match self.check_lapack_layout() {
            LapackLayoutStatus::IsLapack => LapackArrayOwned(self),
            LapackLayoutStatus::CanTransposeSquare => {
                for i in 0..self.rows() {
                    for j in 0..i {
                        unsafe {
                            let elem1: *mut _ = self.uget_mut((i, j));
                            let elem2: *mut _ = self.uget_mut((j, i));
                            ::std::ptr::swap(elem1, elem2);
                        }
                    }
                }
                LapackArrayOwned(self.reversed_axes())
            }
            _ => self.to_lapack_clone(),
        }
    }
}

pub trait ToLapackClone<A: Clone, D: Dimension> {
    fn to_lapack_clone<So: DataOwned<Elem = A>>(&self) -> LapackArrayOwned<So, D>;
}

impl<'a, A, Si> ToLapackClone<A, Ix1> for ArrayBase<Si, Ix1>
where
    A: Clone,
    Si: Data<Elem = A>,
{
    fn to_lapack_clone<So: DataOwned<Elem = A>>(&self) -> LapackArrayOwned<So, Ix1> {
        LapackArrayOwned(ArrayBase::from_iter(self.into_iter().cloned()))
    }
}

impl<'a, A, Si> ToLapackClone<A, Ix2> for ArrayBase<Si, Ix2>
where
    A: Clone,
    Si: Data<Elem = A>,
{
    fn to_lapack_clone<So: DataOwned<Elem = A>>(&self) -> LapackArrayOwned<So, Ix2> {
        LapackArrayOwned(ArrayBase::from_shape_fn(
            self.dim().f(),
            |index| unsafe { self.uget(index) }.clone(),
        ))
    }
}

pub trait WithLapackViewMut<A, D> {
    fn with_lapack_view_mut<F, O>(&mut self, body: F) -> O
    where F: FnOnce(LapackArrayViewMut<A, D>) -> O;
}

impl<A, S, D> WithLapackViewMut<A, D> for ArrayBase<S, D>
where
    Self: CheckLapackLayout + ToLapackClone<A, D>,
    A: Clone,
    S: DataMut<Elem = A>,
    D: Dimension,
{
    fn with_lapack_view_mut<F, O>(&mut self, body: F) -> O
    where
        F: FnOnce(LapackArrayViewMut<A, D>) -> O,
    {
        if let LapackLayoutStatus::IsLapack = self.check_lapack_layout() {
            body(LapackArrayViewMut(self.view_mut()))
        } else {
            let mut new = self.to_lapack_clone::<OwnedRepr<A>>();
            let out = body(new.view_mut());
            self.assign(&new.into());
            out
        }
    }
}

pub trait WithLapackView<A, D> {
    fn with_lapack_view<F, O>(&self, body: F) -> O
    where F: FnOnce(LapackArrayView<A, D>) -> O;
}

impl<A, S, D> WithLapackView<A, D> for ArrayBase<S, D>
where
    Self: CheckLapackLayout + ToLapackClone<A, D>,
    A: Clone,
    S: Data<Elem = A>,
    D: Dimension,
{
    fn with_lapack_view<F, O>(&self, body: F) -> O
    where
        F: FnOnce(LapackArrayView<A, D>) -> O,
    {
        if let LapackLayoutStatus::IsLapack = self.check_lapack_layout() {
            body(LapackArrayView(self.view()))
        } else {
            let new = self.to_lapack_clone::<OwnedRepr<A>>();
            body(new.view())
        }
    }
}

use types::*;

fn solve_owned_1<A, Sa, Sb>(a: &ArrayBase<Sa, Ix2>, b: ArrayBase<Sb, Ix1>) -> ArrayBase<Sb, Ix1>
where
    A: Scalar,
    Sa: Data<Elem = A>,
    Sb: DataOwned + DataMut<Elem = A>,
{
    let x = b.into_lapack();
    let info = a.with_lapack_view(|a: LapackArrayView<A, Ix2>| unimplemented!());
    x.into()
}

fn solve_owned_2<A, Sa, Sb>(a: &ArrayBase<Sa, Ix2>, b: ArrayBase<Sb, Ix2>) -> ArrayBase<Sb, Ix2>
where
    A: Scalar,
    Sa: Data<Elem = A>,
    Sb: DataOwned + DataMut<Elem = A>,
{
    let x = b.into_lapack();
    let info = a.with_lapack_view(|a: LapackArrayView<A, Ix2>| unimplemented!());
    x.into()
}

fn solve_1<A, Sa, Sb, So>(a: &ArrayBase<Sa, Ix2>, b: &ArrayBase<Sb, Ix1>) -> ArrayBase<So, Ix1>
where
    A: Scalar,
    Sa: Data<Elem = A>,
    Sb: Data<Elem = A>,
    So: DataOwned<Elem = A>,
{
    let x = b.to_lapack_clone();
    let info = a.with_lapack_view(|a: LapackArrayView<A, Ix2>| unimplemented!());
    x.into()
}

fn solve_inplace_1<'a, A, Sa, Sb>(a: &ArrayBase<Sa, Ix2>, b: &'a mut ArrayBase<Sb, Ix1>) -> &'a mut ArrayBase<Sb, Ix1>
where
    A: Scalar,
    Sa: Data<Elem = A>,
    Sb: DataMut<Elem = A>,
{
    let info = a.with_lapack_view(|a: LapackArrayView<A, Ix2>| {
        b.with_lapack_view_mut(|b: LapackArrayViewMut<A, Ix1>| unimplemented!())
    });
    b
}

fn inverse<A, Sa, So>(a: &ArrayBase<Sa, Ix2>) -> ArrayBase<So, Ix2>
where
    A: Scalar,
    Sa: Data<Elem = A>,
    So: DataOwned<Elem = A>,
{
    let mut inv = a.to_lapack_clone();
    unimplemented!();
    inv.into()
}

fn inverse_inplace<A, S>(a: &mut ArrayBase<S, Ix2>) -> &mut ArrayBase<S, Ix2>
where
    A: Scalar,
    S: DataMut<Elem = A>,
{
    let info = a.with_lapack_view_mut(|a: LapackArrayViewMut<A, Ix2>| unimplemented!());
    a
}

fn inverse_owned<A, S>(a: ArrayBase<S, Ix2>) -> ArrayBase<S, Ix2>
where
    A: Scalar,
    S: DataOwned<Elem = A> + DataMut,
{
    let inv = a.into_lapack();
    unimplemented!();
    inv.into()
}
