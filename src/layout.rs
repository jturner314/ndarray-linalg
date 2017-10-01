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

pub struct LapackInput<'a, A: 'a> {
    pub rows: i32,
    pub cols: i32,
    pub column_stride: i32,
    /// A slice to the data that is long enough to access all the elements of
    /// the array.
    pub data_slice: &'a [A],
}

pub struct LapackInputOutput<'a, A: 'a> {
    pub rows: i32,
    pub cols: i32,
    pub column_stride: i32,
    /// A slice to the data that is long enough to access all the elements of
    /// the array.
    pub data_slice_mut: &'a mut [A],
}

impl<'a, A: 'a> LapackInput<'a, A> {
    pub fn ensure_square(&self) -> Result<()> {
        if self.rows == self.cols {
            Ok(())
        } else {
            Err(NotSquareError::new(self.rows, self.cols).into())
        }
    }
}

impl<'a, A: 'a> LapackInputOutput<'a, A> {
    pub fn ensure_square(&self) -> Result<()> {
        if self.rows == self.cols {
            Ok(())
        } else {
            Err(NotSquareError::new(self.rows, self.cols).into())
        }
    }
}

trait FromLapackArray<A, D> {
    /// Given an argument that's in Lapack layout, performs the conversion.
    ///
    /// **Warning: This method does not check if the array is actually in
    /// Lapack layout.**
    fn from_lapack_array<S>(array: &ArrayBase<S, D>) -> Self
    where S: Data<Elem = A>;
}

trait FromLapackArrayMut<A, D> {
    /// Given an argument that's in Lapack layout, performs the conversion.
    ///
    /// **Warning: This method does not check if the array is actually in
    /// Lapack layout.**
    fn from_lapack_array_mut<S>(array: &mut ArrayBase<S, D>) -> Self
    where S: DataMut<Elem = A>;
}

impl<'a, A> FromLapackArray<A, Ix1> for LapackInput<'a, A> {
    fn from_lapack_array<S>(array: &ArrayBase<S, Ix1>) -> LapackInput<'a, A>
    where
        S: Data<Elem = A>,
    {
        debug_assert_eq!(array.check_lapack_layout(), LapackLayoutStatus::IsLapack);
        LapackInput {
            rows: array.len() as i32,
            cols: 1,
            column_stride: ::std::cmp::max(1, array.len() as i32),
            data_slice: unsafe { ::std::slice::from_raw_parts(array.as_ptr(), array.len()) },
        }
    }
}

impl<'a, A> FromLapackArrayMut<A, Ix1> for LapackInputOutput<'a, A> {
    fn from_lapack_array_mut<S>(array: &mut ArrayBase<S, Ix1>) -> LapackInputOutput<'a, A>
    where
        S: DataMut<Elem = A>,
    {
        debug_assert_eq!(array.check_lapack_layout(), LapackLayoutStatus::IsLapack);
        LapackInputOutput {
            rows: array.len() as i32,
            cols: 1,
            column_stride: ::std::cmp::max(1, array.len() as i32),
            data_slice_mut: unsafe { ::std::slice::from_raw_parts_mut(array.as_mut_ptr(), array.len()) },
        }
    }
}

impl<'a, A> FromLapackArray<A, Ix2> for LapackInput<'a, A> {
    fn from_lapack_array<S>(array: &ArrayBase<S, Ix2>) -> LapackInput<'a, A>
    where
        S: Data<Elem = A>,
    {
        debug_assert_eq!(array.check_lapack_layout(), LapackLayoutStatus::IsLapack);
        let column_stride = ::std::cmp::max(
            1,
            ::std::cmp::max(array.rows() as i32, array.strides()[1] as i32),
        );
        let min_data_len = if array.cols() == 0 {
            0
        } else {
            (array.cols() - 1) * (column_stride as usize) + array.rows()
        };
        LapackInput {
            rows: array.rows() as i32,
            cols: array.cols() as i32,
            column_stride,
            data_slice: unsafe { ::std::slice::from_raw_parts(array.as_ptr(), min_data_len) },
        }
    }
}

impl<'a, A> FromLapackArrayMut<A, Ix2> for LapackInputOutput<'a, A> {
    fn from_lapack_array_mut<S>(array: &mut ArrayBase<S, Ix2>) -> LapackInputOutput<'a, A>
    where
        S: DataMut<Elem = A>,
    {
        debug_assert_eq!(array.check_lapack_layout(), LapackLayoutStatus::IsLapack);
        let column_stride = ::std::cmp::max(
            1,
            ::std::cmp::max(array.rows() as i32, array.strides()[1] as i32),
        );
        let min_data_len = if array.cols() == 0 {
            0
        } else {
            (array.cols() - 1) * (column_stride as usize) + array.rows()
        };
        LapackInputOutput {
            rows: array.rows() as i32,
            cols: array.cols() as i32,
            column_stride,
            data_slice_mut: unsafe { ::std::slice::from_raw_parts_mut(array.as_mut_ptr(), min_data_len) },
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
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

trait CheckLapackLayout {
    fn check_lapack_layout(&self) -> LapackLayoutStatus;
}

impl<S: Data> CheckLapackLayout for ArrayBase<S, Ix1> {
    fn check_lapack_layout(&self) -> LapackLayoutStatus {
        if self.strides() == &[1] || self.len() == 0 || self.len() == 1 {
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

pub trait IntoLapack {
    fn into_lapack(self) -> Self;
}

impl<A, S> IntoLapack for ArrayBase<S, Ix1>
where
    A: Clone,
    S: DataOwned<Elem = A>,
{
    fn into_lapack(self) -> Self {
        match self.check_lapack_layout() {
            LapackLayoutStatus::IsLapack => self,
            _ => self.to_lapack_clone(),
        }
    }
}

impl<A, S> IntoLapack for ArrayBase<S, Ix2>
where
    A: Clone,
    S: DataOwned<Elem = A> + DataMut,
{
    fn into_lapack(mut self) -> Self {
        match self.check_lapack_layout() {
            LapackLayoutStatus::IsLapack => self,
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
                self.reversed_axes()
            }
            _ => self.to_lapack_clone(),
        }
    }
}

pub trait ToLapackClone<A: Clone, D: Dimension> {
    fn to_lapack_clone<So: DataOwned<Elem = A>>(&self) -> ArrayBase<So, D>;
}

impl<'a, A, Si> ToLapackClone<A, Ix1> for ArrayBase<Si, Ix1>
where
    A: Clone,
    Si: Data<Elem = A>,
{
    fn to_lapack_clone<So: DataOwned<Elem = A>>(&self) -> ArrayBase<So, Ix1> {
        ArrayBase::from_iter(self.into_iter().cloned())
    }
}

impl<'a, A, Si> ToLapackClone<A, Ix2> for ArrayBase<Si, Ix2>
where
    A: Clone,
    Si: Data<Elem = A>,
{
    fn to_lapack_clone<So: DataOwned<Elem = A>>(&self) -> ArrayBase<So, Ix2> {
        ArrayBase::from_shape_fn(self.dim().f(), |index| unsafe { self.uget(index) }.clone())
    }
}

pub trait WithLapackInputOutput<A> {
    fn with_lapack_inout<F, O>(&mut self, body: F) -> O
    where F: FnOnce(&mut LapackInputOutput<A>) -> O;
}

macro_rules! impl_with_lapack_inout {
    ($D:ty) => {
        impl<A, S> WithLapackInputOutput<A> for ArrayBase<S, $D>
        where
            A: Clone,
            S: DataMut<Elem = A>,
        {
            fn with_lapack_inout<F, O>(&mut self, body: F) -> O
            where
                F: FnOnce(&mut LapackInputOutput<A>) -> O,
            {
                if let LapackLayoutStatus::IsLapack = self.check_lapack_layout() {
                    body(&mut LapackInputOutput::from_lapack_array_mut(self))
                } else {
                    let mut new = self.to_lapack_clone::<OwnedRepr<A>>();
                    let out = body(&mut LapackInputOutput::from_lapack_array_mut(&mut new));
                    self.assign(&new);
                    out
                }
            }
        }
    }
}

impl_with_lapack_inout!(Ix1);
impl_with_lapack_inout!(Ix2);

pub trait WithLapackInput<A> {
    fn with_lapack_in<F, O>(&self, body: F) -> O
    where F: FnOnce(&LapackInput<A>) -> O;
}

macro_rules! impl_with_lapack_in {
    ($D:ty) => {
        impl<A, S> WithLapackInput<A> for ArrayBase<S, $D>
        where
            A: Clone,
            S: Data<Elem = A>,
        {
            fn with_lapack_in<F, O>(&self, body: F) -> O
            where
                F: FnOnce(&LapackInput<A>) -> O,
            {
                if let LapackLayoutStatus::IsLapack = self.check_lapack_layout() {
                    body(&LapackInput::from_lapack_array(self))
                } else {
                    let new = self.to_lapack_clone::<OwnedRepr<A>>();
                    body(&LapackInput::from_lapack_array(&new))
                }
            }
        }
    }
}

impl_with_lapack_in!(Ix1);
impl_with_lapack_in!(Ix2);
