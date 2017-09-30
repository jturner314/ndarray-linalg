//! Solve linear problem using LU decomposition

use lapack::{c, fortran};

use error::*;
use layout::{LapackViewProps, LapackViewPropsMut, MatrixLayout};
use types::*;

use super::{into_result, Pivot, Transpose};

/// Wraps `*getrf`, `*getri`, and `*getrs`
pub trait Solve_: Sized {
    /// Computes the LU factorization of a general `m x n` matrix `a` using
    /// partial pivoting with row interchanges.
    ///
    /// If the result matches `Err(LinalgError::Lapack(LapackError {
    /// return_code )) if return_code > 0`, then `U[(return_code-1,
    /// return_code-1)]` is exactly zero. The factorization has been completed,
    /// but the factor `U` is exactly singular, and division by zero will occur
    /// if it is used to solve a system of equations.
    unsafe fn lu<T>(a: &mut T) -> Result<Pivot>
    where T: LapackViewPropsMut<Self>;
    unsafe fn inv(MatrixLayout, a: &mut [Self], &Pivot) -> Result<()>;
    unsafe fn solve<T1, T2>(Transpose, a: &T1, &Pivot, b: &mut T2) -> Result<()>
    where
        T1: LapackViewProps<Self>,
        T2: LapackViewPropsMut<Self>;
}

macro_rules! impl_solve {
    ($scalar:ty, $getrf:path, $getri:path, $getrs:path) => {

impl Solve_ for $scalar {
    unsafe fn lu<T>(a: &mut T) -> Result<Pivot>
    where
        T: LapackViewPropsMut<Self>
    {
        let m = a.rows() as i32;
        let n = a.cols() as i32;
        let lda = a.column_stride() as i32;
        let mut ipiv = vec![0; ::std::cmp::min(m, n) as usize];
        let info = $getrf(c::Layout::ColumnMajor, m, n, a.as_data_slice_mut(), lda, &mut ipiv);
        into_result(info, ipiv)
    }

    unsafe fn inv(l: MatrixLayout, a: &mut [Self], ipiv: &Pivot) -> Result<()> {
        let (n, _) = l.size();
        let info = $getri(l.lapacke_layout(), n, a, l.lda(), ipiv);
        into_result(info, ())
    }

    unsafe fn solve<T1, T2>(trans: Transpose, a: &T1, ipiv: &Pivot, b: &mut T2) -> Result<()>
    where
        T1: LapackViewProps<Self>,
        T2: LapackViewPropsMut<Self>
    {
        a.ensure_square()?;
        let n = a.rows() as i32;
        let nrhs = b.cols() as i32;
        let lda = a.column_stride() as i32;
        let ldb = b.column_stride() as i32;
        let info = $getrs(
            c::Layout::ColumnMajor,
            trans as u8,
            n,
            nrhs,
            a.as_data_slice(),
            lda,
            ipiv,
            b.as_data_slice_mut(),
            ldb);
        into_result(info, ())
    }
}

}} // impl_solve!

impl_solve!(f64, c::dgetrf, c::dgetri, c::dgetrs);
impl_solve!(f32, c::sgetrf, c::sgetri, c::sgetrs);
impl_solve!(c64, c::zgetrf, c::zgetri, c::zgetrs);
impl_solve!(c32, c::cgetrf, c::cgetri, c::cgetrs);
