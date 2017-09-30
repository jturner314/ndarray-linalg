//! Solve linear problem using LU decomposition

use lapack::{c, fortran};
use ndarray::prelude::*;

use error::*;
use layout::{LapackArrayViewMut, MatrixLayout};
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
    unsafe fn lu(a: LapackArrayViewMut<Self, Ix2>) -> Result<Pivot>;
    unsafe fn inv(MatrixLayout, a: &mut [Self], &Pivot) -> Result<()>;
    unsafe fn solve(MatrixLayout, Transpose, a: &[Self], &Pivot, b: &mut [Self]) -> Result<()>;
}

macro_rules! impl_solve {
    ($scalar:ty, $getrf:path, $getri:path, $getrs:path) => {

impl Solve_ for $scalar {
    unsafe fn lu(mut a: LapackArrayViewMut<Self, Ix2>) -> Result<Pivot> {
        let m = a.rows() as i32;
        let n = a.cols() as i32;
        let lda = a.column_stride() as i32;
        let mut ipiv = vec![0; ::std::cmp::min(m, n) as usize];
        let mut info = 0;
        $getrf(m, n, a.as_data_slice_mut(), lda, &mut ipiv, &mut info);
        into_result(info, ipiv)
    }

    unsafe fn inv(l: MatrixLayout, a: &mut [Self], ipiv: &Pivot) -> Result<()> {
        let (n, _) = l.size();
        let info = $getri(l.lapacke_layout(), n, a, l.lda(), ipiv);
        into_result(info, ())
    }

    unsafe fn solve(l: MatrixLayout, t: Transpose, a: &[Self], ipiv: &Pivot, b: &mut [Self]) -> Result<()> {
        let (n, _) = l.size();
        let nrhs = 1;
        let ldb = 1;
        let info = $getrs(l.lapacke_layout(), t as u8, n, nrhs, a, l.lda(), ipiv, b, ldb);
        into_result(info, ())
    }
}

}} // impl_solve!

impl_solve!(f64, fortran::dgetrf, c::dgetri, c::dgetrs);
impl_solve!(f32, fortran::sgetrf, c::sgetri, c::sgetrs);
impl_solve!(c64, fortran::zgetrf, c::zgetri, c::zgetrs);
impl_solve!(c32, fortran::cgetrf, c::cgetri, c::cgetrs);
