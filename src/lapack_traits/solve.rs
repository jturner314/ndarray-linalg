//! Solve linear problem using LU decomposition

use lapack::c;

use error::*;
use layout::{LapackInput, LapackInputOutput};
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
    unsafe fn lu(a: &mut LapackInputOutput<Self>) -> Result<Pivot>;
    unsafe fn inv(a: &mut LapackInputOutput<Self>, &Pivot) -> Result<()>;
    unsafe fn solve(Transpose, a: &LapackInput<Self>, &Pivot, b: &mut LapackInputOutput<Self>) -> Result<()>;
}

macro_rules! impl_solve {
    ($scalar:ty, $getrf:path, $getri:path, $getrs:path) => {

impl Solve_ for $scalar {
    unsafe fn lu(a: &mut LapackInputOutput<Self>) -> Result<Pivot>
    {
        let LapackInputOutput {
            rows: m,
            cols: n,
            column_stride: lda,
            data_slice_mut: ref mut a_slice_mut,
        } = *a;
        let mut ipiv = vec![0; ::std::cmp::min(m, n) as usize];
        let info = $getrf(c::Layout::ColumnMajor, m, n, a_slice_mut, lda, &mut ipiv);
        into_result(info, ipiv)
    }

    unsafe fn inv(a: &mut LapackInputOutput<Self>, ipiv: &Pivot) -> Result<()> {
        a.ensure_square()?;
        let LapackInputOutput {
            rows: n,
            column_stride: lda,
            data_slice_mut: ref mut a_slice_mut,
            ..
        } = *a;
        let info = $getri(c::Layout::ColumnMajor, n, a_slice_mut, lda, ipiv);
        into_result(info, ())
    }

    unsafe fn solve(trans: Transpose, a: &LapackInput<Self>, ipiv: &Pivot, b: &mut LapackInputOutput<Self>) -> Result<()>
    {
        a.ensure_square()?;
        // TODO: return Result here
        assert_eq!(a.rows, b.rows);
        let LapackInput {
            rows: n,
            column_stride: lda,
            data_slice: a_slice,
            ..
        } = *a;
        let LapackInputOutput {
            cols: nrhs,
            column_stride: ldb,
            data_slice_mut: ref mut b_slice_mut,
            ..
        } = *b;
        let info = $getrs(c::Layout::ColumnMajor, trans as u8, n, nrhs, a_slice, lda, ipiv, b_slice_mut, ldb);
        into_result(info, ())
    }
}

}} // impl_solve!

impl_solve!(f64, c::dgetrf, c::dgetri, c::dgetrs);
impl_solve!(f32, c::sgetrf, c::sgetri, c::sgetrs);
impl_solve!(c64, c::zgetrf, c::zgetri, c::zgetrs);
impl_solve!(c32, c::cgetrf, c::cgetri, c::cgetrs);
