//! Solve systems of linear equations and invert matrices
//!
//! # Examples
//!
//! Solve `A * x = b`:
//!
//! ```
//! #[macro_use]
//! extern crate ndarray;
//! extern crate ndarray_linalg;
//!
//! use ndarray::prelude::*;
//! use ndarray_linalg::Solve;
//! # fn main() {
//!
//! let a: Array2<f64> = array![[3., 2., -1.], [2., -2., 4.], [-2., 1., -2.]];
//! let b: Array1<f64> = array![1., -2., 0.];
//! let x = a.solve_into(b).unwrap();
//! assert!(x.all_close(&array![1., -2., -2.], 1e-9));
//!
//! # }
//! ```
//!
//! There are also special functions for solving `A^T * x = b` and
//! `A^H * x = b`.
//!
//! If you are solving multiple systems of linear equations with the same
//! coefficient matrix `A`, it's faster to compute the LU factorization once at
//! the beginning than solving directly using `A`:
//!
//! ```
//! # extern crate ndarray;
//! # extern crate ndarray_linalg;
//! use ndarray::prelude::*;
//! use ndarray_linalg::*;
//! # fn main() {
//!
//! let a: Array2<f64> = random((3, 3));
//! let f = a.factorize_into().unwrap(); // LU factorize A (A is consumed)
//! for _ in 0..10 {
//!     let b: Array1<f64> = random(3);
//!     let x = f.solve_into(b).unwrap(); // Solve A * x = b using factorized L, U
//! }
//!
//! # }
//! ```

use ndarray::*;

use super::convert::*;
use super::error::*;
use super::layout::*;
use super::types::*;

pub use lapack_traits::{Pivot, Transpose};

/// An interface for solving systems of linear equations.
///
/// There are three groups of methods:
///
/// * `solve*` (normal) methods solve `A * x = b` for `x`.
/// * `solve_t*` (transpose) methods solve `A^T * x = b` for `x`.
/// * `solve_h*` (Hermitian conjugate) methods solve `A^H * x = b` for `x`.
///
/// Within each group, there are three methods that handle ownership differently:
///
/// * `*` methods take a reference to `b` and return `x` as a new array.
/// * `*_into` methods take ownership of `b`, store the result in it, and return it.
/// * `*_inplace` methods take a mutable reference to `b` and store the result in that array.
///
/// If you plan to solve many equations with the same `A` matrix but different
/// `b` vectors, it's faster to factor the `A` matrix once using the
/// `Factorize` trait, and then solve using the `LUFactorized` struct.
pub trait Solve<A: Scalar, D: Dimension> {
    /// Solves a system of linear equations `A * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    fn solve<Sb, Sx>(&self, b: &ArrayBase<Sb, D>) -> Result<ArrayBase<Sx, D>>
    where
        Sb: Data<Elem = A>,
        Sx: DataOwned<Elem = A>,
    {
        let mut b = replicate(b);
        self.solve_inplace(&mut b)?;
        Ok(b)
    }
    /// Solves a system of linear equations `A * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    fn solve_into<Sb>(&self, mut b: ArrayBase<Sb, D>) -> Result<ArrayBase<Sb, D>>
    where
        Sb: DataMut<Elem = A>,
    {
        self.solve_inplace(&mut b)?;
        Ok(b)
    }
    /// Solves a system of linear equations `A * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    fn solve_inplace<'a, Sb>(&self, &'a mut ArrayBase<Sb, D>) -> Result<&'a mut ArrayBase<Sb, D>>
    where
        Sb: DataMut<Elem = A>;

    /// Solves a system of linear equations `A^T * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    fn solve_t<S: Data<Elem = A>>(&self, b: &ArrayBase<S, D>) -> Result<Array1<A>> {
        let mut b = replicate(b);
        self.solve_t_inplace(&mut b)?;
        Ok(b)
    }
    /// Solves a system of linear equations `A^T * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    fn solve_t_into<S: DataMut<Elem = A>>(&self, mut b: ArrayBase<S, D>) -> Result<ArrayBase<S, D>> {
        self.solve_t_inplace(&mut b)?;
        Ok(b)
    }
    /// Solves a system of linear equations `A^T * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    fn solve_t_inplace<'a, S: DataMut<Elem = A>>(&self, &'a mut ArrayBase<S, D>)
        -> Result<&'a mut ArrayBase<S, D>>;

    /// Solves a system of linear equations `A^H * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    fn solve_h<S: Data<Elem = A>>(&self, b: &ArrayBase<S, D>) -> Result<Array1<A>> {
        let mut b = replicate(b);
        self.solve_h_inplace(&mut b)?;
        Ok(b)
    }
    /// Solves a system of linear equations `A^H * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    fn solve_h_into<S: DataMut<Elem = A>>(&self, mut b: ArrayBase<S, D>) -> Result<ArrayBase<S, D>> {
        self.solve_h_inplace(&mut b)?;
        Ok(b)
    }
    /// Solves a system of linear equations `A^H * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    fn solve_h_inplace<'a, S: DataMut<Elem = A>>(&self, &'a mut ArrayBase<S, D>)
        -> Result<&'a mut ArrayBase<S, D>>;
}

/// Represents the LU factorization of a matrix `A` as `A = P*L*U`.
pub struct LUFactorized<S: Data> {
    /// The factors `L` and `U`; the unit diagonal elements of `L` are not
    /// stored.
    pub a: ArrayBase<S, Ix2>,
    /// The pivot indices that define the permutation matrix `P`.
    pub ipiv: Pivot,
}

impl<A, S> Solve<A, Ix1> for LUFactorized<S>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    fn solve_inplace<'a, Sb>(&self, mut rhs: &'a mut ArrayBase<Sb, Ix1>) -> Result<&'a mut ArrayBase<Sb, Ix1>>
    where
        Sb: DataMut<Elem = A>,
    {
        self.a.with_lapack_view(|a| {
            rhs.with_lapack_view_mut(|b| unsafe { A::solve(Transpose::No, a, &self.ipiv, b) })
        })?;
        Ok(rhs)
    }
    fn solve_t_inplace<'a, Sb>(&self, mut rhs: &'a mut ArrayBase<Sb, Ix1>) -> Result<&'a mut ArrayBase<Sb, Ix1>>
    where
        Sb: DataMut<Elem = A>,
    {
        self.a.with_lapack_view(|a| {
            rhs.with_lapack_view_mut(|b| unsafe { A::solve(Transpose::Transpose, a, &self.ipiv, b) })
        })?;
        Ok(rhs)
    }
    fn solve_h_inplace<'a, Sb>(&self, mut rhs: &'a mut ArrayBase<Sb, Ix1>) -> Result<&'a mut ArrayBase<Sb, Ix1>>
    where
        Sb: DataMut<Elem = A>,
    {
        self.a.with_lapack_view(|a| {
            rhs.with_lapack_view_mut(|b| unsafe { A::solve(Transpose::Hermite, a, &self.ipiv, b) })
        })?;
        Ok(rhs)
    }
}

impl<A, S> Solve<A, Ix1> for ArrayBase<S, Ix2>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    fn solve_inplace<'a, Sb>(&self, mut rhs: &'a mut ArrayBase<Sb, Ix1>) -> Result<&'a mut ArrayBase<Sb, Ix1>>
    where
        Sb: DataMut<Elem = A>,
    {
        let f = self.factorize()?;
        f.solve_inplace(rhs)
    }
    fn solve_t_inplace<'a, Sb>(&self, mut rhs: &'a mut ArrayBase<Sb, Ix1>) -> Result<&'a mut ArrayBase<Sb, Ix1>>
    where
        Sb: DataMut<Elem = A>,
    {
        let f = self.factorize()?;
        f.solve_t_inplace(rhs)
    }
    fn solve_h_inplace<'a, Sb>(&self, mut rhs: &'a mut ArrayBase<Sb, Ix1>) -> Result<&'a mut ArrayBase<Sb, Ix1>>
    where
        Sb: DataMut<Elem = A>,
    {
        let f = self.factorize()?;
        f.solve_h_inplace(rhs)
    }
}


/// An interface for computing LU factorizations of matrix refs.
pub trait Factorize<S: Data> {
    /// Computes the LU factorization `A = P*L*U`, where `P` is a permutation
    /// matrix.
    fn factorize(&self) -> Result<LUFactorized<S>>;
}

/// An interface for computing LU factorizations of matrices.
pub trait FactorizeInto<S: Data> {
    /// Computes the LU factorization `A = P*L*U`, where `P` is a permutation
    /// matrix.
    fn factorize_into(self) -> Result<LUFactorized<S>>;
}

impl<A, S> FactorizeInto<S> for ArrayBase<S, Ix2>
where
    A: Scalar,
    S: DataOwned<Elem = A> + DataMut,
{
    fn factorize_into(self) -> Result<LUFactorized<S>> {
        let mut a = self.into_lapack();
        let ipiv = unsafe { A::lu(&mut a)? };
        Ok(LUFactorized {
            a: a.into(),
            ipiv: ipiv,
        })
    }
}

impl<A, Si> Factorize<OwnedRepr<A>> for ArrayBase<Si, Ix2>
where
    A: Scalar,
    Si: Data<Elem = A>,
{
    fn factorize(&self) -> Result<LUFactorized<OwnedRepr<A>>> {
        let mut a = self.to_lapack_clone();
        let ipiv = unsafe { A::lu(&mut a)? };
        Ok(LUFactorized {
            a: a.into(),
            ipiv: ipiv,
        })
    }
}

/// An interface for inverting matrix refs.
pub trait Inverse {
    type Output;
    /// Computes the inverse of the matrix.
    fn inv(&self) -> Result<Self::Output>;
}

/// An interface for inverting matrices.
pub trait InverseInto {
    type Output;
    /// Computes the inverse of the matrix.
    fn inv_into(self) -> Result<Self::Output>;
}

impl<A, S> InverseInto for LUFactorized<S>
where
    A: Scalar,
    S: DataMut<Elem = A>,
{
    type Output = ArrayBase<S, Ix2>;

    fn inv_into(mut self) -> Result<ArrayBase<S, Ix2>> {
        unsafe {
            A::inv(
                self.a.square_layout()?,
                self.a.as_allocated_mut()?,
                &self.ipiv,
            )?
        };
        Ok(self.a)
    }
}

impl<A, S> Inverse for LUFactorized<S>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    type Output = Array2<A>;

    fn inv(&self) -> Result<Array2<A>> {
        let f = LUFactorized {
            a: replicate(&self.a),
            ipiv: self.ipiv.clone(),
        };
        f.inv_into()
    }
}

impl<A, S> InverseInto for ArrayBase<S, Ix2>
where
    A: Scalar,
    S: DataOwned<Elem = A> + DataMut,
{
    type Output = Self;

    fn inv_into(self) -> Result<Self::Output> {
        let f = self.factorize_into()?;
        f.inv_into()
    }
}

impl<A, Si> Inverse for ArrayBase<Si, Ix2>
where
    A: Scalar,
    Si: Data<Elem = A>,
{
    type Output = Array2<A>;

    fn inv(&self) -> Result<Self::Output> {
        let f = self.factorize()?;
        f.inv_into()
    }
}

/// An interface for calculating determinants of matrix refs.
pub trait Determinant<A: Scalar> {
    /// Computes the determinant of the matrix.
    fn det(&self) -> Result<A>;
}

/// An interface for calculating determinants of matrices.
pub trait DeterminantInto<A: Scalar> {
    /// Computes the determinant of the matrix.
    fn det_into(self) -> Result<A>;
}

fn lu_det<'a, A, P, U>(ipiv_iter: P, u_diag_iter: U) -> A
where
    A: Scalar,
    P: Iterator<Item = i32>,
    U: Iterator<Item = &'a A>,
{
    let pivot_sign = if ipiv_iter
        .enumerate()
        .filter(|&(i, pivot)| pivot != i as i32 + 1)
        .count() % 2 == 0
    {
        A::one()
    } else {
        -A::one()
    };
    let (upper_sign, ln_det) = u_diag_iter.fold((A::one(), A::zero()), |(upper_sign, ln_det), &elem| {
        let abs_elem = elem.abs();
        (
            upper_sign * elem.div_real(abs_elem),
            ln_det.add_real(abs_elem.ln()),
        )
    });
    pivot_sign * upper_sign * ln_det.exp()
}

impl<A, S> Determinant<A> for LUFactorized<S>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    fn det(&self) -> Result<A> {
        self.a.ensure_square()?;
        Ok(lu_det(self.ipiv.iter().cloned(), self.a.diag().iter()))
    }
}

impl<A, S> DeterminantInto<A> for LUFactorized<S>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    fn det_into(self) -> Result<A> {
        self.a.ensure_square()?;
        Ok(lu_det(self.ipiv.into_iter(), self.a.into_diag().iter()))
    }
}

impl<A, S> Determinant<A> for ArrayBase<S, Ix2>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    fn det(&self) -> Result<A> {
        self.ensure_square()?;
        match self.factorize() {
            Ok(fac) => fac.det(),
            Err(LinalgError::Lapack(LapackError { return_code })) if return_code > 0 => Ok(A::zero()),
            Err(err) => Err(err),
        }
    }
}

impl<A, S> DeterminantInto<A> for ArrayBase<S, Ix2>
where
    A: Scalar,
    S: DataOwned<Elem = A> + DataMut,
{
    fn det_into(self) -> Result<A> {
        self.ensure_square()?;
        match self.factorize_into() {
            Ok(fac) => fac.det_into(),
            Err(LinalgError::Lapack(LapackError { return_code })) if return_code > 0 => Ok(A::zero()),
            Err(err) => Err(err),
        }
    }
}
