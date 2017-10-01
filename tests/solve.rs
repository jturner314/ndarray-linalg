extern crate ndarray;
#[macro_use]
extern crate ndarray_linalg;
extern crate num_traits;

use ndarray::*;
use ndarray_linalg::*;
use num_traits::{One, Zero};

/// Returns the matrix with the specified `row` and `col` removed.
fn matrix_minor<A, S>(a: ArrayBase<S, Ix2>, (row, col): (usize, usize)) -> Array2<A>
where
    A: Scalar,
    S: Data<Elem = A>,
{
    let mut select_rows = (0..a.rows()).collect::<Vec<_>>();
    select_rows.remove(row);
    let mut select_cols = (0..a.cols()).collect::<Vec<_>>();
    select_cols.remove(col);
    a.select(Axis(0), &select_rows).select(
        Axis(1),
        &select_cols,
    )
}

/// Computes the determinant of matrix `a`.
///
/// Note: This implementation is written to be clearly correct so that it's
/// useful for verification, but it's very inefficient.
fn det_naive<A, S>(a: ArrayBase<S, Ix2>) -> A
where
    A: Scalar,
    S: Data<Elem = A>,
{
    assert_eq!(a.rows(), a.cols());
    match a.cols() {
        0 => A::one(),
        1 => a[(0, 0)],
        cols => {
            (0..cols)
                .map(|col| {
                    let sign = if col % 2 == 0 { A::one() } else { -A::one() };
                    sign * a[(0, col)] * det_naive(matrix_minor(a.view(), (0, col)))
                })
                .fold(A::zero(), |sum, subdet| sum + subdet)
        }
    }
}

#[test]
fn det_empty() {
    macro_rules! det_empty {
        ($elem:ty) => {
            let a: Array2<$elem> = Array2::zeros((0, 0));
            assert_eq!(a.factorize().unwrap().det().unwrap(), One::one());
            assert_eq!(a.factorize().unwrap().det_into().unwrap(), One::one());
            assert_eq!(a.det().unwrap(), One::one());
            assert_eq!(a.det_into().unwrap(), One::one());
        }
    }
    det_empty!(f64);
    det_empty!(f32);
    det_empty!(c64);
    det_empty!(c32);
}

#[test]
fn det_zero() {
    macro_rules! det_zero {
        ($elem:ty) => {
            let a: Array2<$elem> = Array2::zeros((1, 1));
            assert_eq!(a.det().unwrap(), Zero::zero());
            assert_eq!(a.det_into().unwrap(), Zero::zero());
        }
    }
    det_zero!(f64);
    det_zero!(f32);
    det_zero!(c64);
    det_zero!(c32);
}

#[test]
fn det_zero_nonsquare() {
    macro_rules! det_zero_nonsquare {
        ($elem:ty, $shape:expr) => {
            let a: Array2<$elem> = Array2::zeros($shape);
            assert!(a.det().is_err());
            assert!(a.det_into().is_err());
        }
    }
    for &shape in &[(1, 2).into_shape(), (1, 2).f()] {
        det_zero_nonsquare!(f64, shape);
        det_zero_nonsquare!(f32, shape);
        det_zero_nonsquare!(c64, shape);
        det_zero_nonsquare!(c32, shape);
    }
}

#[test]
fn det() {
    macro_rules! det {
        ($elem:ty, $shape:expr, $rtol:expr) => {
            let a: Array2<$elem> = random($shape);
            println!("a = \n{:?}", a);
            let det = det_naive(a.view());
            assert_rclose!(a.factorize().unwrap().det().unwrap(), det, $rtol);
            assert_rclose!(a.factorize().unwrap().det_into().unwrap(), det, $rtol);
            assert_rclose!(a.det().unwrap(), det, $rtol);
            assert_rclose!(a.det_into().unwrap(), det, $rtol);
        }
    }
    for rows in 1..5 {
        for &shape in &[(rows, rows).into_shape(), (rows, rows).f()] {
            det!(f64, shape, 1e-9);
            det!(f32, shape, 1e-4);
            det!(c64, shape, 1e-9);
            det!(c32, shape, 1e-4);
        }
    }
}

#[test]
fn det_nonsquare() {
    macro_rules! det_nonsquare {
        ($elem:ty, $shape:expr) => {
            let a: Array2<$elem> = random($shape);
            assert!(a.factorize().unwrap().det().is_err());
            assert!(a.factorize().unwrap().det_into().is_err());
            assert!(a.det().is_err());
            assert!(a.det_into().is_err());
        }
    }
    for &dims in &[(1, 0), (1, 2), (2, 1), (2, 3)] {
        for &shape in &[dims.clone().into_shape(), dims.clone().f()] {
            det_nonsquare!(f64, shape);
            det_nonsquare!(f32, shape);
            det_nonsquare!(c64, shape);
            det_nonsquare!(c32, shape);
        }
    }
}

#[test]
fn solve() {
    macro_rules! solve {
        ($elem:ty, $a_shape:expr, $x_shape:expr, $x_dim:ty, $rtol:expr) => {
            let a: Array2<$elem> = random($a_shape);
            let x: Array<$elem, $x_dim> = random($x_shape);
            let b = a.dot(&x);
            println!("a = \n{:?}", a);
            println!("x = \n{:?}", x);
            assert_close_max!(&a.solve(&b).unwrap(), &x, $rtol);
            assert_close_max!(&a.solve_into(b.clone()).unwrap(), &x, $rtol);
            assert_close_max!(&a.solve_inplace(&mut b.clone()).unwrap(), &x, $rtol);
            assert_close_max!(&a.factorize().unwrap().solve(&b).unwrap(), &x, $rtol);
            assert_close_max!(&a.factorize().unwrap().solve_into(b.clone()).unwrap(), &x, $rtol);
            assert_close_max!(&a.factorize().unwrap().solve_inplace(&mut b.clone()).unwrap(), &x, $rtol);
        }
    }
    macro_rules! solve_all {
        ($a_shape:expr, $x_shape:expr, $x_dim:ty) => {
            solve!(f64, $a_shape, $x_shape, $x_dim, 1e-9);
            solve!(f32, $a_shape, $x_shape, $x_dim, 1e-3);
            solve!(c64, $a_shape, $x_shape, $x_dim, 1e-9);
            solve!(c32, $a_shape, $x_shape, $x_dim, 1e-3);
        }
    }
    for rows in 0..4 {
        for &a_shape in &[(rows, rows).into_shape(), (rows, rows).f()] {
            solve_all!(a_shape, rows.into_shape(), Ix1);
            for nrhs in 0..rows + 2 {
                for &x_shape in &[(rows, nrhs).into_shape(), (rows, nrhs).f()] {
                    solve_all!(a_shape, x_shape, Ix2);
                }
            }
        }
    }
}

#[test]
fn solve_t() {
    macro_rules! solve_t {
        ($elem:ty, $a_shape:expr, $x_shape:expr, $x_dim:ty, $rtol:expr) => {
            let a: Array2<$elem> = random($a_shape);
            let x: Array<$elem, $x_dim> = random($x_shape);
            let b = a.t().dot(&x);
            println!("a = \n{:?}", a);
            println!("x = \n{:?}", x);
            assert_close_max!(&a.solve_t(&b).unwrap(), &x, $rtol);
            assert_close_max!(&a.solve_t_into(b.clone()).unwrap(), &x, $rtol);
            assert_close_max!(&a.solve_t_inplace(&mut b.clone()).unwrap(), &x, $rtol);
            assert_close_max!(&a.factorize().unwrap().solve_t(&b).unwrap(), &x, $rtol);
            assert_close_max!(&a.factorize().unwrap().solve_t_into(b.clone()).unwrap(), &x, $rtol);
            assert_close_max!(&a.factorize().unwrap().solve_t_inplace(&mut b.clone()).unwrap(), &x, $rtol);
        }
    }
    macro_rules! solve_t_all {
        ($a_shape:expr, $x_shape:expr, $x_dim:ty) => {
            solve_t!(f64, $a_shape, $x_shape, $x_dim, 1e-9);
            solve_t!(f32, $a_shape, $x_shape, $x_dim, 1e-3);
            solve_t!(c64, $a_shape, $x_shape, $x_dim, 1e-9);
            solve_t!(c32, $a_shape, $x_shape, $x_dim, 1e-3);
        }
    }
    for rows in 0..4 {
        for &a_shape in &[(rows, rows).into_shape(), (rows, rows).f()] {
            solve_t_all!(a_shape, rows.into_shape(), Ix1);
            for nrhs in 0..rows + 2 {
                for &x_shape in &[(rows, nrhs).into_shape(), (rows, nrhs).f()] {
                    solve_t_all!(a_shape, x_shape, Ix2);
                }
            }
        }
    }
}

#[test]
fn solve_h() {
    macro_rules! solve_h {
        ($elem:ty, $a_shape:expr, $x_shape:expr, $x_dim:ty, $rtol:expr) => {
            let a: Array2<$elem> = random($a_shape);
            let x: Array<$elem, $x_dim> = random($x_shape);
            let b = a.t().mapv(|elem| elem.conj()).dot(&x);
            println!("a = \n{:?}", a);
            println!("x = \n{:?}", x);
            assert_close_max!(&a.solve_h(&b).unwrap(), &x, $rtol);
            assert_close_max!(&a.solve_h_into(b.clone()).unwrap(), &x, $rtol);
            assert_close_max!(&a.solve_h_inplace(&mut b.clone()).unwrap(), &x, $rtol);
            assert_close_max!(&a.factorize().unwrap().solve_h(&b).unwrap(), &x, $rtol);
            assert_close_max!(&a.factorize().unwrap().solve_h_into(b.clone()).unwrap(), &x, $rtol);
            assert_close_max!(&a.factorize().unwrap().solve_h_inplace(&mut b.clone()).unwrap(), &x, $rtol);
        }
    }
    macro_rules! solve_h_all {
        ($a_shape:expr, $x_shape:expr, $x_dim:ty) => {
            solve_h!(f64, $a_shape, $x_shape, $x_dim, 1e-9);
            solve_h!(f32, $a_shape, $x_shape, $x_dim, 1e-3);
            solve_h!(c64, $a_shape, $x_shape, $x_dim, 1e-9);
            solve_h!(c32, $a_shape, $x_shape, $x_dim, 1e-3);
        }
    }
    for rows in 0..4 {
        for &a_shape in &[(rows, rows).into_shape(), (rows, rows).f()] {
            solve_h_all!(a_shape, rows.into_shape(), Ix1);
            for nrhs in 0..rows + 2 {
                for &x_shape in &[(rows, nrhs).into_shape(), (rows, nrhs).f()] {
                    solve_h_all!(a_shape, x_shape, Ix2);
                }
            }
        }
    }
}
