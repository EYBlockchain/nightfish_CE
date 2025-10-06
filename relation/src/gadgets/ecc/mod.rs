// Copyright (c) 2022 Espresso Systems (espressosys.com)
// This file is part of the Jellyfish library.

// You should have received a copy of the MIT License
// along with the Jellyfish library. If not, see <https://mit-license.org/>.

//! Elliptic curve related gates and gadgets. Including both native and
//! non-native fields.

use super::{EmulatedVariable, EmulationConfig};
use crate::{errors::CircuitError, gates::*, BoolVar, Circuit, PlonkCircuit, Variable};

use ark_ec::{
    short_weierstrass::{Affine as SWAffine, SWCurveConfig},
    twisted_edwards::{Affine, TECurveConfig},
    AffineRepr, CurveConfig, CurveGroup, ScalarMul,
};
use ark_ff::PrimeField;
use ark_std::{
    borrow::ToOwned, boxed::Box, format, marker::PhantomData, string::ToString, vec, vec::Vec,
};

mod conversion;
pub mod emulated;
mod emulated_scalar_msm;
//mod glv;
mod msm;
mod root_unity_glv;
pub use conversion::*;
pub use emulated_scalar_msm::*;
pub use msm::*;
use num_bigint::BigUint;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
/// Enum used for passing elliptic curve points into circuits.
pub enum Point<F: PrimeField> {
    /// A point in TE form.
    TE(F, F),
    /// A point in SW form.
    /// For the identity point we use (0, 1) as the representation as this will almost never be a point on the curve apart form the case
    /// when the constant term of the curve is 1 in which case we use (1, 0).
    SW(F, F),
}

impl<F: PrimeField> Point<F> {
    /// Get the x coordinate of the point.
    pub fn get_x(&self) -> F {
        match self {
            Point::TE(x, _) | Point::SW(x, _) => *x,
        }
    }

    /// Get the y coordinate of the point.
    pub fn get_y(&self) -> F {
        match self {
            Point::TE(_, y) | Point::SW(_, y) => *y,
        }
    }

    /// Return the inverse of the point.
    pub fn inverse(&self) -> Self {
        match self {
            Point::TE(x, y) => Point::TE(-*x, *y),
            Point::SW(x, y) => Point::SW(*x, -*y),
        }
    }

    /// Takes in a mutable reference to a vector and pushes all the coordinates of the Point<F> into the Vec.
    pub fn coords(&self) -> Vec<F> {
        vec![self.get_x(), self.get_y()]
    }
}

impl<F, P> From<Point<F>> for SWAffine<P>
where
    F: PrimeField,
    P: HasTEForm<BaseField = F>,
{
    fn from(value: Point<F>) -> Self {
        match value {
            Point::TE(x, y) => {
                if x == F::zero() && y == F::one() {
                    return Self {
                        x: F::zero(),
                        y: F::zero(),
                        infinity: true,
                    };
                }
                let s = P::BaseField::from(P::S);
                let neg_alpha = P::BaseField::from(P::NEG_ALPHA);
                let beta = P::BaseField::from(P::BETA);

                // Convert back into montgomery form point
                // montgomery_x = (1 + y) / (1 - y)
                // montgomery_y = (1 + y) * beta /(1 - y) * x
                let montgomery_x = (F::one() + y) / (F::one() - y);
                let montgomery_y = (montgomery_x * beta) / x;

                // Convert from Montgomery form to short Weierstrass form
                // sw_x = (mont_x / s) + alpha
                // sw_y = mont_y /s
                let sw_x = montgomery_x / s - neg_alpha;
                let sw_y = montgomery_y / s;

                Self {
                    x: sw_x,
                    y: sw_y,
                    infinity: false,
                }
            },
            Point::SW(x, y) => {
                if x == F::zero() && y == F::one() {
                    SWAffine::<P>::zero()
                } else {
                    SWAffine::<P>::new(x, y)
                }
            },
        }
    }
}

impl<F, P> TryFrom<Point<F>> for Affine<P>
where
    P: HasTEForm<BaseField = F> + TECurveConfig<BaseField = F>,
    F: PrimeField,
{
    type Error = CircuitError;
    fn try_from(value: Point<F>) -> Result<Self, Self::Error> {
        match value {
            Point::TE(x, y) => Ok(Affine::<P>::new(x, y)),
            Point::SW(x, y) => {
                if x == F::zero() && y == F::one() {
                    Ok(Affine::<P>::zero())
                } else if P::has_te_form() {
                    let s = P::BaseField::from(P::S);
                    let neg_alpha = P::BaseField::from(P::NEG_ALPHA);
                    let beta = P::BaseField::from(P::BETA);

                    let montgomery_x = s * (x + neg_alpha);
                    let montgomery_y = s * y;

                    let te_y = (montgomery_x - F::one()) / (montgomery_x + F::one());
                    let te_x = beta * montgomery_x / montgomery_y;
                    Ok(Affine::<P>::new(te_x, te_y))
                } else {
                    Err(CircuitError::ParameterError(
                        "Could not convert from Point<F> to TEAffine<P>".to_string(),
                    ))
                }
            },
        }
    }
}

impl<T, F> From<T> for Point<F>
where
    F: PrimeField,
    T: AffineRepr<BaseField = F>,
    <T as AffineRepr>::Config: HasTEForm<BaseField = F>,
{
    fn from(p: T) -> Self {
        if let Some((x, y)) = p.xy() {
            if (*x * *x * *x)
                + (<T as AffineRepr>::Config::COEFF_A * *x)
                + <T as AffineRepr>::Config::COEFF_B
                == *y * *y
            {
                if <T as AffineRepr>::Config::has_te_form() {
                    // we need to firstly convert this point into
                    // TE form, and then build the point

                    // safe unwrap
                    let s = F::from(<T as AffineRepr>::Config::S);
                    let neg_alpha = F::from(<T as AffineRepr>::Config::NEG_ALPHA);
                    let beta = F::from(<T as AffineRepr>::Config::BETA);

                    // we first transform the Weierstrass point (px, py) to Montgomery point (mx,
                    // my) where mx = s * (px - alpha)
                    // my = s * py
                    let montgomery_x = s * (*x + neg_alpha);
                    let montgomery_y = s * *y;
                    // then we transform the Montgomery point (mx, my) to TE point (ex, ey) where
                    // ex = beta * mx / my
                    // ey = (mx - 1) / (mx + 1)
                    let edwards_x = beta * montgomery_x / montgomery_y;
                    let edwards_y = (montgomery_x - F::one()) / (montgomery_x + F::one());

                    Point::TE(edwards_x, edwards_y)
                } else {
                    Point::SW(*x, *y)
                }
            } else {
                Point::TE(*x, *y)
            }
        } else if <T as AffineRepr>::Config::has_te_form() {
            Point::TE(F::zero(), F::one())
        } else {
            Point::SW(F::zero(), F::one())
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
/// Represent variable of an EC point.
pub enum PointVariable {
    /// TE PointVariable.
    TE(Variable, Variable),
    /// SW Point Variable.
    SW(Variable, Variable),
}

impl PointVariable {
    /// Get the variable representing the x coordinate of the point.
    pub fn get_x(&self) -> Variable {
        match self {
            PointVariable::TE(x, _) | PointVariable::SW(x, _) => *x,
        }
    }

    /// Get the variable representing the y coordinate of the point.
    pub fn get_y(&self) -> Variable {
        match self {
            PointVariable::TE(_, y) | PointVariable::SW(_, y) => *y,
        }
    }

    /// Returns the all the coordinates of the PointVariable as a Vec.
    pub fn get_coords(&self) -> Vec<Variable> {
        vec![self.get_x(), self.get_y()]
    }
}

#[derive(Debug, Clone)]
/// Represent variable of an emulated EC point.
pub enum EmulatedPointVariable<E: PrimeField> {
    /// TE PointVariable.
    TE(EmulatedVariable<E>, EmulatedVariable<E>),
    /// SW Point Variable.
    SW(EmulatedVariable<E>, EmulatedVariable<E>),
}

/// Trait used to define common functionality of circuit representations of elliptic curve points.
pub trait CircuitPoint<F: PrimeField> {
    /// The coordinate type of the point.
    type Coordinate;
    /// Get the x coordinate of the point.
    fn get_x(&self) -> Self::Coordinate;
    /// Get the y coordinate of the point.
    fn get_y(&self) -> Self::Coordinate;
    /// Get the coordinates of the point.
    fn get_coords(&self) -> Vec<Self::Coordinate>;
    /// Prepares the coordinates of the point to be appended to a circuit transcript.
    fn prepare_for_transcript(
        &self,
        circuit: &mut PlonkCircuit<F>,
    ) -> Result<Vec<Variable>, CircuitError>;
}

impl<F: PrimeField> CircuitPoint<F> for PointVariable {
    type Coordinate = Variable;
    /// Get the variable representing the x coordinate of the point.
    fn get_x(&self) -> Variable {
        self.get_x()
    }

    /// Get the variable representing the y coordinate of the point.
    fn get_y(&self) -> Variable {
        self.get_y()
    }

    /// Returns the all the coordinates of the PointVariable as a Vec.
    fn get_coords(&self) -> Vec<Variable> {
        self.get_coords()
    }

    fn prepare_for_transcript(
        &self,
        _circuit: &mut PlonkCircuit<F>,
    ) -> Result<Vec<Variable>, CircuitError> {
        Ok(self.get_coords())
    }
}

impl<F, E> CircuitPoint<F> for EmulatedPointVariable<E>
where
    F: PrimeField,
    E: EmulationConfig<F>,
{
    type Coordinate = EmulatedVariable<E>;
    /// Get the x coordinate of the point as [`Variable`]'s.
    fn get_x(&self) -> Self::Coordinate {
        match self {
            EmulatedPointVariable::TE(x, _) | EmulatedPointVariable::SW(x, _) => x.clone(),
        }
    }

    /// Get the y coordinate of the point.
    fn get_y(&self) -> Self::Coordinate {
        match self {
            EmulatedPointVariable::TE(_, y) | EmulatedPointVariable::SW(_, y) => y.clone(),
        }
    }

    /// Get the coordinates of the point.
    fn get_coords(&self) -> Vec<Self::Coordinate> {
        vec![self.get_x(), self.get_y()]
    }

    fn prepare_for_transcript(
        &self,
        circuit: &mut PlonkCircuit<F>,
    ) -> Result<Vec<Variable>, CircuitError> {
        let x_vars = circuit.convert_for_transcript(&self.get_x())?;
        let y_vars = circuit.convert_for_transcript(&self.get_y())?;
        Ok(x_vars.into_iter().chain(y_vars).collect())
    }
}

// ECC related gates
impl<F: PrimeField> PlonkCircuit<F> {
    /// Return the witness point for the circuit
    pub fn point_witness(&self, point_var: &PointVariable) -> Result<Point<F>, CircuitError> {
        self.check_point_var_bound(point_var)?;
        match *point_var {
            PointVariable::SW(x, y) => {
                let x = self.witness(x)?;
                let y = self.witness(y)?;
                Ok(Point::SW(x, y))
            },
            PointVariable::TE(x, y) => {
                let x = self.witness(x)?;
                let y = self.witness(y)?;
                Ok(Point::TE(x, y))
            },
        }
    }

    /// Add a new EC point (as witness) to the circuit
    pub fn create_point_variable(
        &mut self,
        point: &Point<F>,
    ) -> Result<PointVariable, CircuitError> {
        match point {
            Point::TE(x, y) => {
                let x_var = self.create_variable(*x)?;
                let y_var = self.create_variable(*y)?;
                Ok(PointVariable::TE(x_var, y_var))
            },

            Point::SW(x, y) => {
                let x_var = self.create_variable(*x)?;
                let y_var = self.create_variable(*y)?;
                Ok(PointVariable::SW(x_var, y_var))
            },
        }
    }

    /// Add a new EC point (as a constant) to the circuit
    pub fn create_constant_point_variable(
        &mut self,
        point: &Point<F>,
    ) -> Result<PointVariable, CircuitError> {
        match point {
            Point::TE(x, y) => {
                let x_var = self.create_constant_variable(*x)?;
                let y_var = self.create_constant_variable(*y)?;
                Ok(PointVariable::TE(x_var, y_var))
            },

            Point::SW(x, y) => {
                let x_var = self.create_constant_variable(*x)?;
                let y_var = self.create_constant_variable(*y)?;
                Ok(PointVariable::SW(x_var, y_var))
            },
        }
    }

    /// Add a new EC point (as public input) to the circuit
    pub fn create_public_point_variable(
        &mut self,
        point: &Point<F>,
    ) -> Result<PointVariable, CircuitError> {
        match point {
            Point::TE(x, y) => {
                let x_var = self.create_public_variable(*x)?;
                let y_var = self.create_public_variable(*y)?;
                Ok(PointVariable::TE(x_var, y_var))
            },

            Point::SW(x, y) => {
                let x_var = self.create_public_variable(*x)?;
                let y_var = self.create_public_variable(*y)?;
                Ok(PointVariable::SW(x_var, y_var))
            },
        }
    }

    /// Add a new EC point (as emulated variable) to the circuit
    pub fn create_emulated_point_variable<E: EmulationConfig<F>>(
        &mut self,
        point: &Point<E>,
    ) -> Result<EmulatedPointVariable<E>, CircuitError> {
        match point {
            Point::TE(x, y) => {
                let x_var = self.create_emulated_variable(*x)?;
                let y_var = self.create_emulated_variable(*y)?;
                Ok(EmulatedPointVariable::TE(x_var, y_var))
            },

            Point::SW(x, y) => {
                let x_var = self.create_emulated_variable(*x)?;
                let y_var = self.create_emulated_variable(*y)?;
                Ok(EmulatedPointVariable::SW(x_var, y_var))
            },
        }
    }

    /// Return a BoolVar that is 1 iff `point` is the neutral element
    pub fn is_emulated_neutral_point<P, E>(
        &mut self,
        point: &EmulatedPointVariable<E>,
    ) -> Result<BoolVar, CircuitError>
    where
        P: HasTEForm<BaseField = E, ScalarField = F>,
        E: EmulationConfig<F> + PrimeField,
    {
        match point {
            EmulatedPointVariable::TE(x, y) | EmulatedPointVariable::SW(x, y) => {
                let zero = self.emulated_zero::<E>();
                let one = self.emulated_one::<E>();

                let is_x_zero = self.is_emulated_var_equal::<E>(x, &zero)?;
                let is_y_one = self.is_emulated_var_equal::<E>(y, &one)?;
                self.logic_and(is_x_zero, is_y_one)
            },
        }
    }

    /// Constrains the given emulated point variable to be on the curve in SW form.
    pub fn enforce_emulated_on_curve_sw<P, E>(
        &mut self,
        p: &EmulatedPointVariable<E>,
    ) -> Result<(), CircuitError>
    where
        P: HasTEForm<BaseField = E, ScalarField = F>,
        E: EmulationConfig<F> + PrimeField,
    {
        let (x, y) = match p {
            EmulatedPointVariable::SW(x, y) => (x, y),
            EmulatedPointVariable::TE(..) => {
                return Err(CircuitError::ParameterError(
                    "expected SW point, got TE".to_string(),
                ))
            },
        };

        // y^2 ?= x^3 + a*x + b
        let x2 = self.emulated_mul::<E>(x, x)?;
        let x3 = self.emulated_mul::<E>(&x2, x)?;
        let ax = self.emulated_mul_constant::<E>(x, P::COEFF_A)?;
        let rhs = self.emulated_add::<E>(&x3, &ax)?;
        let rhs = self.emulated_add_constant::<E>(&rhs, P::COEFF_B)?;
        let y2 = self.emulated_mul::<E>(y, y)?;

        self.enforce_emulated_var_equal::<E>(&y2, &rhs)
    }

    /// Constrains the given emulated point variable to be on the curve.
    /// Twisted Edwards form is not supported.
    pub fn enforce_emulated_on_curve<P, E>(
        &mut self,
        p: &EmulatedPointVariable<E>,
    ) -> Result<(), CircuitError>
    where
        P: HasTEForm<BaseField = E, ScalarField = F>,
        E: EmulationConfig<F> + PrimeField,
    {
        match p {
            EmulatedPointVariable::SW(..) => self.enforce_emulated_on_curve_sw::<P, E>(p),
            EmulatedPointVariable::TE(..) => unimplemented!(), // We use the BN254 and Grumpkin curves for which has_te_form() is false
        }
    }

    /// Obtain a point variable of the conditional selection from 4 point
    /// candidates, (b0, b1) are two boolean variables indicating the choice
    /// P_b0+2b1 where P0 = (0, 1) the neutral point, P1, P2, P3 are input
    /// parameters.
    /// A bad PointVariable would be returned if (b0, b1) are not boolean
    /// variables, that would ultimately failed to build a correct circuit.
    fn quaternary_point_select(
        &mut self,
        b0: BoolVar,
        b1: BoolVar,
        point1: &PointVariable,
        point2: &PointVariable,
        point3: &PointVariable,
    ) -> Result<PointVariable, CircuitError>
    where
        F:,
    {
        self.check_var_bound(b0.into())?;
        self.check_var_bound(b1.into())?;

        match (point1, point2, point3) {
            (PointVariable::TE(x1, y1), PointVariable::TE(x2, y2), PointVariable::TE(x3, y3)) => {
                let selected_point = {
                    let selected = match (
                        self.witness(b0.into())? == F::one(),
                        self.witness(b1.into())? == F::one(),
                    ) {
                        (false, false) => Point::TE(F::zero(), F::one()),
                        (true, false) => {
                            Point::TE(self.witness(*x1)?, self.witness(*y1)?).to_owned()
                        },
                        (false, true) => {
                            Point::TE(self.witness(*x2)?, self.witness(*y2)?).to_owned()
                        },
                        (true, true) => {
                            Point::TE(self.witness(*x3)?, self.witness(*y3)?).to_owned()
                        },
                    };
                    // create new point with the same (x, y) coordinates
                    self.create_point_variable(&selected)?
                };
                let wire_vars_x = [b0.into(), b1.into(), 0, 0, selected_point.get_x()];
                self.insert_gate(
                    &wire_vars_x,
                    Box::new(QuaternaryPointSelectXGate {
                        x1: self.witness(*x1)?,
                        x2: self.witness(*x2)?,
                        x3: self.witness(*x3)?,
                    }),
                )?;
                let wire_vars_y = [b0.into(), b1.into(), 0, 0, selected_point.get_y()];
                self.insert_gate(
                    &wire_vars_y,
                    Box::new(QuaternaryPointSelectYGate {
                        y1: self.witness(*y1)?,
                        y2: self.witness(*y2)?,
                        y3: self.witness(*y3)?,
                    }),
                )?;

                Ok(selected_point)
            },

            (PointVariable::SW(x1, y1), PointVariable::SW(x2, y2), PointVariable::SW(x3, y3)) => {
                let selected_point = {
                    let selected = match (
                        self.witness(b0.into())? == F::one(),
                        self.witness(b1.into())? == F::one(),
                    ) {
                        (false, false) => Point::SW(F::zero(), F::one()),
                        (true, false) => {
                            Point::SW(self.witness(*x1)?, self.witness(*y1)?).to_owned()
                        },
                        (false, true) => {
                            Point::SW(self.witness(*x2)?, self.witness(*y2)?).to_owned()
                        },
                        (true, true) => {
                            Point::SW(self.witness(*x3)?, self.witness(*y3)?).to_owned()
                        },
                    };
                    // create new point with the same (x, y,z) coordinates
                    self.create_point_variable(&selected)?
                };

                let wire_vars_x = [b0.into(), b1.into(), 0, 0, selected_point.get_x()];
                self.insert_gate(
                    &wire_vars_x,
                    Box::new(QuaternaryPointSelectXGate {
                        x1: self.witness(*x1)?,
                        x2: self.witness(*x2)?,
                        x3: self.witness(*x3)?,
                    }),
                )?;
                let y = selected_point.get_y();
                let wire_vars_y = [b0.into(), b1.into(), 0, 0, y];
                self.insert_gate(
                    &wire_vars_y,
                    Box::new(QuaternaryPointSelectYGate {
                        y1: self.witness(*y1)?,
                        y2: self.witness(*y2)?,
                        y3: self.witness(*y3)?,
                    }),
                )?;

                Ok(selected_point)
            },
            _ => Err(CircuitError::ParameterError(
                "Incompatible point representations quarternary select gate".to_string(),
            )),
        }
    }

    /// Obtain a point variable of the conditional selection from 2 point
    /// variables. `b` is a boolean variable that indicates selection of P_b
    /// from (P0, P1).
    /// Return error if invalid input parameters are provided.
    pub fn binary_point_vars_select(
        &mut self,
        b: BoolVar,
        point0: &PointVariable,
        point1: &PointVariable,
    ) -> Result<PointVariable, CircuitError> {
        self.check_var_bound(b.into())?;
        self.check_point_var_bound(point0)?;
        self.check_point_var_bound(point1)?;
        let point_0_y = if point0 == &PointVariable::SW(0, 1) {
            1
        } else {
            point0.get_y()
        };
        let selected_x = self.conditional_select(b, point0.get_x(), point1.get_x())?;
        let selected_y = self.conditional_select(b, point_0_y, point1.get_y())?;
        match (*point0, *point1) {
            (PointVariable::TE(..), PointVariable::TE(..)) => {
                Ok(PointVariable::TE(selected_x, selected_y))
            },
            (PointVariable::SW(..), PointVariable::SW(..)) => {
                Ok(PointVariable::SW(selected_x, selected_y))
            },

            _ => {
                // Handle the case when point0 and point1 have different representations
                Err(CircuitError::ParameterError(
                    "Incompatible point representations".to_string(),
                ))
            },
        }
    }

    /// Constrain two point variables to be the same.
    /// Return error if the input point variables are invalid.
    pub fn enforce_point_equal(
        &mut self,
        point0: &PointVariable,
        point1: &PointVariable,
    ) -> Result<(), CircuitError> {
        self.check_point_var_bound(point0)?;
        self.check_point_var_bound(point1)?;

        match (*point0, *point1) {
            (PointVariable::TE(x0, y0), PointVariable::TE(x1, y1)) => {
                self.enforce_equal(x0, x1)?;
                self.enforce_equal(y0, y1)?;
                Ok(())
            },

            (PointVariable::SW(x0, y0), PointVariable::SW(x1, y1)) => {
                self.enforce_equal(x0, x1)?;
                self.enforce_equal(y0, y1)?;
                Ok(())
            },
            _ => {
                // Handle the case when point0 and point1 have different representations
                Err(CircuitError::ParameterError(
                    "Incompatible point representations".to_string(),
                ))
            },
        }
    }

    /// Obtain a bool variable representing whether two point variables are
    /// equal. Return error if point variables are invalid.
    pub fn is_point_equal(
        &mut self,
        point0: &PointVariable,
        point1: &PointVariable,
    ) -> Result<BoolVar, CircuitError> {
        self.check_point_var_bound(point0)?;
        self.check_point_var_bound(point1)?;

        match (point0, point1) {
            (PointVariable::TE(x0, y0), PointVariable::TE(x1, y1)) => {
                let x_eq = self.is_equal(*x0, *x1)?;
                let y_eq = self.is_equal(*y0, *y1)?;
                self.logic_and(x_eq, y_eq)
            },
            (PointVariable::SW(x0, y0), PointVariable::SW(x1, y1)) => {
                let x_eq = self.is_equal(*x0, *x1)?;
                let y_eq = self.is_equal(*y0, *y1)?;
                self.logic_and(x_eq, y_eq)
            },
            _ => Err(CircuitError::ParameterError(
                "Incompatible point representations".to_string(),
            )),
        }
    }
}

impl<F: PrimeField> PlonkCircuit<F> {
    /// Inverse a point variable
    pub fn inverse_point(
        &mut self,
        point_var: &PointVariable,
    ) -> Result<PointVariable, CircuitError>
    where
        F:,
    {
        match *point_var {
            PointVariable::TE(x, y) => {
                let x_neg = self.sub(self.zero(), x)?;
                Ok(PointVariable::TE(x_neg, y))
            },
            PointVariable::SW(x, y) => {
                let y_neg = self.sub(self.zero(), y)?;
                let b = self.is_point_equal(point_var, &PointVariable::SW(0, 1))?;
                let y_choice = self.conditional_select(b, y_neg, y)?;
                Ok(PointVariable::SW(x, y_choice))
            },
        }
    }

    /// Return the point variable for the infinity point
    pub fn neutral_point_variable<P>(&self) -> PointVariable
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
    {
        if P::has_te_form() {
            PointVariable::TE(self.zero(), self.one())
        } else if P::COEFF_B != F::one() {
            // if (0, 1) is not on the curve we will use (0, 1) as the neutral point.
            PointVariable::SW(self.zero(), self.one())
        } else {
            // (0,1) was on the curve so we may use (1,0) as the neutral point.
            PointVariable::SW(self.one(), self.zero())
        }
    }

    /// Constrain a point variable to be a neutral point (0, 1) if
    /// `expected_neutral` == 1
    /// Constrain a point variable to be NOT a neutral point if
    /// `expected_neutral` == 0
    /// Note that `expected_neutral` is already constrained as a bool variable
    pub fn neutral_point_gate<P>(
        &mut self,
        point_var: &PointVariable,
        expected_neutral: BoolVar,
    ) -> Result<(), CircuitError>
    where
        P: HasTEForm<BaseField = F>,
    {
        self.check_point_var_bound(point_var)?;
        self.check_var_bound(expected_neutral.into())?;

        match *point_var {
            PointVariable::TE(x, y) => {
                // constraint 1: b_x = is_equal(x, 0);
                let b_x = self.is_equal(x, self.zero())?;
                // constraint 2: b_y = is_equal(y, 1);
                let b_y = self.is_equal(y, self.one())?;
                // constraint 3: b = b_x * b_y;
                self.mul_gate(b_x.into(), b_y.into(), expected_neutral.into())?;
                Ok(())
            },

            PointVariable::SW(x, y) => {
                let xx = self.mul(x, x)?;
                let wires = &[xx, x, y, y, expected_neutral.into()];
                let q_mul = &[-P::BaseField::one(), P::BaseField::one()];
                self.quad_poly_gate(
                    wires,
                    &[P::BaseField::zero(); 4],
                    q_mul,
                    P::BaseField::one() - P::COEFF_B,
                    -P::COEFF_B,
                )?;
                Ok(())
            },
        }
    }

    /// Obtain a boolean variable indicating whether a point is the neutral
    /// TE: point (0, 1) Return variable with value 1 if it is, or 0 otherwise
    /// SW: point (0, 1, 0) Return variable with value 1 if it is, or 0 otherwise
    /// Return error if input variables are invalid
    pub fn is_neutral_point<P>(
        &mut self,
        point_var: &PointVariable,
    ) -> Result<BoolVar, CircuitError>
    where
        P: HasTEForm<BaseField = F>,
    {
        self.check_point_var_bound(point_var)?;

        match *point_var {
            PointVariable::TE(..) => {
                let b = if self.point_witness(point_var)? == Point::TE(F::zero(), F::one()) {
                    self.create_boolean_variable_unchecked(F::one())?
                } else {
                    self.create_boolean_variable_unchecked(F::zero())?
                };

                self.neutral_point_gate::<P>(point_var, b)?;
                Ok(b)
            },

            PointVariable::SW(..) => {
                let b = if self.point_witness(point_var)? == Point::SW(F::zero(), F::one()) {
                    self.create_boolean_variable_unchecked(F::one())?
                } else {
                    self.create_boolean_variable_unchecked(F::zero())?
                };
                self.neutral_point_gate::<P>(point_var, b)?;
                Ok(b)
            },
        }
    }
    /// Constrain a point to be on certain curve, namely its coordinates satisfy
    /// the curve equation, which is curve-dependent. Currently we only support
    /// checks of a `Affine::<P>` over a base field which is the bls12-381
    /// scalar field
    ///
    /// Returns error if input variables are invalid
    pub fn enforce_on_curve<P>(&mut self, point_var: &PointVariable) -> Result<(), CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
    {
        self.check_point_var_bound(point_var)?;

        match *point_var {
            PointVariable::TE(x, y) => {
                let wire_vars = [x, x, y, y, 1];
                self.insert_gate(
                    &wire_vars,
                    Box::new(EdwardsCurveEquationGate::<P> {
                        _phantom: PhantomData,
                    }),
                )?;
                Ok(())
            },
            PointVariable::SW(x_var, y_var) => {
                let wire_vars = [x_var, x_var, y_var, y_var, self.one()];
                self.insert_gate(
                    &wire_vars,
                    Box::new(SWCurveEquationGate::<P> {
                        _phantom: PhantomData,
                    }),
                )?;
                Ok(())
            },
        }
    }

    /// Constrain variable `point_c` to be the point addition of `point_a` and
    /// `point_b` over an elliptic curve.
    /// Currently only supports Affine::<P> addition.
    ///
    /// Returns error if the input variables are invalid.
    ///
    fn ecc_add_gate<P>(
        &mut self,
        point_a: &PointVariable,
        point_b: &PointVariable,
        point_c: &PointVariable,
    ) -> Result<(), CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
    {
        match (*point_a, *point_b, *point_c) {
            (
                PointVariable::TE(x_1, y_1),
                PointVariable::TE(x_2, y_2),
                PointVariable::TE(x_3, y_3),
            ) => {
                self.check_point_var_bound(point_a)?;
                self.check_point_var_bound(point_b)?;
                self.check_point_var_bound(point_c)?;

                let x_coordinate_wire_vars = [x_1, y_2, x_2, y_1, x_3];
                self.insert_gate(
                    &x_coordinate_wire_vars,
                    Box::new(CurvePointXAdditionGate::<P> {
                        _phantom: PhantomData,
                    }),
                )?;
                let y_coordinate_wire_vars = [x_1, x_2, y_1, y_2, y_3];
                self.insert_gate(
                    &y_coordinate_wire_vars,
                    Box::new(CurvePointYAdditionGate::<P> {
                        _phantom: PhantomData,
                    }),
                )?;
                Ok(())
            },

            (
                PointVariable::SW(x_1, y_1),
                PointVariable::SW(x_2, y_2),
                PointVariable::SW(x3, y3),
            ) => {
                self.check_point_var_bound(point_a)?;
                self.check_point_var_bound(point_b)?;
                self.check_point_var_bound(point_c)?;

                // First we create the z1 and z2 variables
                let x_1_val = self.witness(x_1)?;
                let x_2_val = self.witness(x_2)?;
                let y_1_val = self.witness(y_1)?;
                let y_2_val = self.witness(y_2)?;
                let z_1_val = if self.point_witness(point_a)? == Point::SW(F::zero(), F::one()) {
                    F::zero()
                } else {
                    F::one()
                };
                let z_2_val = if self.point_witness(point_b)? == Point::SW(F::zero(), F::one()) {
                    F::zero()
                } else {
                    F::one()
                };
                let x3_non_scaled = (x_1_val * y_2_val + x_2_val * y_1_val)
                    * (y_1_val * y_2_val - (F::from(3u64) * P::COEFF_B * z_1_val * z_2_val))
                    - (F::from(3u64) * P::COEFF_B)
                        * (y_1_val * z_2_val + y_2_val * z_1_val)
                        * (x_1_val * z_2_val + x_2_val * z_1_val);
                let y3_non_scaled = (y_1_val * y_2_val
                    + F::from(3u64) * P::COEFF_B * z_1_val * z_2_val)
                    * (y_1_val * y_2_val - F::from(3u64) * P::COEFF_B * z_1_val * z_2_val)
                    + F::from(9u64)
                        * P::COEFF_B
                        * x_1_val
                        * x_2_val
                        * (x_1_val * z_2_val + x_2_val * z_1_val);

                let z_1 = self.create_variable(z_1_val)?;
                let z_2 = self.create_variable(z_2_val)?;

                let x_3 = self.create_variable(x3_non_scaled)?;
                let y_3 = self.create_variable(y3_non_scaled)?;

                // Calculate the intermediate variables for x3 and y3
                let tmp_1 = self.mul_add(&[x_1, y_2, x_2, y_1], &[F::one(), F::one()])?;
                let tmp_2 = self.mul_add(
                    &[y_1, y_2, z_1, z_2],
                    &[F::one(), -(F::from(3u8) * P::COEFF_B)],
                )?;
                let tmp_3 = self.mul_add(&[x_1, z_2, x_2, z_1], &[F::one(), F::one()])?;
                let tmp_4 = self.mul_add(&[y_1, z_2, y_2, z_1], &[F::one(), F::one()])?;
                self.mul_add_gate(
                    &[tmp_1, tmp_2, tmp_3, tmp_4, x_3],
                    &[F::one(), -F::from(3u8) * P::COEFF_B],
                )?;

                // Now we do y3
                let out_y_gate_one = F::from(9u8)
                    * P::COEFF_B
                    * (x_1_val * x_2_val * x_2_val * z_1_val
                        + x_1_val * x_1_val * x_2_val * z_2_val)
                    - F::from(9u8) * P::COEFF_B * P::COEFF_B * z_1_val * z_2_val;
                let gate_one_var = self.create_variable(out_y_gate_one)?;
                let wires_in = [z_1, z_2, x_1, x_2, gate_one_var];
                self.insert_gate(
                    &wires_in,
                    Box::new(SWCurvePointYAdditionGateNew::<P> {
                        _phantom: PhantomData,
                    }),
                )?;

                let wires_in = [gate_one_var, self.zero(), y_1, y_2, y_3];
                self.insert_gate(
                    &wires_in,
                    Box::new(SWCurvePointYAdditionGateNew2::<P> {
                        _phantom: PhantomData,
                    }),
                )?;

                let wires_in = [x3, y_3, x_3, y3, self.zero()];
                let q_mul = [F::one(), -F::one()];
                self.mul_add_gate(&wires_in, &q_mul)?;

                Ok(())
            },
            _ => {
                // Handle the case when point0 and point1 have different representations
                Err(CircuitError::ParameterError(
                    "Incompatible point representations ecc add gate".to_string(),
                ))
            },
        }
    }

    /// Constrain variable `point_c` to be the point addition of `point_a` and
    /// `point_b` over an elliptic curve.
    /// Currently only supports Affine::<P> addition.
    ///
    /// Returns error if the input variables are invalid.
    ///
    #[allow(dead_code)]
    fn ecc_add_gate_lnz<P>(
        &mut self,
        point_a: &PointVariable,
        point_b: &PointVariable,
        point_c: &PointVariable,
    ) -> Result<(), CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
    {
        match (*point_a, *point_b, *point_c) {
            (
                PointVariable::TE(x_1, y_1),
                PointVariable::TE(x_2, y_2),
                PointVariable::TE(x_3, y_3),
            ) => {
                self.check_point_var_bound(point_a)?;
                self.check_point_var_bound(point_b)?;
                self.check_point_var_bound(point_c)?;

                let x_coordinate_wire_vars = [x_1, y_2, x_2, y_1, x_3];
                self.insert_gate(
                    &x_coordinate_wire_vars,
                    Box::new(CurvePointXAdditionGate::<P> {
                        _phantom: PhantomData,
                    }),
                )?;
                let y_coordinate_wire_vars = [x_1, x_2, y_1, y_2, y_3];
                self.insert_gate(
                    &y_coordinate_wire_vars,
                    Box::new(CurvePointYAdditionGate::<P> {
                        _phantom: PhantomData,
                    }),
                )?;
                Ok(())
            },

            (
                PointVariable::SW(x_1, y_1),
                PointVariable::SW(x_2, y_2),
                PointVariable::SW(x3, y3),
            ) => {
                self.check_point_var_bound(point_a)?;
                self.check_point_var_bound(point_b)?;
                self.check_point_var_bound(point_c)?;

                // First we create the z1 and z2 variables
                let x_1_val = self.witness(x_1)?;
                let x_2_val = self.witness(x_2)?;
                let y_1_val = self.witness(y_1)?;
                let y_2_val = self.witness(y_2)?;

                let z_2_val = if self.point_witness(point_b)? == Point::SW(F::zero(), F::one()) {
                    F::zero()
                } else {
                    F::one()
                };
                let x3_non_scaled = (x_1_val * y_2_val + x_2_val * y_1_val)
                    * (y_1_val * y_2_val - (F::from(3u64) * P::COEFF_B * z_2_val))
                    - (F::from(3u64) * P::COEFF_B)
                        * (y_1_val * z_2_val + y_2_val)
                        * (x_1_val * z_2_val + x_2_val);
                let y3_non_scaled = (y_1_val * y_2_val + F::from(3u64) * P::COEFF_B * z_2_val)
                    * (y_1_val * y_2_val - F::from(3u64) * P::COEFF_B * z_2_val)
                    + F::from(9u64)
                        * P::COEFF_B
                        * x_1_val
                        * x_2_val
                        * (x_1_val * z_2_val + x_2_val);

                let z_2 = self.create_variable(z_2_val)?;

                let x_3 = self.create_variable(x3_non_scaled)?;
                let y_3 = self.create_variable(y3_non_scaled)?;

                // Calculate the intermediate variables for x3 and y3
                let tmp_1 = self.mul_add(&[x_1, y_2, x_2, y_1], &[F::one(), F::one()])?;
                let tmp_2 = self.mul_add(
                    &[y_1, y_2, self.one(), z_2],
                    &[F::one(), -(F::from(6u8) * P::COEFF_B)],
                )?;
                let tmp_3 = self.mul_add(&[x_1, z_2, x_2, self.one()], &[F::one(), F::one()])?;
                let tmp_4 = self.mul_add(&[y_1, z_2, y_2, self.one()], &[F::one(), F::one()])?;
                self.mul_add_gate(
                    &[tmp_1, tmp_2, tmp_3, tmp_4, x_3],
                    &[F::one(), -F::from(3u8) * P::COEFF_B],
                )?;

                // Now we do y3
                let out_y_gate_one = F::from(9u8)
                    * P::COEFF_B
                    * (x_1_val * x_2_val * x_2_val + x_1_val * x_1_val * x_2_val * z_2_val)
                    - F::from(9u8) * P::COEFF_B * P::COEFF_B * z_2_val;
                let gate_one_var = self.create_variable(out_y_gate_one)?;
                let wires_in = [self.one(), z_2, x_1, x_2, gate_one_var];
                self.insert_gate(
                    &wires_in,
                    Box::new(SWCurvePointYAdditionGateNew::<P> {
                        _phantom: PhantomData,
                    }),
                )?;

                let wires_in = [gate_one_var, self.zero(), y_1, y_2, y_3];
                self.insert_gate(
                    &wires_in,
                    Box::new(SWCurvePointYAdditionGateNew2::<P> {
                        _phantom: PhantomData,
                    }),
                )?;

                let wires_in = [x3, y_3, x_3, y3, self.zero()];
                let q_mul = [F::one(), -F::one()];
                self.mul_add_gate(&wires_in, &q_mul)?;

                Ok(())
            },
            _ => {
                // Handle the case when point0 and point1 have different representations
                Err(CircuitError::ParameterError(
                    "Incompatible point representations ecc add gate".to_string(),
                ))
            },
        }
    }

    /// Constrain variable `point_c` to be the point addition of `point_a` and
    /// `point_b` over an elliptic curve.
    /// Currently only supports Affine::<P> addition.
    ///
    /// Returns error if the input variables are invalid.
    ///
    fn ecc_add_gate_no_neutral<P>(
        &mut self,
        point_a: &PointVariable,
        point_b: &PointVariable,
        point_c: &PointVariable,
    ) -> Result<(), CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
    {
        match (*point_a, *point_b, *point_c) {
            (
                PointVariable::SW(x_1, y_1),
                PointVariable::SW(x_2, y_2),
                PointVariable::SW(x3, y3),
            ) => {
                self.check_point_var_bound(point_a)?;
                self.check_point_var_bound(point_b)?;
                self.check_point_var_bound(point_c)?;

                // First we create the z1 and z2 variables
                let x_1_val = self.witness(x_1)?;
                let x_2_val = self.witness(x_2)?;
                let y_1_val = self.witness(y_1)?;
                let y_2_val = self.witness(y_2)?;

                let x3_non_scaled = (x_1_val * y_2_val + x_2_val * y_1_val)
                    * (y_1_val * y_2_val - (F::from(3u64) * P::COEFF_B))
                    - (F::from(3u64) * P::COEFF_B) * (y_1_val + y_2_val) * (x_1_val + x_2_val);
                let y3_non_scaled = (y_1_val * y_2_val + F::from(3u64) * P::COEFF_B)
                    * (y_1_val * y_2_val - F::from(3u64) * P::COEFF_B)
                    + F::from(9u64) * P::COEFF_B * x_1_val * x_2_val * (x_1_val + x_2_val);

                let x_3 = self.create_variable(x3_non_scaled)?;
                let y_3 = self.create_variable(y3_non_scaled)?;
                let wires_in = [x_1, x_2, y_1, y_2, x_3];
                self.insert_gate(
                    &wires_in,
                    Box::new(SWCurvePointXAdditionGate::<P> {
                        _phantom: PhantomData,
                    }),
                )?;
                let wires_in = [x_1, x_2, y_1, y_2, y_3];
                self.insert_gate(
                    &wires_in,
                    Box::new(SWCurvePointYAdditionGate::<P> {
                        _phantom: PhantomData,
                    }),
                )?;

                let wires_in = [x3, y_3, x_3, y3, self.zero()];
                let q_mul = [F::one(), -F::one()];
                self.mul_add_gate(&wires_in, &q_mul)?;

                Ok(())
            },
            _ => {
                // Handle the case when point0 and point1 have different representations
                Err(CircuitError::ParameterError(
                    "Incompatible point representations ecc add gate".to_string(),
                ))
            },
        }
    }

    /// Obtain a variable to the point addition result of `point_a` + `point_b`
    /// where "+" is the group operation over an elliptic curve.
    /// Currently only supports `Affine::<P>` addition.
    ///
    /// Returns error if inputs are invalid
    /// For SW-curve
    /// x3 = (x1y2 +x2y1)(y1y2 - 3bz1z2) -3b(y1z2 +y2z1)(x1z2 +x2z1)
    /// y3 = (y1y2 + 3bz1z2)(y1y2 - 3bz1z2) + 9bx1x2(x1z2 + x2z1)
    /// z3 = (y1z2 + y2z1)(y1y2 + 3bz1z2) + 3x1x2(x1y2 +x2y1)
    pub fn ecc_add<P>(
        &mut self,
        point_a: &PointVariable,
        point_b: &PointVariable,
    ) -> Result<PointVariable, CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
    {
        match (*point_a, *point_b) {
            (PointVariable::TE(x1, y1), PointVariable::TE(x2, y2)) => {
                let (x_1, y_1) = (self.witness(x1)?, self.witness(y1)?);
                let (x_2, y_2) = (self.witness(x2)?, self.witness(y2)?);

                let d: F = edwards_coeff_d::<P>();
                let a = edwards_coeff_a::<P>();

                let z = d * x_1 * y_1 * x_2 * y_2; // temporary intermediate value
                let x3 = (x_1 * y_2 + x_2 * y_1) / (F::one() + z);
                let y3 = (-a * x_1 * x_2 + y_2 * y_1) / (F::one() - z);

                let (x_3, y_3) = (self.create_variable(x3)?, self.create_variable(y3)?);
                let point_c = PointVariable::TE(x_3, y_3);
                self.ecc_add_gate::<P>(point_a, point_b, &point_c)?;
                Ok(point_c)
            },

            (PointVariable::SW(x_1, y_1), PointVariable::SW(x_2, y_2)) => {
                let (x_1_val, y_1_val, x_2_val, y_2_val) = (
                    self.witness(x_1)?,
                    self.witness(y_1)?,
                    self.witness(x_2)?,
                    self.witness(y_2)?,
                );
                let point_c = if x_1_val == F::zero() && y_1_val == F::one() {
                    let x_2 = self.create_variable(self.witness(x_2)?)?;
                    let y_2 = self.create_variable(self.witness(y_2)?)?;
                    PointVariable::SW(x_2, y_2)
                } else if x_2_val == F::zero() && y_2_val == F::one() {
                    let x_1 = self.create_variable(self.witness(x_1)?)?;
                    let y_1 = self.create_variable(self.witness(y_1)?)?;
                    PointVariable::SW(x_1, y_1)
                } else {
                    let x3_non_scaled = (x_1_val * y_2_val + x_2_val * y_1_val)
                        * (y_1_val * y_2_val - (F::from(3u64) * P::COEFF_B))
                        - (F::from(3u64) * P::COEFF_B) * (y_1_val + y_2_val) * (x_1_val + x_2_val);
                    let y3_non_scaled = (y_1_val * y_2_val + F::from(3u64) * P::COEFF_B)
                        * (y_1_val * y_2_val - F::from(3u64) * P::COEFF_B)
                        + F::from(9u64) * P::COEFF_B * x_1_val * x_2_val * (x_1_val + x_2_val);
                    let z3 = (y_1_val + y_2_val) * (y_1_val * y_2_val + F::from(3u64) * P::COEFF_B)
                        + F::from(3u64)
                            * x_1_val
                            * x_2_val
                            * (x_1_val * y_2_val + x_2_val * y_1_val);
                    let z3_inv = z3.inverse();
                    if let Some(z3) = z3_inv {
                        let x3 = x3_non_scaled * z3;
                        let y3 = y3_non_scaled * z3;

                        let x_3 = self.create_variable(x3)?;
                        let y_3 = self.create_variable(y3)?;

                        PointVariable::SW(x_3, y_3)
                    } else {
                        let x_3 = self.create_variable(F::zero())?;
                        let y_3 = self.create_variable(F::one())?;
                        PointVariable::SW(x_3, y_3)
                    }
                };

                self.ecc_add_gate::<P>(point_a, point_b, &point_c)?;
                Ok(point_c)
            },
            _ => {
                // Handle the case when point0 and point1 have different representations
                Err(CircuitError::ParameterError(
                    "Incompatible point representations ecc add".to_string(),
                ))
            },
        }
    }

    /// Obtain a variable to the point addition result of `point_a` + `point_b`
    /// where "+" is the group operation over an elliptic curve and neither point is the neutral point.
    /// Currently only supports `Affine::<P>` addition.
    ///
    /// Returns error if inputs are invalid
    /// For SW-curve
    /// x3 = (x1y2 +x2y1)(y1y2 - 3bz1z2) -3b(y1z2 +y2z1)(x1z2 +x2z1)
    /// y3 = (y1y2 + 3bz1z2)(y1y2 - 3bz1z2) + 9bx1x2(x1z2 + x2z1)
    /// z3 = (y1z2 + y2z1)(y1y2 + 3bz1z2) + 3x1x2(x1y2 +x2y1)
    pub fn ecc_add_no_neutral<P>(
        &mut self,
        point_a: &PointVariable,
        point_b: &PointVariable,
    ) -> Result<PointVariable, CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
    {
        match (*point_a, *point_b) {
            (PointVariable::TE(..), PointVariable::TE(..)) => self.ecc_add::<P>(point_a, point_b),

            (PointVariable::SW(x_1, y_1), PointVariable::SW(x_2, y_2)) => {
                let (x_1_val, y_1_val, x_2_val, y_2_val) = (
                    self.witness(x_1)?,
                    self.witness(y_1)?,
                    self.witness(x_2)?,
                    self.witness(y_2)?,
                );

                if (x_1_val == F::zero() && y_1_val == F::one())
                    || (x_2_val == F::zero() && y_2_val == F::one())
                {
                    return Err(CircuitError::ParameterError(format!(
                        "Neutral point in ecc add no neutral. x1: {}, y1: {}, x2: {}, y2: {}",
                        x_1_val, y_1_val, x_2_val, y_2_val
                    )));
                }

                let point_c = if x_1_val == F::zero() && y_1_val == F::one() {
                    let x_2 = self.create_variable(self.witness(x_2)?)?;
                    let y_2 = self.create_variable(self.witness(y_2)?)?;
                    PointVariable::SW(x_2, y_2)
                } else if x_2_val == F::zero() && y_2_val == F::one() {
                    let x_1 = self.create_variable(self.witness(x_1)?)?;
                    let y_1 = self.create_variable(self.witness(y_1)?)?;
                    PointVariable::SW(x_1, y_1)
                } else {
                    let x3_non_scaled = (x_1_val * y_2_val + x_2_val * y_1_val)
                        * (y_1_val * y_2_val - (F::from(3u64) * P::COEFF_B))
                        - (F::from(3u64) * P::COEFF_B) * (y_1_val + y_2_val) * (x_1_val + x_2_val);
                    let y3_non_scaled = (y_1_val * y_2_val + F::from(3u64) * P::COEFF_B)
                        * (y_1_val * y_2_val - F::from(3u64) * P::COEFF_B)
                        + F::from(9u64) * P::COEFF_B * x_1_val * x_2_val * (x_1_val + x_2_val);
                    let z3 = (y_1_val + y_2_val) * (y_1_val * y_2_val + F::from(3u64) * P::COEFF_B)
                        + F::from(3u64)
                            * x_1_val
                            * x_2_val
                            * (x_1_val * y_2_val + x_2_val * y_1_val);
                    let z3_inv = z3.inverse();
                    if let Some(z3) = z3_inv {
                        let x3 = x3_non_scaled * z3;
                        let y3 = y3_non_scaled * z3;

                        let x_3 = self.create_variable(x3)?;
                        let y_3 = self.create_variable(y3)?;

                        PointVariable::SW(x_3, y_3)
                    } else {
                        let x_3 = self.create_variable(F::zero())?;
                        let y_3 = self.create_variable(F::one())?;
                        PointVariable::SW(x_3, y_3)
                    }
                };

                self.ecc_add_gate_no_neutral::<P>(point_a, point_b, &point_c)?;
                Ok(point_c)
            },
            _ => {
                // Handle the case when point0 and point1 have different representations
                Err(CircuitError::ParameterError(
                    "Incompatible point representations ecc add".to_string(),
                ))
            },
        }
    }

    /// Obtain a variable to the point addition result of `point_a` + `point_b`
    /// where "+" is the group operation over an elliptic curve and point_a is not the neutral point.
    /// Currently only supports `Affine::<P>` addition.
    ///
    /// Returns error if inputs are invalid
    /// For SW-curve
    /// x3 = (x1y2 +x2y1)(y1y2 - 3bz1z2) -3b(y1z2 +y2z1)(x1z2 +x2z1)
    /// y3 = (y1y2 + 3bz1z2)(y1y2 - 3bz1z2) + 9bx1x2(x1z2 + x2z1)
    /// z3 = (y1z2 + y2z1)(y1y2 + 3bz1z2) + 3x1x2(x1y2 +x2y1)
    pub fn ecc_add_left_nz<P>(
        &mut self,
        point_a: &PointVariable,
        point_b: &PointVariable,
    ) -> Result<PointVariable, CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
    {
        match (*point_a, *point_b) {
            (PointVariable::TE(x1, y1), PointVariable::TE(x2, y2)) => {
                let (x_1, y_1) = (self.witness(x1)?, self.witness(y1)?);
                let (x_2, y_2) = (self.witness(x2)?, self.witness(y2)?);

                let d: F = edwards_coeff_d::<P>();
                let a = edwards_coeff_a::<P>();

                let z = d * x_1 * y_1 * x_2 * y_2; // temporary intermediate value
                let x3 = (x_1 * y_2 + x_2 * y_1) / (F::one() + z);
                let y3 = (-a * x_1 * x_2 + y_2 * y_1) / (F::one() - z);

                let (x_3, y_3) = (self.create_variable(x3)?, self.create_variable(y3)?);
                let point_c = PointVariable::TE(x_3, y_3);
                self.ecc_add_gate::<P>(point_a, point_b, &point_c)?;
                Ok(point_c)
            },

            (PointVariable::SW(x_1, y_1), PointVariable::SW(x_2, y_2)) => {
                let (x_1_val, y_1_val, x_2_val, y_2_val) = (
                    self.witness(x_1)?,
                    self.witness(y_1)?,
                    self.witness(x_2)?,
                    self.witness(y_2)?,
                );
                let point_c = if x_1_val == F::zero() && y_1_val == F::one() {
                    let x_2 = self.create_variable(self.witness(x_2)?)?;
                    let y_2 = self.create_variable(self.witness(y_2)?)?;
                    PointVariable::SW(x_2, y_2)
                } else if x_2_val == F::zero() && y_2_val == F::one() {
                    let x_1 = self.create_variable(self.witness(x_1)?)?;
                    let y_1 = self.create_variable(self.witness(y_1)?)?;
                    PointVariable::SW(x_1, y_1)
                } else {
                    let x3_non_scaled = (x_1_val * y_2_val + x_2_val * y_1_val)
                        * (y_1_val * y_2_val - (F::from(3u64) * P::COEFF_B))
                        - (F::from(3u64) * P::COEFF_B) * (y_1_val + y_2_val) * (x_1_val + x_2_val);
                    let y3_non_scaled = (y_1_val * y_2_val + F::from(3u64) * P::COEFF_B)
                        * (y_1_val * y_2_val - F::from(3u64) * P::COEFF_B)
                        + F::from(9u64) * P::COEFF_B * x_1_val * x_2_val * (x_1_val + x_2_val);
                    let z3 = (y_1_val + y_2_val) * (y_1_val * y_2_val + F::from(3u64) * P::COEFF_B)
                        + F::from(3u64)
                            * x_1_val
                            * x_2_val
                            * (x_1_val * y_2_val + x_2_val * y_1_val);
                    let z3_inv = z3.inverse();
                    if let Some(z3) = z3_inv {
                        let x3 = x3_non_scaled * z3;
                        let y3 = y3_non_scaled * z3;

                        let x_3 = self.create_variable(x3)?;
                        let y_3 = self.create_variable(y3)?;

                        PointVariable::SW(x_3, y_3)
                    } else {
                        let x_3 = self.create_variable(F::zero())?;
                        let y_3 = self.create_variable(F::one())?;
                        PointVariable::SW(x_3, y_3)
                    }
                };

                self.ecc_add_gate::<P>(point_a, point_b, &point_c)?;
                Ok(point_c)
            },
            _ => {
                // Handle the case when point0 and point1 have different representations
                Err(CircuitError::ParameterError(
                    "Incompatible point representations ecc add".to_string(),
                ))
            },
        }
    }

    /// Obtain a variable that is the result of doubling `point` on an elliptic curve.
    pub fn ecc_double<P>(&mut self, point: &PointVariable) -> Result<PointVariable, CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
    {
        match *point {
            PointVariable::TE(..) => self.ecc_add::<P>(point, point),
            PointVariable::SW(x, y) => {
                self.check_point_var_bound(point)?;
                let neutral_flag = self.is_neutral_point::<P>(point)?;
                let y_square = self.mul(y, y)?;
                let y_cubed = self.mul(y_square, y)?;
                let point_jf = self.point_witness(point)?;
                let aff_point: SWAffine<P> = point_jf.into();
                let double_point = (aff_point + aff_point).into_affine();
                let new_point = self.create_point_variable(&Point::from(double_point))?;

                let wires_in = [new_point.get_x(), y_square, x, y_square, self.zero()];
                self.insert_gate(
                    &wires_in,
                    Box::new(PointDoubleXGate::<P> {
                        _phantom: PhantomData::<P>,
                    }),
                )?;
                let wires_in = [
                    y_square,
                    y_square,
                    y_cubed,
                    new_point.get_y(),
                    neutral_flag.into(),
                ];
                self.insert_gate(
                    &wires_in,
                    Box::new(PointDoubleYGate::<P> {
                        _phantom: PhantomData::<P>,
                    }),
                )?;
                Ok(new_point)
            },
        }
    }
    /// Obtain the fixed-based scalar multiplication result of `scalar` * `Base`
    /// Currently only supports `Affine::<P>` scalar multiplication.
    pub fn fixed_base_scalar_mul<P>(
        &mut self,
        scalar: Variable,
        base: &SWAffine<P>,
    ) -> Result<PointVariable, CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
    {
        self.check_var_bound(scalar)?;

        let mut num_bits = <P as CurveConfig>::ScalarField::MODULUS_BIT_SIZE as usize;
        // `num_bits` needs to be an even number
        num_bits += num_bits & 1;
        let scalar_bits_le = self.unpack(scalar, num_bits)?;
        let fixed_bases = compute_base_points(&base.into_group(), num_bits / 2)?;
        let mut accum = self.neutral_point_variable::<P>();
        for i in 0..num_bits / 2 {
            let b0 = scalar_bits_le.get(2 * i).ok_or_else(|| {
                CircuitError::InternalError(
                    "scalar binary representation has the wrong length".to_string(),
                )
            })?;
            let b1 = scalar_bits_le.get(2 * i + 1).ok_or_else(|| {
                CircuitError::InternalError(
                    "scalar binary representation has the wrong length".to_string(),
                )
            })?;
            let p1 = fixed_bases[0].get(i).ok_or_else(|| {
                CircuitError::InternalError("fixed_bases_1 has the wrong length".to_string())
            })?;
            let p2 = fixed_bases[1].get(i).ok_or_else(|| {
                CircuitError::InternalError("fixed_bases_2 has the wrong length".to_string())
            })?;
            let p3 = fixed_bases[2].get(i).ok_or_else(|| {
                CircuitError::InternalError("fixed_bases_3 has the wrong length".to_string())
            })?;
            let point1 = self.create_point_variable(&Point::from(p1.into_affine()))?;
            let point2 = self.create_point_variable(&Point::from(p2.into_affine()))?;
            let point3 = self.create_point_variable(&Point::from(p3.into_affine()))?;
            let selected = self.quaternary_point_select(*b0, *b1, &point1, &point2, &point3)?;
            accum = self.ecc_add::<P>(&accum, &selected)?;
        }
        Ok(accum)
    }

    /// Obtain a variable of the result of a variable base scalar
    /// multiplication. both `scalar` and `base` are variables.
    /// Currently only supports `Affine::<P>`.
    /// If the parameter is bandersnatch, we will use GLV multiplication.
    pub fn variable_base_scalar_mul<P>(
        &mut self,
        scalar: Variable,
        base: &PointVariable,
    ) -> Result<PointVariable, CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
    {
        self.check_var_bound(scalar)?;
        self.check_point_var_bound(base)?;

        // The scalar field modulus must be less than the base field modulus.
        let base_field_mod: BigUint = F::MODULUS.into();
        let scalar_field_mod: BigUint = P::ScalarField::MODULUS.into();
        if base_field_mod <= scalar_field_mod {
            return Err(CircuitError::ParameterError(format!(
                "base field modulus ({:?}) must be strictly greater than scalar field modulus ({:?})",
                F::MODULUS,
                P::ScalarField::MODULUS
            )));
        }
        msm::MultiScalarMultiplicationCircuit::<F, P>::msm(self, &[*base], &[scalar])
    }

    /// Obtain a variable of the result of a variable base scalar scalar mul with scalars in an emulated field.
    pub fn variable_base_emulated_scalar_mul<P>(
        &mut self,
        scalar: &EmulatedVariable<P::ScalarField>,
        base: &PointVariable,
    ) -> Result<PointVariable, CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
        <P as CurveConfig>::ScalarField: PrimeField + EmulationConfig<F>,
    {
        self.check_vars_bound(&scalar.0[..])?;
        self.check_point_var_bound(base)?;

        // non-bandersnatch multiplication
        EmulMultiScalarMultiplicationCircuit::<F, P>::msm(self, &[*base], &[scalar.clone()])
    }

    /// Obtain a variable of the result of a variable base scalar
    /// multiplication. Both `scalar_bits_le` and `base` are variables,
    /// where `scalar_bits_le` is the little-endian form of the scalar.
    /// Currently only supports `Affine::<P>`.
    pub fn variable_base_binary_scalar_mul<P>(
        &mut self,
        scalar_bits_le: &[BoolVar],
        base: &PointVariable,
    ) -> Result<PointVariable, CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
    {
        for &bit in scalar_bits_le {
            self.check_var_bound(bit.into())?;
        }
        self.check_point_var_bound(base)?;

        let neutral_point_var = self.neutral_point_variable::<P>();
        let mut accum = neutral_point_var;
        for i in (0..scalar_bits_le.len()).rev() {
            let z = self.binary_point_vars_select(scalar_bits_le[i], &neutral_point_var, base)?;
            accum = self.ecc_double::<P>(&accum)?;
            accum = self.ecc_add::<P>(&accum, &z)?;
        }
        Ok(accum)
    }
}

// private helper functions
impl<F: PrimeField> PlonkCircuit<F> {
    fn check_point_var_bound(&self, point_var: &PointVariable) -> Result<(), CircuitError> {
        match *point_var {
            PointVariable::TE(x, y) => {
                self.check_var_bound(x)?;
                self.check_var_bound(y)?;
                Ok(())
            },
            PointVariable::SW(x, y) => {
                self.check_var_bound(x)?;
                self.check_var_bound(y)?;
                Ok(())
            },
        }
    }
}

// Given a base point [G] and a scalar s of length 2*n, denote as s[G] the
// scalar multiplication.
// The function computes:
// {4^i * [G]}_{i=0..n-1}, {2 * 4^i * [G]}_{i=0..n-1}, and {3 * 4^i *
// [G]}_{i=0..n-1}
// TODO (tessico): this used to operate on Affine points, but now it takes in
// Projective points. There are some known issues with outputting projectives,
// we should make sure that the usage here is safe.
fn compute_base_points<E: ScalarMul>(base: &E, len: usize) -> Result<[Vec<E>; 3], CircuitError> {
    if len == 0 {
        return Err(CircuitError::InternalError(
            "compute base points length input parameter must be positive".to_string(),
        ));
    }
    fn next_base<E: ScalarMul>(bases: &[E]) -> Result<E, CircuitError> {
        let last = *bases.last().ok_or_else(|| {
            CircuitError::InternalError(
                "Initialize the fixed base vector before calling this function".to_string(),
            )
        })?;
        Ok(last.double().double())
    }
    fn fill_bases<E: ScalarMul>(bases: &mut Vec<E>, len: usize) -> Result<(), CircuitError> {
        for _ in 1..len {
            bases.push(next_base(bases)?);
        }
        Ok(())
    }

    let mut b = *base;
    // base1 = (B, 4*B, ..., 4^(l-1)*B)
    let mut bases1 = vec![b];
    b = b.double();
    // base2 = (2*B, 2*4*B, ..., 2*4^(l-1)*B)
    let mut bases2 = vec![b];
    b += base;
    // base3 = (3*B, 3*4*B, ..., 3*4^(l-1)*B)
    let mut bases3 = vec![b];

    #[cfg(feature = "parallel")]
    {
        rayon::join(
            || {
                rayon::join(
                    || fill_bases(&mut bases1, len).ok(),
                    || fill_bases(&mut bases2, len).ok(),
                )
            },
            || fill_bases(&mut bases3, len).ok(),
        );
    }

    #[cfg(not(feature = "parallel"))]
    {
        fill_bases(&mut bases1, len).ok();
        fill_bases(&mut bases2, len).ok();
        fill_bases(&mut bases3, len).ok();
    }

    // converting Affine -> Points here.
    // Cannot do it earlier: in `fill_bases` we need to do `double`
    // todo(ZZ): consider removing `Point<T>` completely and directly use
    // `Affine<P>` let bases1 =
    // bases1.iter().map(|e|Point::<F>::from(*e)).collect(); let bases2 =
    // bases2.iter().map(|e|Point::<F>::from(*e)).collect(); let bases3 =
    // bases3.iter().map(|e|Point::<F>::from(*e)).collect();

    Ok([bases1, bases2, bases3])
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{gadgets::test_utils::test_variable_independence_for_circuit, Circuit};
    use ark_bls12_377::{g1::Config as Param761, Fq as Fq377};
    use ark_bn254::Fr as FqGrump;
    use ark_ec::{
        twisted_edwards::{Affine, Projective, TECurveConfig as Config},
        Group,
    };
    use ark_ed_on_bls12_377::{EdwardsConfig as Param377, Fq as FqEd377, Fr};
    use ark_ed_on_bls12_381::{EdwardsConfig as Param381, Fq as FqEd381};
    use ark_ed_on_bls12_381_bandersnatch::EdwardsConfig;
    use ark_ff::{One, UniformRand, Zero};
    use ark_std::str::FromStr;
    use jf_utils::{field_switching, fr_to_fq, test_rng};
    use nf_curves::{
        ed_on_bls_12_381_bandersnatch::{EdwardsConfig as Param381b, Fq as FqEd381b},
        ed_on_bn254::{BabyJubjub, Fq as FqEd254},
        grumpkin::{self, short_weierstrass::SWGrumpkin},
    };

    #[test]
    fn test_is_neutral() -> Result<(), CircuitError> {
        test_is_neutral_helper::<FqEd254, BabyJubjub>()?;
        test_is_neutral_helper::<FqEd381, Param381>()?;
        test_is_neutral_helper::<FqEd381b, Param381b>()?;
        test_is_neutral_helper_mle::<FqGrump, SWGrumpkin>()?;
        test_is_neutral_helper::<Fq377, Param761>()
    }

    fn test_is_neutral_helper<F, P>() -> Result<(), CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
    {
        let mut circuit: PlonkCircuit<F> = PlonkCircuit::new_turbo_plonk();
        let p1 = circuit.create_point_variable(&Point::TE(F::zero(), F::one()))?;
        let p2 = circuit.create_point_variable(&Point::TE(F::from(2353u32), F::one()))?;
        let p1_check = circuit.is_neutral_point::<P>(&p1)?;
        let p2_check = circuit.is_neutral_point::<P>(&p2)?;

        assert_eq!(circuit.witness(p1_check.into())?, F::one());
        assert_eq!(circuit.witness(p2_check.into())?, F::zero());
        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
        *circuit.witness_mut(p1.get_x()) = F::one();
        assert!(circuit.check_circuit_satisfiability(&[]).is_err());
        // Check variable out of bound error.
        assert!(circuit
            .is_neutral_point::<P>(&PointVariable::TE(
                circuit.num_vars(),
                circuit.num_vars() - 1
            ))
            .is_err());

        let circuit_1 = build_is_neutral_circuit::<F, P>(Point::TE(F::zero(), F::one()))?;
        let circuit_2 = build_is_neutral_circuit::<F, P>(Point::TE(F::one(), F::zero()))?;
        test_variable_independence_for_circuit(circuit_1, circuit_2)?;

        Ok(())
    }

    fn test_is_neutral_helper_mle<F, P>() -> Result<(), CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
    {
        let mut circuit: PlonkCircuit<F> = PlonkCircuit::new_turbo_plonk();
        let p1 = circuit.create_point_variable(&Point::TE(F::zero(), F::one()))?;
        let p2 = circuit.create_point_variable(&Point::TE(F::from(2353u32), F::one()))?;
        let p1_check = circuit.is_neutral_point::<P>(&p1)?;
        let p2_check = circuit.is_neutral_point::<P>(&p2)?;

        assert_eq!(circuit.witness(p1_check.into())?, F::one());
        assert_eq!(circuit.witness(p2_check.into())?, F::zero());
        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
        *circuit.witness_mut(p1.get_x()) = F::one();
        assert!(circuit.check_circuit_satisfiability(&[]).is_err());
        // Check variable out of bound error.
        assert!(circuit
            .is_neutral_point::<P>(&PointVariable::TE(
                circuit.num_vars(),
                circuit.num_vars() - 1
            ))
            .is_err());

        Ok(())
    }

    fn build_is_neutral_circuit<F, P>(point: Point<F>) -> Result<PlonkCircuit<F>, CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
    {
        let mut circuit: PlonkCircuit<F> = PlonkCircuit::new_turbo_plonk();
        let p = circuit.create_point_variable(&point)?;
        circuit.is_neutral_point::<P>(&p)?;
        circuit.finalize_for_arithmetization()?;
        Ok(circuit)
    }

    #[test]
    fn test_is_emulated_neutral() -> Result<(), CircuitError> {
        test_is_emulated_neutral_helper::<FqGrump, _, SWGrumpkin>()?;
        test_is_emulated_neutral_helper::<_, FqGrump, ark_bn254::g1::Config>()?;

        Ok(())
    }

    fn test_is_emulated_neutral_helper<E, F, P>() -> Result<(), CircuitError>
    where
        E: EmulationConfig<F>,
        F: PrimeField,
        P: HasTEForm<BaseField = E, ScalarField = F>,
    {
        let mut circuit: PlonkCircuit<F> = PlonkCircuit::new_turbo_plonk();

        let p_neutral = SWAffine::<P>::zero().into();
        let p_other = SWAffine::<P>::rand(&mut ark_std::test_rng()).into();

        let v1 = circuit.create_emulated_point_variable(&p_neutral)?;
        let v2 = circuit.create_emulated_point_variable(&p_other)?;

        let b1 = circuit.is_emulated_neutral_point::<P, E>(&v1)?;
        let b2 = circuit.is_emulated_neutral_point::<P, E>(&v2)?;

        assert_eq!(circuit.witness(b1.into())?, F::one());
        assert_eq!(circuit.witness(b2.into())?, F::zero());
        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
        Ok(())
    }

    fn test_enforce_emulated_on_curve_helper<E, F, P>() -> Result<(), CircuitError>
    where
        E: EmulationConfig<F> + PrimeField,
        F: PrimeField,
        P: HasTEForm<BaseField = E, ScalarField = F>,
    {
        let mut circuit_ok: PlonkCircuit<F> = PlonkCircuit::new_turbo_plonk();

        let p_on = SWAffine::<P>::rand(&mut test_rng());

        let v_on = circuit_ok.create_emulated_point_variable(&p_on.into())?;
        circuit_ok.enforce_emulated_on_curve::<P, E>(&v_on)?;
        assert!(circuit_ok.check_circuit_satisfiability(&[]).is_ok());

        let mut circuit_bad: PlonkCircuit<F> = PlonkCircuit::new_turbo_plonk();

        let (x, y) = (p_on.x, p_on.y);
        let bad_y = y + E::one();

        let bad_point = Point::SW(x, bad_y);
        let v_bad = circuit_bad.create_emulated_point_variable(&bad_point)?;

        circuit_bad.enforce_emulated_on_curve::<P, E>(&v_bad)?;
        assert!(circuit_bad.check_circuit_satisfiability(&[]).is_err());

        Ok(())
    }

    #[test]
    fn test_enforce_emulated_on_curve() -> Result<(), CircuitError> {
        test_enforce_emulated_on_curve_helper::<FqGrump, _, SWGrumpkin>()?;
        test_enforce_emulated_on_curve_helper::<_, FqGrump, ark_bn254::g1::Config>()
    }

    macro_rules! test_enforce_on_curve {
        ($fq:tt, $param:tt, $pt:tt) => {
            let mut circuit: PlonkCircuit<$fq> = PlonkCircuit::new_turbo_plonk();
            let p1 = circuit.create_point_variable(&Point::TE($fq::zero(), $fq::one()))?;
            circuit.enforce_on_curve::<$param>(&p1)?;
            let p2 = circuit.create_point_variable(&Point::from($pt))?;
            circuit.enforce_on_curve::<$param>(&p2)?;
            assert!(circuit.check_circuit_satisfiability(&[]).is_ok());

            let p3 = circuit.create_point_variable(&Point::TE($fq::one(), $fq::one()))?;
            circuit.enforce_on_curve::<$param>(&p3)?;
            assert!(circuit.check_circuit_satisfiability(&[]).is_err());
            // Check variable out of bound error.
            assert!(circuit
                .enforce_on_curve::<$param>(&PointVariable::TE(
                    circuit.num_vars(),
                    circuit.num_vars() - 1
                ))
                .is_err());

            let circuit_1 =
                build_enforce_on_curve_circuit::<_, $param>(Point::TE($fq::zero(), $fq::one()))?;
            let circuit_2 = build_enforce_on_curve_circuit::<_, $param>(Point::TE(
                $fq::from(5u32),
                $fq::from(89u32),
            ))?;
            test_variable_independence_for_circuit(circuit_1, circuit_2)?;
        };
    }

    #[test]
    fn test_enforce_on_curve() -> Result<(), CircuitError> {
        // generator for ed_on_bn254 curve
        let ed_on_254_gen = Point::TE(
            FqEd254::from_str(
                "19698561148652590122159747500897617769866003486955115824547446575314762165298",
            )
            .unwrap(),
            FqEd254::from_str(
                "19298250018296453272277890825869354524455968081175474282777126169995084727839",
            )
            .unwrap(),
        );
        // generator for ed_on_bls377 curve
        let _ed_on_377_gen = Point::TE(
            FqEd377::from_str(
                "4497879464030519973909970603271755437257548612157028181994697785683032656389",
            )
            .unwrap(),
            FqEd377::from_str(
                "4357141146396347889246900916607623952598927460421559113092863576544024487809",
            )
            .unwrap(),
        );
        // generator for ed_on_bls381 curve
        let ed_on_381_gen = Point::TE(
            FqEd381::from_str(
                "8076246640662884909881801758704306714034609987455869804520522091855516602923",
            )
            .unwrap(),
            FqEd381::from_str(
                "13262374693698910701929044844600465831413122818447359594527400194675274060458",
            )
            .unwrap(),
        );
        // generator for ed_on_bls381_bandersnatch curve
        let ed_on_381b_gen = Point::TE(
            FqEd381b::from_str(
                "18886178867200960497001835917649091219057080094937609519140440539760939937304",
            )
            .unwrap(),
            FqEd381b::from_str(
                "19188667384257783945677642223292697773471335439753913231509108946878080696678",
            )
            .unwrap(),
        );
        // generator for bls377 G1 curve
        let bls377_gen = Point::TE(
            Fq377::from_str(
                "71222569531709137229370268896323705690285216175189308202338047559628438110820800641278662592954630774340654489393",
            )
            .unwrap(),
            Fq377::from_str(
                "6177051365529633638563236407038680211609544222665285371549726196884440490905471891908272386851767077598415378235",
            )
            .unwrap(),
        );

        test_enforce_on_curve!(FqEd254, BabyJubjub, ed_on_254_gen);
        // test_enforce_on_curve!(FqEd377, Param377, ed_on_377_gen);
        test_enforce_on_curve!(FqEd381, Param381, ed_on_381_gen);
        test_enforce_on_curve!(FqEd381b, Param381b, ed_on_381b_gen);
        test_enforce_on_curve!(Fq377, Param761, bls377_gen);
        Ok(())
    }

    fn build_enforce_on_curve_circuit<F, P>(
        point: Point<F>,
    ) -> Result<PlonkCircuit<F>, CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
    {
        let mut circuit: PlonkCircuit<F> = PlonkCircuit::new_turbo_plonk();
        let p = circuit.create_point_variable(&point)?;
        circuit.enforce_on_curve::<P>(&p)?;
        circuit.finalize_for_arithmetization()?;
        Ok(circuit)
    }

    #[test]
    fn test_curve_point_addition() -> Result<(), CircuitError> {
        test_curve_point_addition_helper::<FqEd254, BabyJubjub>()?;
        test_curve_point_addition_helper_mle::<FqGrump, SWGrumpkin>()?;
        // test_curve_point_addition_helper::<FqEd377, Param377>()?;
        test_curve_point_addition_helper::<FqEd381, Param381>()?;
        test_curve_point_addition_helper::<FqEd381b, Param381b>()?;
        test_curve_point_addition_helper::<Fq377, Param761>()
    }

    fn test_curve_point_addition_helper<F, P>() -> Result<(), CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
    {
        let mut rng = jf_utils::test_rng();
        let p1 = SWAffine::<P>::rand(&mut rng);
        let p2 = SWAffine::<P>::rand(&mut rng);
        let p3 = (p1 + p2).into_affine();
        let p3 = Point::from(p3);

        let mut circuit: PlonkCircuit<F> = PlonkCircuit::new_turbo_plonk();
        let p1_var = circuit.create_point_variable(&Point::from(p1))?;
        let p2_var = circuit.create_point_variable(&Point::from(p2))?;
        let p3_var = circuit.ecc_add::<P>(&p1_var, &p2_var)?;

        assert_eq!(circuit.witness(p3_var.get_x())?, p3.get_x());
        assert_eq!(circuit.witness(p3_var.get_y())?, p3.get_y());
        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
        *circuit.witness_mut(p3_var.get_x()) = F::zero();
        assert!(circuit.check_circuit_satisfiability(&[]).is_err());
        // Check variable out of bound error.
        assert!(circuit
            .ecc_add::<P>(
                &PointVariable::TE(0, 0),
                &PointVariable::TE(1, circuit.num_vars())
            )
            .is_err());

        let p1 = SWAffine::<P>::rand(&mut rng);
        let p2 = SWAffine::<P>::rand(&mut rng);
        let p3 = SWAffine::<P>::rand(&mut rng);
        let p4 = SWAffine::<P>::rand(&mut rng);
        let circuit_1 =
            build_curve_point_addition_circuit::<F, P>(Point::from(p1), Point::from(p2))?;
        let circuit_2 =
            build_curve_point_addition_circuit::<F, P>(Point::from(p3), Point::from(p4))?;
        test_variable_independence_for_circuit(circuit_1, circuit_2)?;

        Ok(())
    }

    fn test_curve_point_addition_helper_mle<F, P>() -> Result<(), CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
    {
        let mut rng = jf_utils::test_rng();
        let _pass = SWAffine::<P>::rand(&mut rng);
        let p1 = SWAffine::<P>::rand(&mut rng);
        let p2 = SWAffine::<P>::rand(&mut rng);
        let p3 = (p1 + p2).into_affine();
        let p3 = Point::from(p3);

        let mut circuit: PlonkCircuit<F> = PlonkCircuit::new_turbo_plonk();
        let p1_var = circuit.create_point_variable(&Point::from(p1))?;
        let p2_var = circuit.create_point_variable(&Point::from(p2))?;

        let p3_var = circuit.ecc_add::<P>(&p1_var, &p2_var)?;

        assert_eq!(circuit.witness(p3_var.get_x())?, p3.get_x());
        assert_eq!(circuit.witness(p3_var.get_y())?, p3.get_y());
        circuit.check_circuit_satisfiability(&[]).unwrap();
        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
        *circuit.witness_mut(p3_var.get_x()) = F::zero();

        assert!(circuit.check_circuit_satisfiability(&[]).is_err());
        // Check variable out of bound error.
        assert!(circuit
            .ecc_add::<P>(
                &PointVariable::TE(0, 0),
                &PointVariable::TE(1, circuit.num_vars())
            )
            .is_err());

        Ok(())
    }

    fn build_curve_point_addition_circuit<F, P>(
        p1: Point<F>,
        p2: Point<F>,
    ) -> Result<PlonkCircuit<F>, CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
    {
        let mut circuit: PlonkCircuit<F> = PlonkCircuit::new_turbo_plonk();
        let p1_var = circuit.create_point_variable(&p1)?;
        let p2_var = circuit.create_point_variable(&p2)?;
        circuit.ecc_add::<P>(&p1_var, &p2_var)?;
        circuit.finalize_for_arithmetization()?;
        Ok(circuit)
    }

    #[test]
    fn test_quaternary_point_select() -> Result<(), CircuitError> {
        test_quaternary_point_select_helper::<FqEd254, BabyJubjub>()?;
        test_quaternary_point_select_helper_mle::<FqGrump, SWGrumpkin>()?;
        // test_quaternary_point_select_helper::<FqEd377, Param377>()?;
        test_quaternary_point_select_helper::<FqEd381, Param381>()?;
        test_quaternary_point_select_helper::<FqEd381b, Param381b>()?;
        test_quaternary_point_select_helper::<Fq377, Param761>()
    }

    fn test_quaternary_point_select_helper<F, P>() -> Result<(), CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
    {
        let mut rng = jf_utils::test_rng();
        let p1 = SWAffine::<P>::rand(&mut rng);
        let p2 = SWAffine::<P>::rand(&mut rng);
        let p3 = SWAffine::<P>::rand(&mut rng);

        let mut circuit: PlonkCircuit<F> = PlonkCircuit::new_turbo_plonk();
        let false_var = circuit.false_var();
        let true_var = circuit.true_var();
        let point1 = circuit.create_point_variable(&Point::from(p1))?;
        let point2 = circuit.create_point_variable(&Point::from(p2))?;
        let point3 = circuit.create_point_variable(&Point::from(p3))?;

        let select_p0 =
            circuit.quaternary_point_select(false_var, false_var, &point1, &point2, &point3)?;
        assert_eq!(
            Point::TE(F::zero(), F::one()),
            Point::TE(
                circuit.witness(select_p0.get_x())?,
                circuit.witness(select_p0.get_y())?
            )
        );
        let point1 = circuit.create_point_variable(&Point::from(p1))?;
        let point2 = circuit.create_point_variable(&Point::from(p2))?;
        let point3 = circuit.create_point_variable(&Point::from(p3))?;

        let select_p1 =
            circuit.quaternary_point_select(true_var, false_var, &point1, &point2, &point3)?;
        assert_eq!(
            Point::from(p1),
            Point::TE(
                circuit.witness(select_p1.get_x())?,
                circuit.witness(select_p1.get_y())?
            )
        );
        let point1 = circuit.create_point_variable(&Point::from(p1))?;
        let point2 = circuit.create_point_variable(&Point::from(p2))?;
        let point3 = circuit.create_point_variable(&Point::from(p3))?;

        let select_p2 =
            circuit.quaternary_point_select(false_var, true_var, &point1, &point2, &point3)?;
        assert_eq!(
            Point::from(p2),
            Point::TE(
                circuit.witness(select_p2.get_x())?,
                circuit.witness(select_p2.get_y())?
            )
        );
        let point1 = circuit.create_point_variable(&Point::from(p1))?;
        let point2 = circuit.create_point_variable(&Point::from(p2))?;
        let point3 = circuit.create_point_variable(&Point::from(p3))?;

        let select_p3 =
            circuit.quaternary_point_select(true_var, true_var, &point1, &point2, &point3)?;
        assert_eq!(
            Point::from(p3),
            Point::TE(
                circuit.witness(select_p3.get_x())?,
                circuit.witness(select_p3.get_y())?
            )
        );

        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());

        *circuit.witness_mut(select_p3.get_x()) = p2.x;
        *circuit.witness_mut(select_p3.get_y()) = p2.y;
        assert!(circuit.check_circuit_satisfiability(&[]).is_err());

        let circuit_1 = build_quaternary_select_gate::<F, P>(false, false)?;
        let circuit_2 = build_quaternary_select_gate::<F, P>(true, true)?;
        test_variable_independence_for_circuit(circuit_1, circuit_2)?;
        Ok(())
    }

    fn test_quaternary_point_select_helper_mle<F, P>() -> Result<(), CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
    {
        let mut rng = jf_utils::test_rng();
        let p1 = SWAffine::<P>::rand(&mut rng);
        let p2 = SWAffine::<P>::rand(&mut rng);
        let p3 = SWAffine::<P>::rand(&mut rng);

        let mut circuit: PlonkCircuit<F> = PlonkCircuit::new_turbo_plonk();
        let false_var = circuit.false_var();
        let true_var = circuit.true_var();
        let point1 = circuit.create_point_variable(&Point::from(p1))?;
        let point2 = circuit.create_point_variable(&Point::from(p2))?;
        let point3 = circuit.create_point_variable(&Point::from(p3))?;

        let select_p0 =
            circuit.quaternary_point_select(false_var, false_var, &point1, &point2, &point3)?;

        assert_eq!(F::zero(), circuit.witness(select_p0.get_x())?);
        assert_eq!(F::one(), circuit.witness(select_p0.get_y())?);
        let point1 = circuit.create_point_variable(&Point::from(p1))?;
        let point2 = circuit.create_point_variable(&Point::from(p2))?;
        let point3 = circuit.create_point_variable(&Point::from(p3))?;

        let select_p1 =
            circuit.quaternary_point_select(true_var, false_var, &point1, &point2, &point3)?;
        assert_eq!(p1.x, circuit.witness(select_p1.get_x())?);
        assert_eq!(p1.y, circuit.witness(select_p1.get_y())?);
        let point1 = circuit.create_point_variable(&Point::from(p1))?;
        let point2 = circuit.create_point_variable(&Point::from(p2))?;
        let point3 = circuit.create_point_variable(&Point::from(p3))?;

        let select_p2 =
            circuit.quaternary_point_select(false_var, true_var, &point1, &point2, &point3)?;
        assert_eq!(p2.x, circuit.witness(select_p2.get_x())?);
        assert_eq!(p2.y, circuit.witness(select_p2.get_y())?);
        let point1 = circuit.create_point_variable(&Point::from(p1))?;
        let point2 = circuit.create_point_variable(&Point::from(p2))?;
        let point3 = circuit.create_point_variable(&Point::from(p3))?;

        let select_p3 =
            circuit.quaternary_point_select(true_var, true_var, &point1, &point2, &point3)?;
        assert_eq!(p3.x, circuit.witness(select_p3.get_x())?);
        assert_eq!(p3.y, circuit.witness(select_p3.get_y())?);
        circuit.check_circuit_satisfiability(&[]).unwrap();
        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());

        *circuit.witness_mut(select_p3.get_x()) = p2.x;
        *circuit.witness_mut(select_p3.get_y()) = p2.y;
        assert!(circuit.check_circuit_satisfiability(&[]).is_err());

        Ok(())
    }

    fn build_quaternary_select_gate<F, P>(
        b0: bool,
        b1: bool,
    ) -> Result<PlonkCircuit<F>, CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
    {
        let mut circuit: PlonkCircuit<F> = PlonkCircuit::new_turbo_plonk();
        let b0_var = circuit.create_boolean_variable(b0)?;
        let b1_var = circuit.create_boolean_variable(b1)?;

        let mut rng = jf_utils::test_rng();
        let p1 = SWAffine::<P>::rand(&mut rng);
        let p2 = SWAffine::<P>::rand(&mut rng);
        let p3 = SWAffine::<P>::rand(&mut rng);
        let point1 = circuit.create_point_variable(&Point::from(p1))?;
        let point2 = circuit.create_point_variable(&Point::from(p2))?;
        let point3 = circuit.create_point_variable(&Point::from(p3))?;

        circuit.quaternary_point_select(b0_var, b1_var, &point1, &point2, &point3)?;
        circuit.finalize_for_arithmetization()?;
        Ok(circuit)
    }

    #[test]
    fn test_point_equal_gate() -> Result<(), CircuitError> {
        test_point_equal_gate_helper::<FqEd254, BabyJubjub>()?;
        test_point_equal_gate_helper_mle::<FqGrump, SWGrumpkin>()?;
        test_point_equal_gate_helper::<FqEd381, Param381>()?;
        test_point_equal_gate_helper::<FqEd381b, Param381b>()?;
        test_point_equal_gate_helper::<Fq377, Param761>()
    }

    fn test_point_equal_gate_helper<F, P>() -> Result<(), CircuitError>
    where
        F: PrimeField,
        P: Config<BaseField = F> + HasTEForm<BaseField = F>,
    {
        let mut rng = jf_utils::test_rng();
        let p = SWAffine::<P>::rand(&mut rng);

        let mut circuit = PlonkCircuit::<F>::new_turbo_plonk();
        let p1_var = circuit.create_point_variable(&Point::from(p))?;
        let p2_var = circuit.create_point_variable(&Point::from(p))?;
        circuit.enforce_point_equal(&p1_var, &p2_var)?;

        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
        *circuit.witness_mut(p2_var.get_x()) = F::zero();
        assert!(circuit.check_circuit_satisfiability(&[]).is_err());
        // Check variable out of bound error.
        assert!(circuit
            .enforce_point_equal(
                &PointVariable::TE(0, 0),
                &PointVariable::TE(1, circuit.num_vars())
            )
            .is_err());

        let new_p = Affine::<P>::rand(&mut rng);
        let circuit_1 = build_point_equal_circuit(Point::from(p), Point::from(p))?;
        let circuit_2 = build_point_equal_circuit(Point::from(new_p), Point::from(new_p))?;
        test_variable_independence_for_circuit(circuit_1, circuit_2)?;

        Ok(())
    }

    fn test_point_equal_gate_helper_mle<F, P>() -> Result<(), CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
    {
        let mut rng = jf_utils::test_rng();
        let p = SWAffine::<P>::rand(&mut rng);

        let mut circuit = PlonkCircuit::<F>::new_turbo_plonk();
        let p1_var = circuit.create_point_variable(&Point::from(p))?;
        let p2_var = circuit.create_point_variable(&Point::from(p))?;
        circuit.enforce_point_equal(&p1_var, &p2_var)?;

        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
        *circuit.witness_mut(p2_var.get_x()) = F::zero();
        assert!(circuit.check_circuit_satisfiability(&[]).is_err());
        // Check variable out of bound error.
        assert!(circuit
            .enforce_point_equal(
                &PointVariable::TE(0, 0),
                &PointVariable::TE(1, circuit.num_vars())
            )
            .is_err());

        Ok(())
    }

    fn build_point_equal_circuit<F: PrimeField>(
        p1: Point<F>,
        p2: Point<F>,
    ) -> Result<PlonkCircuit<F>, CircuitError> {
        let mut circuit: PlonkCircuit<F> = PlonkCircuit::new_turbo_plonk();
        let p1_var = circuit.create_point_variable(&p1)?;
        let p2_var = circuit.create_point_variable(&p2)?;
        circuit.enforce_point_equal(&p1_var, &p2_var)?;
        circuit.finalize_for_arithmetization()?;
        Ok(circuit)
    }

    #[test]
    fn test_is_equal_point() -> Result<(), CircuitError> {
        test_is_equal_point_helper::<FqEd254, BabyJubjub>()?;
        test_is_equal_point_helper_mle::<FqGrump, SWGrumpkin>()?;
        test_is_equal_point_helper::<FqEd381, Param381>()?;
        test_is_equal_point_helper::<FqEd381b, Param381b>()?;
        test_is_equal_point_helper::<Fq377, Param761>()
    }

    fn test_is_equal_point_helper<F, P>() -> Result<(), CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F> + Config<BaseField = F>,
    {
        let mut rng = jf_utils::test_rng();
        let p1 = Affine::<P>::rand(&mut rng);
        let p2 = p1;
        let p3 = Affine::<P>::rand(&mut rng);

        let mut circuit: PlonkCircuit<F> = PlonkCircuit::new_turbo_plonk();
        let p1_var = circuit.create_point_variable(&Point::from(p1))?;
        let p2_var = circuit.create_point_variable(&Point::from(p2))?;
        let p3_var = circuit.create_point_variable(&Point::from(p3))?;
        let p1_p2_eq = circuit.is_point_equal(&p1_var, &p2_var)?;
        let p1_p3_eq = circuit.is_point_equal(&p1_var, &p3_var)?;

        assert_eq!(circuit.witness(p1_p2_eq.into())?, F::one());
        assert_eq!(circuit.witness(p1_p3_eq.into())?, F::zero());
        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
        *circuit.witness_mut(p2_var.get_x()) = F::zero();
        assert!(circuit.check_circuit_satisfiability(&[]).is_err());
        // Check variable out of bound error.
        assert!(circuit
            .is_point_equal(
                &PointVariable::TE(0, 0),
                &PointVariable::TE(1, circuit.num_vars())
            )
            .is_err());

        let circuit_1 =
            build_is_equal_point_circuit::<F>(Point::from(p1), Point::from(p2), Point::from(p3))?;
        let circuit_2 =
            build_is_equal_point_circuit::<F>(Point::from(p3), Point::from(p3), Point::from(p1))?;
        test_variable_independence_for_circuit(circuit_1, circuit_2)?;

        Ok(())
    }

    fn test_is_equal_point_helper_mle<F, P>() -> Result<(), CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
    {
        let mut rng = jf_utils::test_rng();
        let p1 = SWAffine::<P>::rand(&mut rng);
        let p2 = p1;
        let p3 = SWAffine::<P>::rand(&mut rng);

        let mut circuit: PlonkCircuit<F> = PlonkCircuit::new_turbo_plonk();
        let p1_var = circuit.create_point_variable(&Point::from(p1))?;
        let p2_var = circuit.create_point_variable(&Point::from(p2))?;
        let p3_var = circuit.create_point_variable(&Point::from(p3))?;
        let p1_p2_eq = circuit.is_point_equal(&p1_var, &p2_var)?;
        let p1_p3_eq = circuit.is_point_equal(&p1_var, &p3_var)?;

        assert_eq!(circuit.witness(p1_p2_eq.into())?, F::one());
        assert_eq!(circuit.witness(p1_p3_eq.into())?, F::zero());
        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
        *circuit.witness_mut(p2_var.get_x()) = F::zero();
        assert!(circuit.check_circuit_satisfiability(&[]).is_err());
        // Check variable out of bound error.
        assert!(circuit
            .is_point_equal(
                &PointVariable::TE(0, 0),
                &PointVariable::TE(1, circuit.num_vars())
            )
            .is_err());

        Ok(())
    }

    fn build_is_equal_point_circuit<F>(
        p1: Point<F>,
        p2: Point<F>,
        p3: Point<F>,
    ) -> Result<PlonkCircuit<F>, CircuitError>
    where
        F: PrimeField,
    {
        let mut circuit = PlonkCircuit::new_turbo_plonk();
        let p1_var = circuit.create_point_variable(&p1)?;
        let p2_var = circuit.create_point_variable(&p2)?;
        let p3_var = circuit.create_point_variable(&p3)?;
        circuit.is_point_equal(&p1_var, &p2_var)?;
        circuit.is_point_equal(&p1_var, &p3_var)?;
        circuit.finalize_for_arithmetization()?;
        Ok(circuit)
    }

    #[test]
    fn test_compute_fixed_bases() -> Result<(), CircuitError> {
        test_compute_fixed_bases_helper::<FqEd254, BabyJubjub>()?;
        test_compute_fixed_bases_helper::<FqEd377, Param377>()?;
        test_compute_fixed_bases_helper::<FqEd381, Param381>()?;
        test_compute_fixed_bases_helper::<FqEd381b, Param381b>()?;
        test_compute_fixed_bases_helper::<Fq377, Param761>()
    }

    fn test_compute_fixed_bases_helper<F, P>() -> Result<(), CircuitError>
    where
        F: PrimeField,
        P: Config<BaseField = F>,
    {
        fn check_base_list<F, P>(bases: &[Projective<P>])
        where
            F: PrimeField,
            P: Config<BaseField = F>,
        {
            bases
                .windows(2)
                .for_each(|neighbors| assert!(neighbors[1] == neighbors[0].double().double()));
        }

        let mut rng = jf_utils::test_rng();

        let base = Affine::<P>::rand(&mut rng);
        let base2 = base.into_group().double();
        let base3 = base + base2;

        assert_eq!(
            compute_base_points(&base.into_group(), 1)?,
            [vec![base.into_group()], vec![base2], vec![base3]]
        );
        let size = 10;
        let result = compute_base_points(&base.into_group(), size)?;
        let bases1 = &result[0];
        assert_eq!(bases1.len(), size);
        let bases2 = &result[1];
        assert_eq!(bases2.len(), size);
        let bases3 = &result[2];
        assert_eq!(bases3.len(), size);
        check_base_list(bases1);
        check_base_list(bases2);
        check_base_list(bases3);
        bases1
            .iter()
            .zip(bases2.iter())
            .zip(bases3.iter())
            .for_each(|((&b1, &b2), &b3)| {
                assert!(b2 == b1.double());
                assert!(b3 == b1 + b2);
            });

        Ok(())
    }

    #[test]
    fn test_fixed_based_scalar_mul() -> Result<(), CircuitError> {
        test_fixed_based_scalar_mul_helper_mle::<FqGrump, SWGrumpkin>()?;
        test_fixed_based_scalar_mul_helper::<FqEd254, BabyJubjub>()?;
        test_fixed_based_scalar_mul_helper::<FqEd381, Param381>()?;
        test_fixed_based_scalar_mul_helper::<FqEd381b, Param381b>()?;
        test_fixed_based_scalar_mul_helper::<Fq377, Param761>()
    }

    fn test_fixed_based_scalar_mul_helper<F, P>() -> Result<(), CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
    {
        let mut rng = jf_utils::test_rng();
        let mut circuit: PlonkCircuit<F> = PlonkCircuit::new_turbo_plonk();

        for _ in 0..6 {
            let mut base = SWAffine::<P>::rand(&mut rng);
            let s = P::ScalarField::rand(&mut rng);
            let scalar = circuit.create_variable(fr_to_fq::<F, P>(&s))?;
            let result = circuit.fixed_base_scalar_mul(scalar, &base)?;
            base = (base * s).into();
            assert_eq!(
                Point::from(base),
                Point::TE(
                    circuit.witness(result.get_x())?,
                    circuit.witness(result.get_y())?
                )
            );
        }
        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());

        // wrong witness should fail
        *circuit.witness_mut(2) = F::rand(&mut rng);
        assert!(circuit.check_circuit_satisfiability(&[]).is_err());
        // Check variable out of bound error.
        assert!(circuit
            .fixed_base_scalar_mul(circuit.num_vars(), &SWAffine::<P>::rand(&mut rng))
            .is_err());

        let circuit_1 = build_fixed_based_scalar_mul_circuit::<F, P>(F::from(87u32))?;
        let circuit_2 = build_fixed_based_scalar_mul_circuit::<F, P>(F::from(2u32))?;
        test_variable_independence_for_circuit(circuit_1, circuit_2)?;
        Ok(())
    }

    fn test_fixed_based_scalar_mul_helper_mle<F, P>() -> Result<(), CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
    {
        let mut rng = jf_utils::test_rng();
        let mut circuit: PlonkCircuit<F> = PlonkCircuit::new_turbo_plonk();

        for _ in 0..6 {
            let mut base = SWAffine::<P>::rand(&mut rng);
            let s = P::ScalarField::rand(&mut rng);
            let scalar = circuit.create_variable(fr_to_fq::<F, P>(&s)).unwrap();

            let result = circuit.fixed_base_scalar_mul(scalar, &base).unwrap();

            base = (base * s).into();
            assert_eq!(base.x, circuit.witness(result.get_x()).unwrap());
            assert_eq!(base.y, circuit.witness(result.get_y()).unwrap());
        }
        circuit.check_circuit_satisfiability(&[]).unwrap();
        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());

        // wrong witness should fail
        *circuit.witness_mut(2) = F::rand(&mut rng);
        assert!(circuit.check_circuit_satisfiability(&[]).is_err());
        // Check variable out of bound error.
        assert!(circuit
            .fixed_base_scalar_mul(circuit.num_vars(), &SWAffine::<P>::rand(&mut rng))
            .is_err());

        Ok(())
    }

    fn build_fixed_based_scalar_mul_circuit<F, P>(
        scalar: F,
    ) -> Result<PlonkCircuit<F>, CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
    {
        let mut rng = jf_utils::test_rng();
        let mut circuit: PlonkCircuit<F> = PlonkCircuit::new_turbo_plonk();
        let base = SWAffine::<P>::rand(&mut rng);
        let scalar_var = circuit.create_variable(scalar)?;
        circuit.fixed_base_scalar_mul(scalar_var, &base)?;
        circuit.finalize_for_arithmetization()?;
        Ok(circuit)
    }

    #[test]
    fn test_binary_point_vars_select() -> Result<(), CircuitError> {
        test_binary_point_vars_select_helper::<FqEd254, BabyJubjub>()?;
        test_binary_point_vars_select_helper_mle::<FqGrump, SWGrumpkin>()?;
        test_binary_point_vars_select_helper::<FqEd381, Param381>()?;
        test_binary_point_vars_select_helper::<FqEd381b, Param381b>()?;
        test_binary_point_vars_select_helper::<Fq377, Param761>()
    }
    fn test_binary_point_vars_select_helper<F, P>() -> Result<(), CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F> + Config<BaseField = F>,
    {
        let mut rng = jf_utils::test_rng();
        let p0 = Affine::<P>::rand(&mut rng);
        let p1 = Affine::<P>::rand(&mut rng);
        let p2 = Affine::<P>::rand(&mut rng);

        let mut circuit: PlonkCircuit<F> = PlonkCircuit::new_turbo_plonk();
        let p0_var = circuit.create_point_variable(&Point::from(p0))?;
        let p1_var = circuit.create_point_variable(&Point::from(p1))?;
        let true_var = circuit.true_var();
        let false_var = circuit.false_var();

        let select_p0 = circuit.binary_point_vars_select(false_var, &p0_var, &p1_var)?;
        assert_eq!(
            Point::TE(
                circuit.witness(select_p0.get_x())?,
                circuit.witness(select_p0.get_y())?
            ),
            Point::from(p0)
        );
        let select_p1 = circuit.binary_point_vars_select(true_var, &p0_var, &p1_var)?;
        assert_eq!(
            Point::TE(
                circuit.witness(select_p1.get_x())?,
                circuit.witness(select_p1.get_y())?
            ),
            Point::from(p1)
        );
        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());

        // wrong witness should fail
        *circuit.witness_mut(p1_var.get_x()) = F::rand(&mut rng);
        assert!(circuit.check_circuit_satisfiability(&[]).is_err());
        assert!(circuit
            .binary_point_vars_select(
                false_var,
                &p0_var,
                &PointVariable::TE(p1_var.get_x(), circuit.num_vars()),
            )
            .is_err());

        let circuit_1 =
            build_binary_point_vars_select_circuit::<F>(true, &Point::from(p0), &Point::from(p1))?;
        let circuit_2 =
            build_binary_point_vars_select_circuit::<F>(false, &Point::from(p1), &Point::from(p2))?;
        test_variable_independence_for_circuit(circuit_1, circuit_2)?;

        Ok(())
    }

    fn test_binary_point_vars_select_helper_mle<F, P>() -> Result<(), CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
    {
        let mut rng = jf_utils::test_rng();
        let p0 = SWAffine::<P>::rand(&mut rng);
        let p1 = SWAffine::<P>::rand(&mut rng);

        let mut circuit: PlonkCircuit<F> = PlonkCircuit::new_turbo_plonk();
        let p0_var = circuit.create_point_variable(&Point::from(p0))?;
        let p1_var = circuit.create_point_variable(&Point::from(p1))?;
        let true_var = circuit.true_var();
        let false_var = circuit.false_var();

        let select_p0 = circuit.binary_point_vars_select(false_var, &p0_var, &p1_var)?;
        assert_eq!(circuit.witness(select_p0.get_x())?, p0.x);
        assert_eq!(circuit.witness(select_p0.get_y())?, p0.y);
        let select_p1 = circuit.binary_point_vars_select(true_var, &p0_var, &p1_var)?;
        assert_eq!(circuit.witness(select_p1.get_x())?, p1.x);
        assert_eq!(circuit.witness(select_p1.get_y())?, p1.y);
        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());

        // wrong witness should fail
        *circuit.witness_mut(p1_var.get_x()) = F::rand(&mut rng);
        assert!(circuit.check_circuit_satisfiability(&[]).is_err());
        assert!(circuit
            .binary_point_vars_select(
                false_var,
                &p0_var,
                &PointVariable::TE(p1_var.get_x(), circuit.num_vars()),
            )
            .is_err());

        Ok(())
    }

    fn build_binary_point_vars_select_circuit<F>(
        b: bool,
        p0: &Point<F>,
        p1: &Point<F>,
    ) -> Result<PlonkCircuit<F>, CircuitError>
    where
        F: PrimeField,
    {
        let mut circuit: PlonkCircuit<F> = PlonkCircuit::new_turbo_plonk();
        let b_var = circuit.create_boolean_variable(b)?;
        let p0_var = circuit.create_point_variable(p0)?;
        let p1_var = circuit.create_point_variable(p1)?;
        circuit.binary_point_vars_select(b_var, &p0_var, &p1_var)?;
        circuit.finalize_for_arithmetization()?;
        Ok(circuit)
    }

    #[test]
    fn test_variable_base_scalar_mul() -> Result<(), CircuitError> {
        // test_variable_base_scalar_mul_helper_mle::<FqGrump, SWGrumpkin>()?;
        test_variable_base_scalar_mul_helper::<FqEd254, BabyJubjub>()?;
        // test_variable_base_scalar_mul_helper::<FqEd377, Param377>()?;
        test_variable_base_scalar_mul_helper::<FqEd381, Param381>()?;
        test_variable_base_scalar_mul_helper::<FqEd381b, Param381b>()?;
        test_variable_base_scalar_mul_helper::<Fq377, Param761>()
    }
    fn test_variable_base_scalar_mul_helper<F, P>() -> Result<(), CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
    {
        let mut rng = jf_utils::test_rng();
        let mut circuit: PlonkCircuit<F> = PlonkCircuit::new_ultra_plonk(8);

        for _ in 0..6 {
            let mut base = SWAffine::<P>::rand(&mut rng);
            let s = P::ScalarField::rand(&mut rng);
            let s_var = circuit.create_variable(fr_to_fq::<F, P>(&s))?;
            let base_var = circuit.create_point_variable(&Point::from(base))?;
            base = (base * s).into();
            let result = circuit.variable_base_scalar_mul::<P>(s_var, &base_var)?;
            assert_eq!(
                Point::from(base),
                Point::TE(
                    circuit.witness(result.get_x())?,
                    circuit.witness(result.get_y())?
                )
            );
        }
        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());

        let base = SWAffine::<P>::rand(&mut rng);
        let s = P::ScalarField::rand(&mut rng);
        let s_var = circuit.create_variable(fr_to_fq::<F, P>(&s))?;
        let base_var = circuit.create_point_variable(&Point::from(base))?;
        // wrong witness should fail
        *circuit.witness_mut(2) = F::rand(&mut rng);
        assert!(circuit.check_circuit_satisfiability(&[]).is_err());
        // Check variable out of bound error.
        assert!(circuit
            .variable_base_scalar_mul::<P>(circuit.num_vars(), &base_var)
            .is_err());
        assert!(circuit
            .variable_base_scalar_mul::<P>(
                s_var,
                &PointVariable::TE(circuit.num_vars(), circuit.num_vars())
            )
            .is_err());

        let circuit_1 =
            build_variable_base_scalar_mul_circuit::<F, P>(F::zero(), Point::from(base))?;
        let circuit_2 = build_variable_base_scalar_mul_circuit::<F, P>(
            F::from(314u32),
            Point::from(SWAffine::<P>::rand(&mut rng)),
        )?;
        test_variable_independence_for_circuit(circuit_1, circuit_2)?;

        Ok(())
    }

    /*fn test_variable_base_scalar_mul_helper_mle<F, P>() -> Result<(), CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
    {
        let mut rng = jf_utils::test_rng();
        let mut circuit: PlonkCircuit<F> = PlonkCircuit::new_turbo_plonk();

        for _ in 0..6 {
            let mut base = SWAffine::<P>::rand(&mut rng);
            let s = P::ScalarField::rand(&mut rng);
            let s_var =
                circuit.create_variable(field_switching::<P::ScalarField, P::BaseField>(&s))?;

            let base_var = circuit.create_point_variable(&Point::from(base))?;
            base = (base * s).into_affine();

            let result = circuit
                .variable_base_scalar_mul::<P>(s_var, &base_var)
                .unwrap();
            assert_eq!(base.x, circuit.witness(result.get_x())?);
            assert_eq!(base.y, circuit.witness(result.get_y())?);
        }
        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());

        let base = SWAffine::<P>::rand(&mut rng);
        let s = P::ScalarField::rand(&mut rng);
        let s_var = circuit.create_variable(fr_to_fq::<F, P>(&s))?;
        let base_var = circuit.create_point_variable(&Point::from(base))?;
        // wrong witness should fail
        *circuit.witness_mut(2) = F::rand(&mut rng);
        assert!(circuit.check_circuit_satisfiability(&[]).is_err());
        // Check variable out of bound error.
        assert!(circuit
            .variable_base_scalar_mul::<P>(circuit.num_vars(), &base_var)
            .is_err());
        assert!(circuit
            .variable_base_scalar_mul::<P>(
                s_var,
                &PointVariable::SW(circuit.num_vars(), circuit.num_vars())
            )
            .is_err());

        Ok(())
    }*/

    fn build_variable_base_scalar_mul_circuit<F, P>(
        scalar: F,
        base: Point<F>,
    ) -> Result<PlonkCircuit<F>, CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
    {
        let mut circuit: PlonkCircuit<F> = PlonkCircuit::new_turbo_plonk();
        let scalar_var = circuit.create_variable(scalar)?;
        let base_var = circuit.create_point_variable(&base)?;
        circuit.variable_base_scalar_mul::<P>(scalar_var, &base_var)?;
        circuit.finalize_for_arithmetization()?;
        Ok(circuit)
    }

    #[test]
    fn test_point_double() {
        test_point_double_helper::<FqGrump, SWGrumpkin>().unwrap();
    }

    fn test_point_double_helper<F, P>() -> Result<(), CircuitError>
    where
        F: PrimeField,
        P: HasTEForm<BaseField = F>,
    {
        let mut rng = jf_utils::test_rng();
        let p = SWAffine::<P>::rand(&mut rng);
        let zero_point = SWAffine::<P>::zero();

        let mut circuit = PlonkCircuit::<F>::new_turbo_plonk();
        let p_var = circuit.create_point_variable(&Point::from(p))?;
        let p2_var = circuit.ecc_double::<P>(&p_var)?;
        let double_p = (p + p).into_affine();
        assert_eq!(double_p.x, circuit.witness(p2_var.get_x())?,);
        assert_eq!(double_p.y, circuit.witness(p2_var.get_y())?,);
        circuit.check_circuit_satisfiability(&[]).unwrap();
        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
        *circuit.witness_mut(p2_var.get_x()) = F::zero();
        assert!(circuit.check_circuit_satisfiability(&[]).is_err());

        let mut circuit = PlonkCircuit::<F>::new_turbo_plonk();
        let p_var = circuit.create_point_variable(&Point::from(zero_point))?;
        let p2_var = circuit.ecc_double::<P>(&p_var)?;

        assert_eq!(F::zero(), circuit.witness(p2_var.get_x())?,);
        assert_eq!(F::one(), circuit.witness(p2_var.get_y())?,);
        circuit.check_circuit_satisfiability(&[]).unwrap();
        assert!(circuit.check_circuit_satisfiability(&[]).is_ok());

        Ok(())
    }
}
