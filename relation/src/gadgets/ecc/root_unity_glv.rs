use crate::{
    errors::CircuitError,
    gadgets::{
        ecc::{Point, PointVariable},
        EmulatedVariable, EmulationConfig,
    },
    BoolVar, Circuit, PlonkCircuit, Variable,
};

use ark_ff::PrimeField;
use ark_std::string::ToString;
use num_bigint::BigUint;

use super::{HasTEForm, MultiScalarMultiplicationCircuit};

impl<F: PrimeField> PlonkCircuit<F> {
    /// Decomposes a scalar into two scalars using the GLV method.
    pub fn scalar_decomposition_gate<E>(
        &mut self,
        k_var: &EmulatedVariable<E::ScalarField>,
    ) -> Result<(Variable, Variable, BoolVar), CircuitError>
    where
        E: HasTEForm<BaseField = F>,
        E::ScalarField: EmulationConfig<F>,
    {
        if !E::has_glv() {
            return Err(CircuitError::ParameterError(
                "GLV is not supported for this curve".to_string(),
            ));
        }
        let k = self.emulated_witness(k_var)?;
        let ((k1_sign, k1), (k2_sign, k2)) = E::scalar_decomposition(k);
        if !k1_sign {
            return Err(CircuitError::ParameterError(
                "k1 should be positive".to_string(),
            ));
        }
        let k2_with_sign = if k2_sign { k2 } else { -k2 };
        let k1_var = self.create_emulated_variable(k1)?;
        let k2_var = self.create_emulated_variable(k2_with_sign)?;
        let k2_sign_var = self.create_boolean_variable(k2_sign)?;
        let lambda_k2 = self.emulated_mul_constant(&k2_var, E::LAMBDA)?;

        // We need to ensure that `k1_var` and `k2_var` represent values less than the modulus of the base field.
        // Then, when we apply `mod_to_native_field`, we will get the true values in the base field.
        // We do this crudely by just demanding that the appropriate number of top limbs are zero.
        // Since they should both actually represent elements around the square root of the modulus scalar field, this should be fine.
        let base_modulus: BigUint = F::MODULUS.into();
        let scalar_modulus: BigUint = E::ScalarField::MODULUS.into();
        if base_modulus < scalar_modulus {
            let mut current_limb = E::ScalarField::NUM_LIMBS;
            while (BigUint::from(1u8) << (current_limb * E::ScalarField::B)) > base_modulus {
                current_limb -= 1;
                self.enforce_constant(k1_var.0[current_limb], F::zero())?;
                self.enforce_constant(k2_var.0[current_limb], F::zero())?;
            }
        }

        self.emulated_add_gate(&k1_var, &lambda_k2, k_var)?;

        let k1_var = self.mod_to_native_field(&k1_var)?;

        let k2_var = self.mod_to_native_field(&k2_var)?;

        Ok((k1_var, k2_var, k2_sign_var))
    }

    /// Applies the endomorphism to a point.
    pub fn endomorphism_circuit<E>(
        &mut self,
        point_var: &PointVariable,
    ) -> Result<PointVariable, CircuitError>
    where
        E: HasTEForm<BaseField = F>,
        E::ScalarField: EmulationConfig<F>,
    {
        if !E::has_glv() {
            return Err(CircuitError::ParameterError(
                "GLV is not supported for this curve".to_string(),
            ));
        }
        let point_wrapper = self.point_witness(point_var)?;
        if !matches!(point_wrapper, Point::SW(..)) {
            return Err(CircuitError::ParameterError("Expected SWPoint".to_string()));
        }
        let x_var = self.mul_constant(point_var.get_x(), &E::ENDO_COEFFS[0])?;
        Ok(PointVariable::SW(x_var, point_var.get_y()))
    }

    /*/// Multiplies a point by a scalar using the GLV method.
    pub fn glv_mul<E>(
        &mut self,
        k_var: &EmulatedVariable<E::ScalarField>,
        base_var: &PointVariable,
    ) -> Result<PointVariable, CircuitError>
    where
        E: HasTEForm<BaseField = F>,
        E::ScalarField: EmulationConfig<F>,
    {
        if !E::has_glv() {
            return Err(CircuitError::ParameterError(
                "GLV is not supported for this curve".to_string(),
            ));
        }
        let (k1_var, k2_var, k2_sign_var) = self.scalar_decomposition_gate::<E>(k_var)?;

        let endo_base_var = self.endomorphism_circuit::<E>(base_var)?;
        let endo_base_neg_var = self.inverse_point(&endo_base_var)?;
        let endo_base_var =
            self.binary_point_vars_select(k2_sign_var, &endo_base_neg_var, &endo_base_var)?;

        MultiScalarMultiplicationCircuit::<_, E>::msm_with_var_scalar_length(
            self,
            &[*base_var, endo_base_var],
            &[k1_var, k2_var],
            128,
        )
    }*/

    /*/// This function can be used when adding two GLV points together.
    pub fn glv_ecc_add<E>(
        &mut self,
        point_1: &PointVariable,
        point_2: &PointVariable,
    ) -> Result<PointVariable, CircuitError>
    where
        E: HasTEForm<BaseField = F>,
        E::ScalarField: EmulationConfig<F>,
    {
        if !E::has_glv() {
            return Err(CircuitError::ParameterError(
                "GLV is not supported for this curve".to_string(),
            ));
        }

        // Since point_2 is the endomorphism applied to point_1 the lambda in point addition is zero.
        let indicator = self.is_neutral_point::<E>(point_1)?;

        let x_out = self.lc(
            &[point_1.get_x(), point_2.get_x(), self.zero(), self.zero()],
            &[-F::one(), -F::one(), F::zero(), F::zero()],
        )?;
        let y_out = self.sub(self.zero(), point_1.get_y())?;

        let point_var = PointVariable::SW(x_out, y_out);

        self.binary_point_vars_select(indicator, &point_var, point_1)
    }*/
}

#[cfg(test)]
/// return the highest non-zero bits of a bit string.
fn get_bits(a: &[bool]) -> u16 {
    let mut res = 256;
    for e in a.iter().rev() {
        if !e {
            res -= 1;
        } else {
            return res;
        }
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        constraint_system::Circuit,
        gadgets::{ecc::HasTEForm, EmulationConfig},
    };
    use ark_ec::CurveGroup;
    use ark_ff::{BigInteger, PrimeField};
    use ark_std::test_rng;
    use nf_curves::grumpkin::short_weierstrass::SWGrumpkin;
    use num_bigint::BigUint;

    #[test]
    fn test_decomposition() -> Result<(), CircuitError> {
        test_decomposition_helper::<_, ark_bn254::g1::Config>()?;
        test_decomposition_helper::<_, SWGrumpkin>()
    }

    fn test_decomposition_helper<E, P>() -> Result<(), CircuitError>
    where
        E: EmulationConfig<P::BaseField>,
        P: HasTEForm<ScalarField = E>,
        P::BaseField: PrimeField,
    {
        let mut rng = test_rng();
        let lambda = P::LAMBDA;
        for _ in 0..100 {
            let scalar = P::ScalarField::rand(&mut rng);
            let ((k1_sign, k1), (k2_sign, k2)) = P::scalar_decomposition(scalar);
            assert!(get_bits(&k1.into_bigint().to_bits_le()) <= 128);
            assert!(k1_sign);
            let k2 = if k2_sign { k2 } else { -k2 };
            assert!(get_bits(&k2.into_bigint().to_bits_le()) <= 128);
            assert_eq!(k1 + k2 * lambda, scalar);
            let mut circuit: PlonkCircuit<P::BaseField> = PlonkCircuit::new_ultra_plonk(16);
            let scalar_var = circuit.create_emulated_variable(scalar)?;
            let (k1_var, k2_var, _k2_sign_var) =
                circuit.scalar_decomposition_gate::<P>(&scalar_var)?;
            assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
            let k1_biguint: BigUint = k1.into();
            let k1_var_biguint: BigUint = circuit.witness(k1_var)?.into();
            assert_eq!(k1_biguint, k1_var_biguint);
            let k2_biguint: BigUint = k2.into();
            let k2_var_biguint: BigUint = circuit.witness(k2_var)?.into();
            assert_eq!(k2_biguint, k2_var_biguint);
        }
        Ok(())
    }

    #[test]
    fn test_endomorphism() -> Result<(), CircuitError> {
        test_enodmorphism_helper::<_, ark_bn254::g1::Config>()?;
        test_enodmorphism_helper::<_, SWGrumpkin>()
    }

    fn test_enodmorphism_helper<E, P>() -> Result<(), CircuitError>
    where
        E: EmulationConfig<P::BaseField>,
        P: HasTEForm<ScalarField = E>,
        P::BaseField: PrimeField,
    {
        let mut rng = test_rng();
        let lambda = P::LAMBDA;
        let gen = P::GENERATOR;
        for _ in 0..100 {
            // Generate random point on curve and its image under the endomorphism
            let scalar = P::ScalarField::rand(&mut rng);
            let affine_point = (gen * scalar).into_affine();
            let affine_point_image = P::endomorphism_affine(&affine_point);

            // Check endomorphism
            assert_eq!(affine_point_image, affine_point * lambda);

            let point: Point<P::BaseField> = affine_point.into();
            let point_image: Point<P::BaseField> = affine_point_image.into();

            // Do the endomorphism in the circuit
            let mut circuit: PlonkCircuit<P::BaseField> = PlonkCircuit::new_ultra_plonk(16);
            let point_var = circuit.create_point_variable(&point)?;
            let point_image_var = circuit.endomorphism_circuit::<P>(&point_var)?;

            // Check circuit
            assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
            assert_eq!(point_image, circuit.point_witness(&point_image_var)?);
        }
        Ok(())
    }

    /*#[test]
    #[ignore]
    fn test_glv_mul() -> Result<(), CircuitError> {
        test_glv_mul_helper::<_, ark_bn254::g1::Config>()?;
        test_glv_mul_helper::<_, SWGrumpkin>()
    }

    fn test_glv_mul_helper<E, P>() -> Result<(), CircuitError>
    where
        E: EmulationConfig<P::BaseField>,
        P: HasTEForm<ScalarField = E>,
        P::BaseField: PrimeField,
    {
        let mut rng = test_rng();
        let gen = P::GENERATOR;
        let base: Point<P::BaseField> = gen.into();
        for _ in 0..100 {
            let scalar = P::ScalarField::rand(&mut rng) + P::ScalarField::one();
            let mut circuit: PlonkCircuit<P::BaseField> = PlonkCircuit::new_ultra_plonk(16);
            let base_var = circuit.create_point_variable(&base)?;
            let scalar_var = circuit.create_emulated_variable(scalar)?;
            let gen_times_scalar_var = circuit.glv_mul::<P>(&scalar_var, &base_var)?;
            let gen_times_scalar = circuit.point_witness(&gen_times_scalar_var)?;

            assert!(circuit.check_circuit_satisfiability(&[]).is_ok());
            assert_eq!(gen_times_scalar, (gen * scalar).into_affine().into());
        }
        Ok(())
    }*/
}
