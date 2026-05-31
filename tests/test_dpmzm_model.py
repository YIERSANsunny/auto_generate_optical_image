from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.special import jv

from dpmzm_model import (
    ARM_VIEW_ORDER,
    DPMZMParams,
    PUSH_PULL_ARM_RF_PHASE_DIFF_DEG,
    SpectralLine,
    VIEW_COUPLED,
    VIEW_I,
    VIEW_I_LOWER,
    VIEW_I_TOTAL,
    VIEW_I_UPPER,
    VIEW_ORDER,
    VIEW_Q,
    VIEW_Q_AFTER_P,
    child_mzm_coefficients,
    child_mzm_components,
    child_mzm_components_nonideal,
    extinction_ratio_delta,
    field_loss_factor,
    phase_to_display_angle_deg,
    simulate_arm_spectra,
    simulate_spectra,
    write_spectra_csv,
)
from app import (
    ARROW_BASE_LENGTH_PX,
    arrow_display_delta_px,
    format_hover_text,
    is_visible_sideband,
    phase_deg_to_voltage,
    phase_to_color,
    vector_arrow_length_px,
    voltage_to_phase_deg,
)


class DPMZMModelTests(unittest.TestCase):
    def test_no_rf_voltage_keeps_only_carrier(self) -> None:
        params = DPMZMParams(
            rf_amplitude_i_v=0.0,
            rf_amplitude_q_v=0.0,
            sideband_order=4,
        )

        spectra = simulate_spectra(params)

        for lines in spectra.values():
            for line in lines:
                if line.order == 0:
                    self.assertGreater(line.magnitude, 1e-9)
                else:
                    self.assertLess(line.magnitude, 1e-12)

    def test_child_mzm_matches_bessel_expansion_for_orders(self) -> None:
        vpi = 4.0
        bias_voltage = 1.1
        rf_peak_voltage = 0.7
        rf_relative_phase_deg = 35.0

        coefficients = child_mzm_coefficients(
            bias_voltage=bias_voltage,
            vpi=vpi,
            rf_peak_voltage=rf_peak_voltage,
            sideband_order=2,
            rf_relative_phase_deg=rf_relative_phase_deg,
        )

        for order in [-2, -1, 0, 1, 2]:
            expected = _manual_bessel_coefficient(
                order=order,
                bias_voltage=bias_voltage,
                vpi=vpi,
                rf_peak_voltage=rf_peak_voltage,
                rf_relative_phase_deg=rf_relative_phase_deg,
            )
            self.assertAlmostEqual(coefficients[order].real, expected.real, places=12)
            self.assertAlmostEqual(coefficients[order].imag, expected.imag, places=12)

    def test_child_mzm_components_match_sampled_time_domain_push_pull(self) -> None:
        vpi = 4.0
        bias_voltage = 1.1
        rf_peak_voltage = 0.7
        rf_relative_phase_deg = 35.0
        sideband_order = 3

        components = child_mzm_components(
            bias_voltage=bias_voltage,
            vpi=vpi,
            rf_peak_voltage=rf_peak_voltage,
            sideband_order=sideband_order,
            rf_relative_phase_deg=rf_relative_phase_deg,
        )
        sampled = _sampled_child_mzm_components(
            bias_voltage=bias_voltage,
            vpi=vpi,
            rf_peak_voltage=rf_peak_voltage,
            sideband_order=sideband_order,
            rf_relative_phase_deg=rf_relative_phase_deg,
        )

        for order in range(-sideband_order, sideband_order + 1):
            for attr in ("upper", "lower", "total"):
                expected = sampled[attr][order]
                actual = getattr(components, attr)[order]
                self.assertAlmostEqual(actual.real, expected.real, places=11)
                self.assertAlmostEqual(actual.imag, expected.imag, places=11)

    def test_child_mzm_components_sum_to_total(self) -> None:
        components = child_mzm_components(
            bias_voltage=1.1,
            vpi=4.0,
            rf_peak_voltage=0.7,
            sideband_order=2,
            rf_relative_phase_deg=35.0,
        )

        for order in [-2, -1, 0, 1, 2]:
            expected = components.upper[order] + components.lower[order]
            self.assertAlmostEqual(components.total[order].real, expected.real, places=12)
            self.assertAlmostEqual(components.total[order].imag, expected.imag, places=12)

    def test_child_mzm_coefficients_wrap_component_total(self) -> None:
        kwargs = {
            "bias_voltage": 0.9,
            "vpi": 4.0,
            "rf_peak_voltage": 0.7,
            "sideband_order": 3,
            "rf_relative_phase_deg": 30.0,
        }
        components = child_mzm_components(**kwargs)
        coefficients = child_mzm_coefficients(**kwargs)

        for order in range(-3, 4):
            self.assertAlmostEqual(coefficients[order].real, components.total[order].real, places=12)
            self.assertAlmostEqual(coefficients[order].imag, components.total[order].imag, places=12)

    def test_negative_sideband_signs_follow_bessel_expansion(self) -> None:
        coefficients = child_mzm_coefficients(
            bias_voltage=1.0,
            vpi=4.0,
            rf_peak_voltage=0.6,
            sideband_order=2,
        )

        self.assertAlmostEqual(coefficients[-1].real, -coefficients[1].real, places=12)
        self.assertAlmostEqual(coefficients[-1].imag, -coefficients[1].imag, places=12)
        self.assertAlmostEqual(coefficients[-2].real, coefficients[2].real, places=12)
        self.assertAlmostEqual(coefficients[-2].imag, coefficients[2].imag, places=12)

    def test_zero_bias_push_pull_cancels_odd_sidebands(self) -> None:
        components = child_mzm_components(
            bias_voltage=0.0,
            vpi=4.0,
            rf_peak_voltage=0.8,
            sideband_order=5,
        )

        for order in [-5, -3, -1, 1, 3, 5]:
            self.assertGreater(abs(components.upper[order]), 1e-12)
            self.assertGreater(abs(components.lower[order]), 1e-12)
            self.assertLess(abs(components.total[order]), 1e-12)
        for order in [-4, -2, 0, 2, 4]:
            self.assertGreater(abs(components.total[order]), 1e-12)

    def test_q_rf_relative_phase_rotates_each_sideband_by_order(self) -> None:
        base = child_mzm_components(
            bias_voltage=0.9,
            vpi=4.0,
            rf_peak_voltage=0.7,
            sideband_order=3,
            rf_relative_phase_deg=0.0,
        )
        shifted = child_mzm_components(
            bias_voltage=0.9,
            vpi=4.0,
            rf_peak_voltage=0.7,
            sideband_order=3,
            rf_relative_phase_deg=30.0,
        )

        psi = math.radians(30.0)
        for order in [-3, -2, -1, 0, 1, 2, 3]:
            rotation = np.exp(1j * order * psi)
            for attr in ("upper", "lower", "total"):
                expected = getattr(base, attr)[order] * rotation
                actual = getattr(shifted, attr)[order]
                self.assertAlmostEqual(actual.real, expected.real, places=12)
                self.assertAlmostEqual(actual.imag, expected.imag, places=12)

    def test_p_bias_shift_by_one_vpi_rotates_q_after_p_by_180_degrees(self) -> None:
        params = DPMZMParams(sideband_order=4)
        base = simulate_spectra(params)
        shifted = simulate_spectra(
            DPMZMParams(voltage_p=params.vpi_p, vpi_p=params.vpi_p, sideband_order=4)
        )

        base_by_order = {line.order: line for line in base[VIEW_Q_AFTER_P]}
        shifted_by_order = {line.order: line for line in shifted[VIEW_Q_AFTER_P]}

        for order, base_line in base_by_order.items():
            if base_line.magnitude < 1e-8:
                continue
            phase_delta = _phase_delta_deg(
                base_line.phase_deg,
                shifted_by_order[order].phase_deg,
            )
            self.assertAlmostEqual(abs(phase_delta), 180.0, places=6)

    def test_single_tone_rf_generates_symmetric_i_sidebands(self) -> None:
        spectra = simulate_spectra(DPMZMParams(sideband_order=5))
        by_order = {line.order: line for line in spectra[VIEW_I]}

        for order in range(1, 6):
            self.assertAlmostEqual(
                by_order[order].magnitude,
                by_order[-order].magnitude,
                delta=1e-12,
            )

    def test_coupled_output_matches_i_plus_shifted_q(self) -> None:
        spectra = simulate_spectra(DPMZMParams(voltage_i=0.8, voltage_q=-0.3, voltage_p=1.2))
        i_by_order = {line.order: line for line in spectra[VIEW_I]}
        q_after_p_by_order = {line.order: line for line in spectra[VIEW_Q_AFTER_P]}
        coupled_by_order = {line.order: line for line in spectra[VIEW_COUPLED]}

        for order, coupled_line in coupled_by_order.items():
            expected = (
                complex(i_by_order[order].real, i_by_order[order].imag)
                + complex(q_after_p_by_order[order].real, q_after_p_by_order[order].imag)
            ) / math.sqrt(2.0)
            actual = complex(coupled_line.real, coupled_line.imag)
            self.assertAlmostEqual(actual.real, expected.real, places=12)
            self.assertAlmostEqual(actual.imag, expected.imag, places=12)

    def test_power_fields_are_derived_after_complex_field_calculation(self) -> None:
        spectra = simulate_spectra(DPMZMParams(voltage_i=0.8, voltage_q=-0.3, voltage_p=1.2))
        lines = [line for view_lines in spectra.values() for line in view_lines]
        max_power = max(line.power for line in lines)

        for line in lines:
            expected_power = line.real**2 + line.imag**2
            self.assertAlmostEqual(line.power, expected_power, places=12)
            self.assertAlmostEqual(line.power, line.magnitude**2, places=12)
            self.assertAlmostEqual(line.power_db, line.magnitude_db, places=12)
            if line.power > 1e-30:
                self.assertAlmostEqual(
                    line.power_db,
                    10.0 * math.log10(line.power / max_power),
                    places=12,
                )

    def test_coupled_phase_is_derived_from_complex_output_field(self) -> None:
        spectra = simulate_spectra(DPMZMParams(voltage_i=0.8, voltage_q=-0.3, voltage_p=1.2))

        for line in spectra[VIEW_COUPLED]:
            expected_phase = math.degrees(math.atan2(line.imag, line.real))
            self.assertAlmostEqual(
                _phase_delta_deg(line.phase_deg, expected_phase),
                0.0,
                places=12,
            )

    def test_ideal_model_ignores_nonideal_parameter_values(self) -> None:
        base = DPMZMParams(voltage_i=0.8, voltage_q=-0.3, voltage_p=1.2, sideband_order=3)
        disabled_nonideal = DPMZMParams(
            voltage_i=0.8,
            voltage_q=-0.3,
            voltage_p=1.2,
            sideband_order=3,
            use_nonideal=False,
            extinction_ratio_i_db=5.0,
            extinction_ratio_q_db=8.0,
            insertion_loss_i_db=11.0,
            insertion_loss_q_db=12.0,
            insertion_loss_p_db=13.0,
            insertion_loss_global_db=14.0,
        )

        base_spectra = simulate_spectra(base)
        disabled_spectra = simulate_spectra(disabled_nonideal)

        for view in VIEW_ORDER:
            for base_line, disabled_line in zip(base_spectra[view], disabled_spectra[view]):
                self.assertEqual(base_line.order, disabled_line.order)
                self.assertAlmostEqual(base_line.real, disabled_line.real, places=12)
                self.assertAlmostEqual(base_line.imag, disabled_line.imag, places=12)

    def test_nonideal_with_no_loss_and_high_er_approaches_ideal(self) -> None:
        ideal = DPMZMParams(voltage_i=0.8, voltage_q=-0.3, voltage_p=1.2, sideband_order=3)
        nearly_ideal = DPMZMParams(
            voltage_i=0.8,
            voltage_q=-0.3,
            voltage_p=1.2,
            sideband_order=3,
            use_nonideal=True,
            extinction_ratio_i_db=120.0,
            extinction_ratio_q_db=120.0,
            insertion_loss_i_db=0.0,
            insertion_loss_q_db=0.0,
            insertion_loss_p_db=0.0,
            insertion_loss_global_db=0.0,
        )

        ideal_spectra = simulate_spectra(ideal)
        nearly_ideal_spectra = simulate_spectra(nearly_ideal)

        for view in VIEW_ORDER:
            for ideal_line, nonideal_line in zip(ideal_spectra[view], nearly_ideal_spectra[view]):
                self.assertEqual(ideal_line.order, nonideal_line.order)
                self.assertAlmostEqual(ideal_line.real, nonideal_line.real, places=5)
                self.assertAlmostEqual(ideal_line.imag, nonideal_line.imag, places=5)

    def test_nonideal_helpers_match_vpi_er_and_loss_definitions(self) -> None:
        self.assertAlmostEqual(field_loss_factor(6.0), 10 ** (-6.0 / 20.0))
        self.assertAlmostEqual(extinction_ratio_delta(30.0), 1.0 / (10 ** (30.0 / 20.0) - 1.0))

    def test_nonideal_child_max_transmission_field_equals_sqrt_loss(self) -> None:
        loss_db = 6.0
        components = child_mzm_components_nonideal(
            bias_voltage=0.0,
            vpi=5.0,
            rf_peak_voltage=0.0,
            sideband_order=0,
            insertion_loss_db=loss_db,
            extinction_ratio_db=30.0,
        )

        self.assertAlmostEqual(abs(components.total[0]), field_loss_factor(loss_db), places=12)

    def test_nonideal_child_null_residual_matches_extinction_ratio(self) -> None:
        loss_db = 6.0
        er_db = 30.0
        components = child_mzm_components_nonideal(
            bias_voltage=5.0,
            vpi=5.0,
            rf_peak_voltage=0.0,
            sideband_order=0,
            insertion_loss_db=loss_db,
            extinction_ratio_db=er_db,
        )

        expected = field_loss_factor(loss_db) * 10 ** (-er_db / 20.0)
        self.assertAlmostEqual(abs(components.total[0]), expected, places=12)

    def test_nonideal_p_loss_scales_only_q_after_p_view(self) -> None:
        common = {
            "use_nonideal": True,
            "voltage_i": 0.8,
            "voltage_q": -0.3,
            "voltage_p": 1.2,
            "sideband_order": 3,
            "extinction_ratio_i_db": 120.0,
            "extinction_ratio_q_db": 120.0,
            "insertion_loss_i_db": 0.0,
            "insertion_loss_q_db": 0.0,
            "insertion_loss_global_db": 0.0,
        }
        no_p_loss = simulate_spectra(DPMZMParams(**common, insertion_loss_p_db=0.0))
        with_p_loss = simulate_spectra(DPMZMParams(**common, insertion_loss_p_db=6.0))
        p_factor = field_loss_factor(6.0)

        for no_loss_q, with_loss_q in zip(no_p_loss[VIEW_Q], with_p_loss[VIEW_Q]):
            self.assertEqual(no_loss_q.order, with_loss_q.order)
            self.assertAlmostEqual(no_loss_q.real, with_loss_q.real, places=12)
            self.assertAlmostEqual(no_loss_q.imag, with_loss_q.imag, places=12)

        for no_loss_qp, with_loss_qp in zip(no_p_loss[VIEW_Q_AFTER_P], with_p_loss[VIEW_Q_AFTER_P]):
            self.assertEqual(no_loss_qp.order, with_loss_qp.order)
            expected = complex(no_loss_qp.real, no_loss_qp.imag) * p_factor
            actual = complex(with_loss_qp.real, with_loss_qp.imag)
            self.assertAlmostEqual(actual.real, expected.real, places=12)
            self.assertAlmostEqual(actual.imag, expected.imag, places=12)

    def test_nonideal_arm_spectra_components_sum_to_total(self) -> None:
        params = DPMZMParams(
            use_nonideal=True,
            voltage_i=0.8,
            voltage_q=-0.3,
            sideband_order=3,
        )
        spectra = simulate_arm_spectra(params)
        upper_by_order = {line.order: line for line in spectra[VIEW_I_UPPER]}
        lower_by_order = {line.order: line for line in spectra[VIEW_I_LOWER]}
        total_by_order = {line.order: line for line in spectra[VIEW_I_TOTAL]}

        for order, total_line in total_by_order.items():
            expected = (
                complex(upper_by_order[order].real, upper_by_order[order].imag)
                + complex(lower_by_order[order].real, lower_by_order[order].imag)
            )
            actual = complex(total_line.real, total_line.imag)
            self.assertAlmostEqual(actual.real, expected.real, places=12)
            self.assertAlmostEqual(actual.imag, expected.imag, places=12)

    def test_nonideal_coupled_output_matches_complex_field_sum_with_global_loss(self) -> None:
        params = DPMZMParams(
            use_nonideal=True,
            voltage_i=0.8,
            voltage_q=-0.3,
            voltage_p=1.2,
            insertion_loss_global_db=3.0,
            sideband_order=3,
        )
        spectra = simulate_spectra(params)
        i_by_order = {line.order: line for line in spectra[VIEW_I]}
        q_after_p_by_order = {line.order: line for line in spectra[VIEW_Q_AFTER_P]}
        coupled_by_order = {line.order: line for line in spectra[VIEW_COUPLED]}
        global_factor = field_loss_factor(params.insertion_loss_global_db)

        for order, coupled_line in coupled_by_order.items():
            expected = (
                global_factor
                * (
                    complex(i_by_order[order].real, i_by_order[order].imag)
                    + complex(q_after_p_by_order[order].real, q_after_p_by_order[order].imag)
                )
                / math.sqrt(2.0)
            )
            actual = complex(coupled_line.real, coupled_line.imag)
            self.assertAlmostEqual(actual.real, expected.real, places=12)
            self.assertAlmostEqual(actual.imag, expected.imag, places=12)

    def test_bias_voltage_phase_conversion_helpers(self) -> None:
        self.assertAlmostEqual(voltage_to_phase_deg(3.0, 5.0), 108.0)
        self.assertAlmostEqual(phase_deg_to_voltage(108.0, 5.0), 3.0)

    def test_phase_bias_inputs_match_equivalent_voltage_spectra(self) -> None:
        vpi_i = 5.0
        vpi_q = 6.0
        vpi_p = 4.0
        phase_i = 108.0
        phase_q = -45.0
        phase_p = 180.0
        phase_equivalent = DPMZMParams(
            voltage_i=phase_deg_to_voltage(phase_i, vpi_i),
            voltage_q=phase_deg_to_voltage(phase_q, vpi_q),
            voltage_p=phase_deg_to_voltage(phase_p, vpi_p),
            vpi_i=vpi_i,
            vpi_q=vpi_q,
            vpi_p=vpi_p,
            sideband_order=3,
        )
        voltage_equivalent = DPMZMParams(
            voltage_i=3.0,
            voltage_q=-1.5,
            voltage_p=4.0,
            vpi_i=vpi_i,
            vpi_q=vpi_q,
            vpi_p=vpi_p,
            sideband_order=3,
        )

        phase_spectra = simulate_spectra(phase_equivalent)
        voltage_spectra = simulate_spectra(voltage_equivalent)

        for view in VIEW_ORDER:
            for phase_line, voltage_line in zip(phase_spectra[view], voltage_spectra[view]):
                self.assertEqual(phase_line.order, voltage_line.order)
                self.assertAlmostEqual(phase_line.real, voltage_line.real, places=12)
                self.assertAlmostEqual(phase_line.imag, voltage_line.imag, places=12)
                self.assertAlmostEqual(phase_line.power, voltage_line.power, places=12)

    def test_p_phase_180_matches_p_bias_of_one_vpi(self) -> None:
        phase_params = DPMZMParams(
            voltage_p=phase_deg_to_voltage(180.0, 5.0),
            vpi_p=5.0,
            sideband_order=3,
        )
        voltage_params = DPMZMParams(voltage_p=5.0, vpi_p=5.0, sideband_order=3)

        phase_lines = simulate_spectra(phase_params)[VIEW_Q_AFTER_P]
        voltage_lines = simulate_spectra(voltage_params)[VIEW_Q_AFTER_P]

        for phase_line, voltage_line in zip(phase_lines, voltage_lines):
            self.assertEqual(phase_line.order, voltage_line.order)
            self.assertAlmostEqual(phase_line.real, voltage_line.real, places=12)
            self.assertAlmostEqual(phase_line.imag, voltage_line.imag, places=12)

    def test_phase_to_display_angle_mapping(self) -> None:
        self.assertEqual(phase_to_display_angle_deg(0.0), 90.0)
        self.assertEqual(phase_to_display_angle_deg(0.5), 90.0)
        self.assertEqual(phase_to_display_angle_deg(90.0), 90.0)
        self.assertEqual(phase_to_display_angle_deg(-90.0), -90.0)
        self.assertEqual(phase_to_display_angle_deg(180.0), -90.0)
        self.assertEqual(phase_to_display_angle_deg(179.5), -90.0)
        self.assertEqual(phase_to_display_angle_deg(-180.0), -90.0)
        self.assertEqual(phase_to_display_angle_deg(45.0), 45.0)
        self.assertEqual(phase_to_display_angle_deg(-45.0), -45.0)

    def test_phase_color_zero_is_high_contrast_not_white(self) -> None:
        red, green, blue, _alpha = phase_to_color(0.0)

        self.assertLess(_relative_luminance(red, green, blue), 0.35)
        self.assertFalse(red > 0.85 and green > 0.85 and blue > 0.85)
        self.assertGreater(green, red)
        self.assertGreater(green, blue)

    def test_phase_color_wraps_at_plus_minus_180(self) -> None:
        negative = phase_to_color(-180.0)
        positive = phase_to_color(180.0)

        for negative_component, positive_component in zip(negative, positive):
            self.assertAlmostEqual(negative_component, positive_component, places=12)

    def test_hover_text_shows_only_frequency_phase_and_power(self) -> None:
        text = format_hover_text(_line_with_db(-20.0))

        self.assertEqual(len(text.splitlines()), 3)
        self.assertIn("频偏", text)
        self.assertIn("相位", text)
        self.assertIn("功率", text)
        self.assertIn("dB", text)
        self.assertNotIn("场幅", text)
        self.assertNotIn("复数", text)
        self.assertNotIn("边带阶数", text)

    def test_csv_export_contains_all_views_orders_and_true_phase_only(self) -> None:
        params = DPMZMParams(sideband_order=2)
        spectra = simulate_spectra(params)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "spectra.csv"
            write_spectra_csv(spectra, path)
            with path.open(newline="", encoding="utf-8-sig") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(len(rows), len(VIEW_ORDER) * (2 * params.sideband_order + 1))
        self.assertEqual({row["view"] for row in rows}, set(VIEW_ORDER))
        self.assertEqual(
            sorted({int(row["order"]) for row in rows}),
            [-2, -1, 0, 1, 2],
        )
        self.assertIn("phase_deg", rows[0])
        self.assertIn("power", rows[0])
        self.assertIn("power_db", rows[0])
        self.assertNotIn("arrow_angle_deg", rows[0])

    def test_arm_spectra_contains_iq_arms_and_totals(self) -> None:
        params = DPMZMParams(sideband_order=2)
        spectra = simulate_arm_spectra(params)

        self.assertEqual(set(spectra), set(ARM_VIEW_ORDER))
        for lines in spectra.values():
            self.assertEqual([line.order for line in lines], [-2, -1, 0, 1, 2])

    def test_arm_spectra_csv_exports_arm_views(self) -> None:
        params = DPMZMParams(sideband_order=1)
        spectra = simulate_arm_spectra(params)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "arm_spectra.csv"
            write_spectra_csv(spectra, path)
            with path.open(newline="", encoding="utf-8-sig") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(len(rows), len(ARM_VIEW_ORDER) * (2 * params.sideband_order + 1))
        self.assertEqual({row["view"] for row in rows}, set(ARM_VIEW_ORDER))
        self.assertIn("phase_deg", rows[0])
        self.assertIn("power", rows[0])
        self.assertIn("power_db", rows[0])
        self.assertNotIn("arrow_angle_deg", rows[0])

    def test_vector_arrow_length_px_is_gently_scaled_and_monotonic(self) -> None:
        weak = _line_with_db(-60.0)
        middle = _line_with_db(-20.0)
        strong = _line_with_db(0.0)

        self.assertGreater(vector_arrow_length_px(middle), vector_arrow_length_px(weak))
        self.assertGreater(vector_arrow_length_px(strong), vector_arrow_length_px(middle))
        self.assertAlmostEqual(vector_arrow_length_px(strong), ARROW_BASE_LENGTH_PX)
        self.assertGreaterEqual(vector_arrow_length_px(weak), ARROW_BASE_LENGTH_PX * 0.75)

    def test_arrow_display_delta_px_matches_visual_phase_directions(self) -> None:
        up_dx, up_dy = arrow_display_delta_px(90.0, 10.0)
        down_dx, down_dy = arrow_display_delta_px(-90.0, 10.0)
        right_dx, right_dy = arrow_display_delta_px(0.0, 10.0)
        left_dx, left_dy = arrow_display_delta_px(-180.0, 10.0)

        self.assertAlmostEqual(up_dx, 0.0, places=12)
        self.assertGreater(up_dy, 0.0)
        self.assertAlmostEqual(down_dx, 0.0, places=12)
        self.assertLess(down_dy, 0.0)
        self.assertGreater(right_dx, 0.0)
        self.assertAlmostEqual(right_dy, 0.0, places=12)
        self.assertLess(left_dx, 0.0)
        self.assertAlmostEqual(left_dy, 0.0, places=12)

    def test_phase_display_mapping_drives_expected_arrow_directions(self) -> None:
        zero_dx, zero_dy = arrow_display_delta_px(phase_to_display_angle_deg(0.0), 10.0)
        ninety_dx, ninety_dy = arrow_display_delta_px(phase_to_display_angle_deg(90.0), 10.0)
        minus_ninety_dx, minus_ninety_dy = arrow_display_delta_px(
            phase_to_display_angle_deg(-90.0),
            10.0,
        )
        one_eighty_dx, one_eighty_dy = arrow_display_delta_px(
            phase_to_display_angle_deg(180.0),
            10.0,
        )
        forty_five_dx, forty_five_dy = arrow_display_delta_px(
            phase_to_display_angle_deg(45.0),
            10.0,
        )

        self.assertAlmostEqual(zero_dx, 0.0, places=12)
        self.assertGreater(zero_dy, 0.0)
        self.assertAlmostEqual(ninety_dx, 0.0, places=12)
        self.assertGreater(ninety_dy, 0.0)
        self.assertAlmostEqual(minus_ninety_dx, 0.0, places=12)
        self.assertLess(minus_ninety_dy, 0.0)
        self.assertAlmostEqual(one_eighty_dx, 0.0, places=12)
        self.assertLess(one_eighty_dy, 0.0)
        self.assertGreater(forty_five_dx, 0.0)
        self.assertGreater(forty_five_dy, 0.0)

    def test_low_power_sidebands_are_not_visible_or_hover_targets(self) -> None:
        self.assertTrue(is_visible_sideband(_line_with_db(-60.0)))
        self.assertFalse(is_visible_sideband(_line_with_db(-60.01)))


def _manual_bessel_coefficient(
    *,
    order: int,
    bias_voltage: float,
    vpi: float,
    rf_peak_voltage: float,
    rf_relative_phase_deg: float,
) -> complex:
    delta = math.pi * bias_voltage / vpi
    phi_1 = 0.5 * delta
    phi_2 = -0.5 * delta
    modulation_depth = math.pi * rf_peak_voltage / vpi
    arm_rf_phase_diff = math.radians(PUSH_PULL_ARM_RF_PHASE_DIFF_DEG)
    rf_relative_phase = math.radians(rf_relative_phase_deg)

    if order == 0:
        return 0.5 * (np.exp(1j * phi_1) + np.exp(1j * phi_2)) * jv(0, modulation_depth)

    n = abs(order)
    if order > 0:
        return (
            0.5
            * (np.exp(1j * phi_1) + np.exp(1j * (phi_2 + n * arm_rf_phase_diff)))
            * jv(n, modulation_depth)
            * np.exp(1j * n * rf_relative_phase)
        )
    return (
        0.5
        * ((-1) ** n)
        * (np.exp(1j * phi_1) + np.exp(1j * (phi_2 - n * arm_rf_phase_diff)))
        * jv(n, modulation_depth)
        * np.exp(-1j * n * rf_relative_phase)
    )


def _sampled_child_mzm_components(
    *,
    bias_voltage: float,
    vpi: float,
    rf_peak_voltage: float,
    sideband_order: int,
    rf_relative_phase_deg: float,
) -> dict[str, dict[int, complex]]:
    sample_count = 65536
    theta = np.linspace(0.0, 2.0 * math.pi, sample_count, endpoint=False)
    delta = math.pi * bias_voltage / vpi
    phi_1 = 0.5 * delta
    phi_2 = -0.5 * delta
    modulation_depth = math.pi * rf_peak_voltage / vpi
    rf_relative_phase = math.radians(rf_relative_phase_deg)
    rf_phase = theta + rf_relative_phase

    upper_signal = 0.5 * np.exp(1j * (phi_1 + modulation_depth * np.sin(rf_phase)))
    lower_signal = 0.5 * np.exp(1j * (phi_2 - modulation_depth * np.sin(rf_phase)))
    total_signal = upper_signal + lower_signal

    def coefficients(signal) -> dict[int, complex]:
        return {
            order: complex(np.mean(signal * np.exp(-1j * order * theta)))
            for order in range(-sideband_order, sideband_order + 1)
        }

    return {
        "upper": coefficients(upper_signal),
        "lower": coefficients(lower_signal),
        "total": coefficients(total_signal),
    }


def _phase_delta_deg(start: float, end: float) -> float:
    return ((end - start + 180.0) % 360.0) - 180.0


def _relative_luminance(red: float, green: float, blue: float) -> float:
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _line_with_db(magnitude_db: float) -> SpectralLine:
    magnitude = 10 ** (magnitude_db / 20.0)
    return SpectralLine(
        view="test",
        order=0,
        freq_offset_ghz=0.0,
        magnitude=magnitude,
        power=magnitude**2,
        magnitude_db=magnitude_db,
        power_db=magnitude_db,
        phase_deg=0.0,
        real=1.0,
        imag=0.0,
    )


if __name__ == "__main__":
    unittest.main()
