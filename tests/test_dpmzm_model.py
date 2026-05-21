from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.special import jv

from dpmzm_model import (
    DPMZMParams,
    PUSH_PULL_ARM_RF_PHASE_DIFF_DEG,
    SpectralLine,
    VIEW_COUPLED,
    VIEW_I,
    VIEW_ORDER,
    VIEW_Q_AFTER_P,
    child_mzm_coefficients,
    phase_to_arrow_angle_deg,
    simulate_spectra,
    write_spectra_csv,
)
from app import (
    ARROW_BASE_LENGTH_PX,
    arrow_display_delta_px,
    is_visible_sideband,
    vector_arrow_length_px,
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
        coefficients = child_mzm_coefficients(
            bias_voltage=0.0,
            vpi=4.0,
            rf_peak_voltage=0.8,
            sideband_order=5,
        )

        for order in [-5, -3, -1, 1, 3, 5]:
            self.assertLess(abs(coefficients[order]), 1e-12)
        for order in [-4, -2, 0, 2, 4]:
            self.assertGreater(abs(coefficients[order]), 1e-12)

    def test_q_rf_relative_phase_rotates_each_sideband_by_order(self) -> None:
        base = child_mzm_coefficients(
            bias_voltage=0.9,
            vpi=4.0,
            rf_peak_voltage=0.7,
            sideband_order=3,
            rf_relative_phase_deg=0.0,
        )
        shifted = child_mzm_coefficients(
            bias_voltage=0.9,
            vpi=4.0,
            rf_peak_voltage=0.7,
            sideband_order=3,
            rf_relative_phase_deg=30.0,
        )

        psi = math.radians(30.0)
        for order in [-3, -2, -1, 0, 1, 2, 3]:
            expected = base[order] * np.exp(1j * order * psi)
            self.assertAlmostEqual(shifted[order].real, expected.real, places=12)
            self.assertAlmostEqual(shifted[order].imag, expected.imag, places=12)

    def test_p_bias_shift_by_one_vpi_rotates_q_after_p_by_180_degrees(self) -> None:
        base = simulate_spectra(DPMZMParams(sideband_order=4))
        shifted = simulate_spectra(DPMZMParams(voltage_p=4.0, sideband_order=4))

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

    def test_phase_to_arrow_angle_mapping(self) -> None:
        self.assertEqual(phase_to_arrow_angle_deg(0.0), 90.0)
        self.assertEqual(phase_to_arrow_angle_deg(180.0), -90.0)
        self.assertEqual(phase_to_arrow_angle_deg(-180.0), -90.0)
        self.assertEqual(phase_to_arrow_angle_deg(90.0), 0.0)

    def test_csv_export_contains_all_views_orders_and_arrow_angle(self) -> None:
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
        self.assertIn("arrow_angle_deg", rows[0])

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


def _phase_delta_deg(start: float, end: float) -> float:
    return ((end - start + 180.0) % 360.0) - 180.0


def _line_with_db(magnitude_db: float) -> SpectralLine:
    return SpectralLine(
        view="test",
        order=0,
        freq_offset_ghz=0.0,
        magnitude=10 ** (magnitude_db / 20.0),
        magnitude_db=magnitude_db,
        phase_deg=0.0,
        arrow_angle_deg=90.0,
        real=1.0,
        imag=0.0,
    )


if __name__ == "__main__":
    unittest.main()
