"""DPMZM Bessel sideband simulation helpers.

Each child MZM is modeled as two phase arms whose sidebands are coherently
summed. The RF modulation depth controls the Bessel coefficients J_n(m), while
the arm phases set the vector direction of every sideband.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.special import jv


VIEW_I = "I路"
VIEW_Q = "Q路"
VIEW_Q_AFTER_P = "Q路经过P路"
VIEW_COUPLED = "耦合输出"
VIEW_ORDER = (VIEW_I, VIEW_Q, VIEW_Q_AFTER_P, VIEW_COUPLED)

PUSH_PULL_ARM_RF_PHASE_DIFF_DEG = 180.0


@dataclass(frozen=True)
class DPMZMParams:
    """Input parameters for the DPMZM Bessel sideband model."""

    voltage_i: float = 0.0
    voltage_q: float = 0.0
    voltage_p: float = 0.0
    vpi_i: float = 4.0
    vpi_q: float = 4.0
    vpi_p: float = 4.0
    rf_frequency_ghz: float = 10.0
    rf_amplitude_i_v: float = 0.4
    rf_amplitude_q_v: float = 0.4
    q_rf_phase_deg: float = 90.0
    sideband_order: int = 5


@dataclass(frozen=True)
class SpectralLine:
    """One optical spectral line in a simulated spectrum."""

    view: str
    order: int
    freq_offset_ghz: float
    magnitude: float
    magnitude_db: float
    phase_deg: float
    arrow_angle_deg: float
    real: float
    imag: float


def default_params() -> DPMZMParams:
    """Return the GUI/default simulation parameters."""

    return DPMZMParams()


def child_mzm_coefficients(
    *,
    bias_voltage: float,
    vpi: float,
    rf_peak_voltage: float,
    sideband_order: int,
    rf_relative_phase_deg: float = 0.0,
    arm_rf_phase_diff_deg: float = PUSH_PULL_ARM_RF_PHASE_DIFF_DEG,
) -> dict[int, complex]:
    """Return sideband coefficients for one child MZM.

    The static arm phases are derived from a differential bias:
        delta = pi * Vbias / Vpi, phi1 = +delta/2, phi2 = -delta/2.

    The positive and negative sidebands follow the Bessel expansion:
        C_+n = 0.5 * [e^(j phi1) + e^(j(phi2+n phi))] J_n(m) e^(j n psi)
        C_-n = 0.5 * (-1)^n [e^(j phi1) + e^(j(phi2-n phi))] J_n(m) e^(-j n psi)
    """

    if vpi == 0.0:
        raise ValueError("Vpi不能为0。")
    if rf_peak_voltage < 0.0:
        raise ValueError("RF峰值电压不能为负。")
    if sideband_order < 0:
        raise ValueError("边带阶数不能为负。")

    delta = math.pi * bias_voltage / vpi
    phi_1 = 0.5 * delta
    phi_2 = -0.5 * delta
    modulation_depth = math.pi * rf_peak_voltage / vpi
    arm_rf_phase_diff = math.radians(arm_rf_phase_diff_deg)
    rf_relative_phase = math.radians(rf_relative_phase_deg)

    coefficients: dict[int, complex] = {}
    coefficients[0] = (
        0.5
        * (np.exp(1j * phi_1) + np.exp(1j * phi_2))
        * float(jv(0, modulation_depth))
    )

    for order in range(1, sideband_order + 1):
        bessel_value = float(jv(order, modulation_depth))
        positive = (
            0.5
            * (np.exp(1j * phi_1) + np.exp(1j * (phi_2 + order * arm_rf_phase_diff)))
            * bessel_value
            * np.exp(1j * order * rf_relative_phase)
        )
        negative = (
            0.5
            * ((-1) ** order)
            * (np.exp(1j * phi_1) + np.exp(1j * (phi_2 - order * arm_rf_phase_diff)))
            * bessel_value
            * np.exp(-1j * order * rf_relative_phase)
        )
        coefficients[order] = complex(positive)
        coefficients[-order] = complex(negative)

    return coefficients


def simulate_spectra(params: DPMZMParams) -> dict[str, list[SpectralLine]]:
    """Simulate I, Q, Q-after-P, and coherently coupled DPMZM spectra."""

    _validate_params(params)

    i_coeffs = child_mzm_coefficients(
        bias_voltage=params.voltage_i,
        vpi=params.vpi_i,
        rf_peak_voltage=params.rf_amplitude_i_v,
        sideband_order=params.sideband_order,
        rf_relative_phase_deg=0.0,
    )
    q_coeffs = child_mzm_coefficients(
        bias_voltage=params.voltage_q,
        vpi=params.vpi_q,
        rf_peak_voltage=params.rf_amplitude_q_v,
        sideband_order=params.sideband_order,
        rf_relative_phase_deg=params.q_rf_phase_deg,
    )
    phi_p = math.pi * params.voltage_p / params.vpi_p
    p_phase = np.exp(1j * phi_p)

    q_after_p_coeffs = {
        order: complex(p_phase * q_coeffs[order])
        for order in _orders(params.sideband_order)
    }
    coupled_coeffs = {
        order: complex((i_coeffs[order] + q_after_p_coeffs[order]) / math.sqrt(2.0))
        for order in _orders(params.sideband_order)
    }

    raw = {
        VIEW_I: i_coeffs,
        VIEW_Q: q_coeffs,
        VIEW_Q_AFTER_P: q_after_p_coeffs,
        VIEW_COUPLED: coupled_coeffs,
    }

    max_magnitude = max(
        (abs(value) for coeffs in raw.values() for value in coeffs.values()),
        default=0.0,
    )
    if max_magnitude <= 0.0:
        max_magnitude = 1.0

    spectra: dict[str, list[SpectralLine]] = {}
    for view in VIEW_ORDER:
        lines: list[SpectralLine] = []
        for order in _orders(params.sideband_order):
            coefficient = raw[view][order]
            magnitude = float(abs(coefficient))
            magnitude_db = _relative_db(magnitude, max_magnitude)
            phase_deg = _wrap_phase_deg(math.degrees(float(np.angle(coefficient))))
            lines.append(
                SpectralLine(
                    view=view,
                    order=order,
                    freq_offset_ghz=order * params.rf_frequency_ghz,
                    magnitude=magnitude,
                    magnitude_db=magnitude_db,
                    phase_deg=phase_deg,
                    arrow_angle_deg=phase_to_arrow_angle_deg(phase_deg),
                    real=float(np.real(coefficient)),
                    imag=float(np.imag(coefficient)),
                )
            )
        spectra[view] = lines
    return spectra


def phase_to_arrow_angle_deg(phase_deg: float) -> float:
    """Map true optical phase to display angle for the vector arrows."""

    return _wrap_phase_deg(90.0 - phase_deg)


def flatten_spectra(spectra: dict[str, list[SpectralLine]]) -> list[SpectralLine]:
    """Return spectral lines in the standard view/order sequence."""

    lines: list[SpectralLine] = []
    for view in VIEW_ORDER:
        lines.extend(spectra.get(view, []))
    return lines


def write_spectra_csv(
    spectra: dict[str, list[SpectralLine]],
    path: str | Path,
) -> None:
    """Write spectrum data to a CSV file readable by Excel."""

    fieldnames = [
        "view",
        "order",
        "freq_offset_ghz",
        "magnitude",
        "magnitude_db",
        "phase_deg",
        "arrow_angle_deg",
        "real",
        "imag",
    ]
    with Path(path).open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for line in flatten_spectra(spectra):
            writer.writerow(
                {
                    "view": line.view,
                    "order": line.order,
                    "freq_offset_ghz": _format_float(line.freq_offset_ghz),
                    "magnitude": _format_float(line.magnitude),
                    "magnitude_db": _format_float(line.magnitude_db),
                    "phase_deg": _format_float(line.phase_deg),
                    "arrow_angle_deg": _format_float(line.arrow_angle_deg),
                    "real": _format_float(line.real),
                    "imag": _format_float(line.imag),
                }
            )


def _orders(sideband_order: int) -> Iterable[int]:
    return range(-sideband_order, sideband_order + 1)


def _relative_db(magnitude: float, reference: float) -> float:
    if magnitude <= 1e-15:
        return -300.0
    return 20.0 * math.log10(magnitude / reference)


def _wrap_phase_deg(value: float) -> float:
    return ((value + 180.0) % 360.0) - 180.0


def _format_float(value: float) -> str:
    return f"{value:.12g}"


def _validate_params(params: DPMZMParams) -> None:
    if params.vpi_i == 0.0 or params.vpi_q == 0.0 or params.vpi_p == 0.0:
        raise ValueError("Vpi不能为0。")
    if params.rf_frequency_ghz <= 0.0:
        raise ValueError("RF频率必须大于0。")
    if params.rf_amplitude_i_v < 0.0 or params.rf_amplitude_q_v < 0.0:
        raise ValueError("RF峰值电压不能为负。")
    if params.sideband_order < 0:
        raise ValueError("边带阶数不能为负。")
