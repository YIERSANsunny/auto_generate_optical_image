"""Desktop GUI for automatic DPMZM optical spectrum drawing."""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")

from matplotlib import colormaps
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.transforms import offset_copy

from dpmzm_model import (
    ARM_VIEW_ORDER,
    DPMZMParams,
    SpectralLine,
    VIEW_ORDER,
    default_params,
    phase_to_display_angle_deg,
    simulate_arm_spectra,
    simulate_spectra,
    write_spectra_csv,
)


PHASE_LABEL_DB_THRESHOLD = -60.0
ARROW_BASE_LENGTH_PX = 56.0
HOVER_DISTANCE_PX = 18.0
DISPLAY_MODE_OVERVIEW = "总览"
DISPLAY_MODE_ARMS = "臂分解"


def is_visible_sideband(line: SpectralLine) -> bool:
    """Return whether a sideband should be drawn and exposed to hover."""

    return line.magnitude_db >= PHASE_LABEL_DB_THRESHOLD


def vector_arrow_length_px(line: SpectralLine) -> float:
    """Return a gently magnitude-scaled arrow length in display pixels.

    Power is not encoded as a y-coordinate. This small length change only helps
    the eye notice strong and weak components without turning the plot back into
    a dB spectrum.
    """

    normalized_magnitude = min(1.0, max(0.0, 10 ** (line.magnitude_db / 20.0)))
    return ARROW_BASE_LENGTH_PX * (0.75 + 0.25 * normalized_magnitude)


def arrow_display_delta_px(display_angle_deg: float, length_px: float) -> tuple[float, float]:
    """Return the display-space x/y offset for an arrow angle and length."""

    angle_rad = math.radians(display_angle_deg)
    return length_px * math.cos(angle_rad), length_px * math.sin(angle_rad)


class DPMZMSpectrumApp(tk.Tk):
    """Tkinter application that plots DPMZM spectra with vector sidebands."""

    def __init__(self) -> None:
        super().__init__()
        self.title("DPMZM自动光谱图工具")
        self.geometry("1360x900")
        self.minsize(1100, 760)

        self._update_job: str | None = None
        self.vars: dict[str, tk.StringVar] = {}
        self.display_mode_var = tk.StringVar(value=DISPLAY_MODE_OVERVIEW)
        self.current_params = default_params()
        self.current_spectra = None
        self.hover_targets: list[dict[str, object]] = []
        self.hover_annotation = None

        self._configure_matplotlib()
        self._build_layout()
        self._install_traces()
        self._update_plot(show_error=True)

    def _configure_matplotlib(self) -> None:
        matplotlib.rcParams["font.sans-serif"] = [
            "Microsoft YaHei",
            "SimHei",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]
        matplotlib.rcParams["axes.unicode_minus"] = False

    def _build_layout(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        controls = ttk.Frame(self, padding=(12, 12, 8, 12))
        controls.grid(row=0, column=0, sticky="ns")

        self._add_display_controls(controls)
        self._add_bias_controls(controls)
        self._add_rf_controls(controls)
        self._add_action_buttons(controls)

        plot_area = ttk.Frame(self, padding=(4, 8, 12, 12))
        plot_area.grid(row=0, column=1, sticky="nsew")
        plot_area.columnconfigure(0, weight=1)
        plot_area.rowconfigure(0, weight=1)

        self.figure = Figure(figsize=(10.6, 7.6), dpi=100, constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_area)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_area, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=1, column=0, sticky="ew")

        self.status_var = tk.StringVar(value="准备就绪")
        ttk.Label(self, textvariable=self.status_var, anchor="w", padding=(12, 4)).grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="ew",
        )

    def _add_display_controls(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="显示模式", padding=10)
        frame.pack(fill="x", pady=(0, 10))
        combo = ttk.Combobox(
            frame,
            textvariable=self.display_mode_var,
            values=(DISPLAY_MODE_OVERVIEW, DISPLAY_MODE_ARMS),
            state="readonly",
            width=14,
        )
        combo.pack(fill="x")
        combo.bind("<<ComboboxSelected>>", lambda _event: self._update_plot(show_error=True))

    def _add_bias_controls(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="直流偏压与Vpi", padding=10)
        frame.pack(fill="x", pady=(0, 10))
        fields = [
            ("voltage_i", "I路偏压 VI (V)", self.current_params.voltage_i),
            ("voltage_q", "Q路偏压 VQ (V)", self.current_params.voltage_q),
            ("voltage_p", "P路偏压 VP (V)", self.current_params.voltage_p),
            ("vpi_i", "I路 Vpi (V)", self.current_params.vpi_i),
            ("vpi_q", "Q路 Vpi (V)", self.current_params.vpi_q),
            ("vpi_p", "P路 Vpi (V)", self.current_params.vpi_p),
        ]
        self._add_entry_grid(frame, fields)

    def _add_rf_controls(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="单音RF与边带", padding=10)
        frame.pack(fill="x", pady=(0, 10))
        fields = [
            ("rf_frequency_ghz", "RF频率 (GHz)", self.current_params.rf_frequency_ghz),
            ("rf_amplitude_i_v", "I路RF峰值电压 (V)", self.current_params.rf_amplitude_i_v),
            ("rf_amplitude_q_v", "Q路RF峰值电压 (V)", self.current_params.rf_amplitude_q_v),
            ("q_rf_phase_deg", "Q路RF相位 (deg)", self.current_params.q_rf_phase_deg),
            ("sideband_order", "边带阶数", self.current_params.sideband_order),
        ]
        self._add_entry_grid(frame, fields)

    def _add_entry_grid(
        self,
        parent: ttk.LabelFrame,
        fields: list[tuple[str, str, float | int]],
    ) -> None:
        for row, (key, label, value) in enumerate(fields):
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
            var = tk.StringVar(value=self._format_default(value))
            entry = ttk.Entry(parent, textvariable=var, width=16)
            entry.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=3)
            entry.bind("<Return>", lambda _event: self._update_plot(show_error=True))
            self.vars[key] = var
        parent.columnconfigure(1, weight=1)

    def _add_action_buttons(self, parent: ttk.Frame) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=(2, 0))
        ttk.Button(frame, text="更新图像", command=lambda: self._update_plot(show_error=True)).pack(
            fill="x",
            pady=3,
        )
        ttk.Button(frame, text="重置参数", command=self._reset_params).pack(fill="x", pady=3)
        ttk.Button(frame, text="保存PNG", command=self._save_png).pack(fill="x", pady=3)
        ttk.Button(frame, text="导出CSV", command=self._export_csv).pack(fill="x", pady=3)

        note = (
            "说明：I/Q子MZM按推挽建模；图面只显示边带相位矢量。"
            "功率不再作为纵坐标，鼠标悬停到边带时显示。"
        )
        ttk.Label(parent, text=note, wraplength=245, foreground="#555").pack(
            fill="x",
            pady=(14, 0),
        )

    def _install_traces(self) -> None:
        for var in self.vars.values():
            var.trace_add("write", lambda *_args: self._schedule_update())

    def _schedule_update(self) -> None:
        if self._update_job is not None:
            self.after_cancel(self._update_job)
        self._update_job = self.after(350, self._run_scheduled_update)

    def _run_scheduled_update(self) -> None:
        self._update_job = None
        self._update_plot(show_error=False)

    def _reset_params(self) -> None:
        params = default_params()
        for key, value in {
            "voltage_i": params.voltage_i,
            "voltage_q": params.voltage_q,
            "voltage_p": params.voltage_p,
            "vpi_i": params.vpi_i,
            "vpi_q": params.vpi_q,
            "vpi_p": params.vpi_p,
            "rf_frequency_ghz": params.rf_frequency_ghz,
            "rf_amplitude_i_v": params.rf_amplitude_i_v,
            "rf_amplitude_q_v": params.rf_amplitude_q_v,
            "q_rf_phase_deg": params.q_rf_phase_deg,
            "sideband_order": params.sideband_order,
        }.items():
            self.vars[key].set(self._format_default(value))
        self._update_plot(show_error=True)

    def _parse_params(self) -> DPMZMParams:
        values = {key: self._parse_float(key) for key in self.vars if key != "sideband_order"}
        sideband_order = self._parse_int("sideband_order")
        return DPMZMParams(
            voltage_i=values["voltage_i"],
            voltage_q=values["voltage_q"],
            voltage_p=values["voltage_p"],
            vpi_i=values["vpi_i"],
            vpi_q=values["vpi_q"],
            vpi_p=values["vpi_p"],
            rf_frequency_ghz=values["rf_frequency_ghz"],
            rf_amplitude_i_v=values["rf_amplitude_i_v"],
            rf_amplitude_q_v=values["rf_amplitude_q_v"],
            q_rf_phase_deg=values["q_rf_phase_deg"],
            sideband_order=sideband_order,
        )

    def _parse_float(self, key: str) -> float:
        raw = self.vars[key].get().strip()
        if raw == "":
            raise ValueError("参数不能为空。")
        return float(raw)

    def _parse_int(self, key: str) -> int:
        raw = self.vars[key].get().strip()
        if raw == "":
            raise ValueError("边带阶数不能为空。")
        value = float(raw)
        if not value.is_integer():
            raise ValueError("边带阶数必须是整数。")
        return int(value)

    def _update_plot(self, show_error: bool) -> None:
        if self._update_job is not None:
            self.after_cancel(self._update_job)
            self._update_job = None
        try:
            params = self._parse_params()
            mode = self.display_mode_var.get()
            if mode == DISPLAY_MODE_ARMS:
                spectra = simulate_arm_spectra(params)
                view_order = ARM_VIEW_ORDER
            else:
                spectra = simulate_spectra(params)
                view_order = VIEW_ORDER
        except Exception as exc:
            self.status_var.set(f"参数错误：{exc}")
            if show_error:
                messagebox.showerror("参数错误", str(exc))
            return

        self.current_params = params
        self.current_spectra = spectra
        self._draw_spectra(params, spectra, view_order)
        self.status_var.set(f"已更新图像：{mode}")

    def _draw_spectra(
        self,
        params: DPMZMParams,
        spectra,
        view_order: tuple[str, ...],
    ) -> None:
        self.figure.clear()
        columns = 3 if view_order == ARM_VIEW_ORDER else 2
        axes = self.figure.subplots(2, columns, sharex=True, sharey=True, squeeze=False).ravel()
        cmap = colormaps["twilight_shifted"]
        norm = Normalize(vmin=-180.0, vmax=180.0)
        self.hover_targets = []
        self.hover_annotation = None

        for index, (axis, view) in enumerate(zip(axes, view_order)):
            axis.axhline(0.0, color="#777", linewidth=1.1, alpha=0.8)
            axis.grid(True, linewidth=0.5, alpha=0.24)
            axis.set_title(view, fontsize=13, pad=8)
            axis.set_ylim(-1.18, 1.18)
            axis.set_xlim(-params.sideband_order - 0.6, params.sideband_order + 0.6)
            axis.set_xticks(list(range(-params.sideband_order, params.sideband_order + 1)))
            axis.set_yticks([])
            axis.set_ylabel("")
            is_bottom_row = index >= columns
            axis.tick_params(axis="x", length=0, labelbottom=is_bottom_row)
            axis.spines["left"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["top"].set_visible(False)
            axis.spines["bottom"].set_visible(False)
            if is_bottom_row:
                axis.set_xlabel("边带阶数 k")

            for line in spectra[view]:
                if not is_visible_sideband(line):
                    continue
                color = cmap(norm(line.phase_deg))
                self._draw_phase_arrow(axis, line, color)

        axes[-1].set_xlabel(f"边带阶数 k，频偏 = k × {params.rf_frequency_ghz:g} GHz")

        scalar_mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar_mappable.set_array([])
        colorbar = self.figure.colorbar(
            scalar_mappable,
            ax=axes.tolist(),
            orientation="horizontal",
            fraction=0.035,
            pad=0.035,
        )
        colorbar.set_label("真实相位 (deg)")
        colorbar.set_ticks([-180, -90, 0, 90, 180])
        self.canvas.draw_idle()

    def _draw_phase_arrow(self, axis, line: SpectralLine, color) -> None:
        start = (line.order, 0.0)
        length_px = vector_arrow_length_px(line)
        display_angle_deg = phase_to_display_angle_deg(line.phase_deg)
        dx_px, dy_px = arrow_display_delta_px(display_angle_deg, length_px)
        tip_transform = offset_copy(
            axis.transData,
            fig=self.figure,
            x=dx_px,
            y=dy_px,
            units="dots",
        )

        axis.annotate(
            "",
            xy=start,
            xycoords=tip_transform,
            xytext=start,
            textcoords=axis.transData,
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "linewidth": 2.2,
                "shrinkA": 0.0,
                "shrinkB": 0.0,
            },
            zorder=4,
        )
        axis.scatter([line.order], [0.0], color=[color], s=28, zorder=5)

        label_transform = offset_copy(
            axis.transData,
            fig=self.figure,
            x=dx_px,
            y=dy_px + (12.0 if dy_px >= 0.0 else -12.0),
            units="dots",
        )
        axis.text(
            line.order,
            0.0,
            f"{line.phase_deg:+.1f}°",
            transform=label_transform,
            ha="center",
            va="bottom" if dy_px >= 0.0 else "top",
            fontsize=10,
            color=color,
            zorder=5,
        )
        self.hover_targets.append(
            {
                "axis": axis,
                "line": line,
                "start": start,
                "offset_px": (dx_px, dy_px),
            }
        )

    def _on_mouse_move(self, event) -> None:
        if event.inaxes is None or event.x is None or event.y is None:
            self._hide_hover_annotation()
            return

        target = self._nearest_hover_target(event)
        if target is None:
            self._hide_hover_annotation()
            return

        line = target["line"]
        axis = target["axis"]
        tip_px = self._hover_points_px(target)[-1]
        tip = axis.transData.inverted().transform(tip_px)
        if self.hover_annotation is None or self.hover_annotation.axes is not axis:
            self._hide_hover_annotation(draw=False)
            self.hover_annotation = axis.annotate(
                "",
                xy=tip,
                xytext=(12, 12),
                textcoords="offset points",
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "fc": "#fffff2",
                    "ec": "#777",
                    "alpha": 0.95,
                },
                arrowprops={"arrowstyle": "->", "color": "#666"},
                fontsize=9,
                zorder=10,
            )
        self.hover_annotation.xy = tip
        self.hover_annotation.set_text(self._format_hover_text(line))
        self.hover_annotation.set_visible(True)
        self.canvas.draw_idle()

    def _nearest_hover_target(self, event):
        nearest = None
        nearest_distance = HOVER_DISTANCE_PX
        for target in self.hover_targets:
            if target["axis"] is not event.inaxes:
                continue
            for px, py in self._hover_points_px(target):
                distance = math.hypot(event.x - px, event.y - py)
                if distance <= nearest_distance:
                    nearest = target
                    nearest_distance = distance
        return nearest

    def _hover_points_px(self, target) -> tuple[tuple[float, float], ...]:
        axis = target["axis"]
        start_px = axis.transData.transform(target["start"])
        dx_px, dy_px = target["offset_px"]
        middle_px = (start_px[0] + dx_px * 0.5, start_px[1] + dy_px * 0.5)
        tip_px = (start_px[0] + dx_px, start_px[1] + dy_px)
        return (tuple(start_px), middle_px, tip_px)

    def _hide_hover_annotation(self, draw: bool = True) -> None:
        if self.hover_annotation is None:
            return
        if self.hover_annotation.get_visible():
            self.hover_annotation.set_visible(False)
            if draw:
                self.canvas.draw_idle()

    def _format_hover_text(self, line: SpectralLine) -> str:
        return (
            f"{line.view}\n"
            f"边带阶数: {line.order:+d}\n"
            f"频偏: {line.freq_offset_ghz:+.6g} GHz\n"
            f"真实相位: {line.phase_deg:+.2f}°\n"
            f"场幅 |Ck|: {line.magnitude:.6g}\n"
            f"光功率 |Ck|²: {line.power:.6g}\n"
            f"相对功率: {line.power_db:.2f} dB\n"
            f"复数: {line.real:+.6g} {line.imag:+.6g}j"
        )

    def _save_png(self) -> None:
        if not self._ensure_current_spectra():
            return
        path = filedialog.asksaveasfilename(
            title="保存光谱图",
            defaultextension=".png",
            filetypes=[
                ("PNG图像", "*.png"),
                ("SVG矢量图", "*.svg"),
                ("PDF文档", "*.pdf"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return
        self.figure.savefig(path, dpi=200, bbox_inches="tight")
        self.status_var.set(f"已保存图像：{path}")

    def _export_csv(self) -> None:
        if not self._ensure_current_spectra():
            return
        path = filedialog.asksaveasfilename(
            title="导出光谱数据",
            defaultextension=".csv",
            filetypes=[("CSV表格", "*.csv"), ("所有文件", "*.*")],
        )
        if not path:
            return
        write_spectra_csv(self.current_spectra, path)
        self.status_var.set(f"已导出CSV：{path}")

    def _ensure_current_spectra(self) -> bool:
        if self.current_spectra is not None:
            return True
        self._update_plot(show_error=True)
        return self.current_spectra is not None

    def _format_default(self, value: float | int) -> str:
        if isinstance(value, int):
            return str(value)
        return f"{value:g}"


def main() -> None:
    app = DPMZMSpectrumApp()
    app.mainloop()


if __name__ == "__main__":
    main()
