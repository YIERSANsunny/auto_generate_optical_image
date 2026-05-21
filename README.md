# DPMZM自动光谱图工具

这是一个用于自动绘制 DPMZM 光谱图的桌面小工具。当前版本按贝塞尔边带展开计算 I/Q 子 MZM 的复数边带系数，再绘制相位矢量图。

四个视图包括：

- I路光谱
- Q路光谱
- Q路经过P路相移后的光谱
- I/Q相干耦合后的输出光谱

该版本用于相对光谱和相位趋势展示，不做绝对光功率、插损、消光比或实验标定建模。

## 运行

```powershell
python app.py
```

需要 Python 3，以及：

```powershell
python -m pip install -r requirements.txt
```

GUI 使用 Python 自带的 `Tkinter`，不依赖 PyQt/PySide。贝塞尔函数由 `scipy.special.jv` 提供。

## 贝塞尔边带模型

每个子 MZM 用一个偏压差得到两臂静态相位：

```text
delta = pi * Vbias / Vpi
phi1 = +delta / 2
phi2 = -delta / 2
```

RF 峰值电压换算为调制深度：

```text
m = pi * Vrf_peak / Vpi
```

子 MZM 内部两臂 RF 相位差固定为推挽：

```text
phi = 180 deg
```

对每个边带阶数计算复系数：

```text
C0   = 0.5 * [exp(j*phi1) + exp(j*phi2)] * J0(m)
C+n  = 0.5 * [exp(j*phi1) + exp(j*(phi2+n*phi))] * Jn(m) * exp(j*n*psi)
C-n  = 0.5 * (-1)^n * [exp(j*phi1) + exp(j*(phi2-n*phi))] * Jn(m) * exp(-j*n*psi)
```

其中：

- I路使用 `psi=0`
- Q路使用GUI里的 `Q路RF相位`，默认 `psi=90 deg`
- `J_n(m)` 是 n 阶第一类贝塞尔函数

DPMZM组合关系：

```text
I       = child_mzm(VI, VrfI, psi=0)
Q       = child_mzm(VQ, VrfQ, psi=q_rf_phase)
Q_after = exp(j*pi*VP/VpiP) * Q
Out     = (I + Q_after) / sqrt(2)
```

## 矢量边带显示

图面不使用纵坐标表示功率，也不画 dB 谱线。每个可见边带在中间参考轴 `y=0` 上从对应阶数 `k` 出发画一个相位箭头：

- 横轴位置表示边带阶数。
- 中间灰色横线是相位矢量的出发参考轴。
- 底部坐标轴横线和刻度小竖线已隐藏，只保留必要的边带阶数数字。
- 箭头方向表示相位矢量。
- 箭头文字标注真实物理相位 `phase_deg`。
- 箭头长度是视觉显示尺寸，只轻微参考边带幅度，不是功率坐标。
- 精确功率、线性幅度、复数值等信息通过鼠标悬停提示框或 CSV 查看。

低于 `-60 dB` 的边带不会显示箭头或相位文字，也不会进入悬停目标，避免近零边带的随机相位污染画面。

箭头方向采用显示映射：

```text
arrow_angle_deg = wrap(90 - phase_deg)
```

这个映射的目的只是为了看图时更容易区分 0° 和 180°：

```text
真实相位 0°      -> 箭头向上 90°
真实相位 180°   -> 箭头向下 -90°
真实相位 90°    -> 箭头向右 0°
真实相位 -90°   -> 箭头向左 180°/-180°
```

CSV 中同时导出真实相位 `phase_deg` 和箭头显示角度 `arrow_angle_deg`。后续做物理分析时应使用 `phase_deg`。

## GUI参数

默认值：

- `VI=0 V, VQ=0 V, VP=0 V`
- `VpiI=VpiQ=VpiP=4 V`
- `RF频率=10 GHz`
- `I/Q RF峰值电压=0.4 V`
- `Q路RF相位=90 deg`
- `边带阶数=5`

按钮功能：

- `更新图像`：按当前输入重新计算并绘图
- `重置参数`：恢复默认值
- `保存PNG`：保存当前四视图，也可在文件名中选择 SVG/PDF
- `导出CSV`：导出每条谱线的数据

鼠标悬停到可见边带时，提示框会显示视图、边带阶数、频偏、真实相位、箭头角度、线性幅度、归一化功率 dB、复数实部/虚部。

CSV 字段包括：

```text
view, order, freq_offset_ghz, magnitude, magnitude_db, phase_deg, arrow_angle_deg, real, imag
```

## 测试

```powershell
python -m unittest discover
```

测试覆盖：

- 贝塞尔展开式逐项验证
- 负阶边带的 `(-1)^n` 符号
- 零偏压推挽结构下奇数阶边带相消
- Q路RF相对相位按边带阶数旋转
- P路相移和最终耦合关系
- 箭头角度映射、像素箭头长度缩放、显示阈值和 CSV 导出字段
