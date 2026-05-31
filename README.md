# DPMZM自动光谱图工具

这是一个用于自动绘制 DPMZM 光谱图的桌面小工具。当前版本按贝塞尔边带展开计算 I/Q 子 MZM 的复数边带系数，再绘制相位矢量图。

完整物理模型、符号约定和验证方法见 [MODEL.md](MODEL.md)。

显示模式包括：

- `总览`：I路、Q路、Q路经过P路、I/Q相干耦合输出
- `臂分解`：I上臂、I下臂、I合成、Q上臂、Q下臂、Q合成

该版本用于相对光谱和相位趋势展示，不做绝对光功率、插损、消光比或实验标定建模。相干耦合始终使用复数场系数 `Ck` 相加；光功率只在复数场合成之后由 `|Ck|²` 计算。

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

GUI 可以用 `偏压(V)` 或 `相位(deg)` 设置同一个静态偏置。两种输入完全等价，换算关系为：

```text
phase_deg = 180 * Vbias / Vpi
Vbias = phase_deg * Vpi / 180
```

RF 峰值电压换算为调制深度：

```text
m = pi * Vrf_peak / Vpi
```

子 MZM 内部两臂 RF 相位差固定为推挽：

```text
phi = 180 deg
```

每个边带先分解为上臂贡献和下臂贡献，再相干相加：

```text
upper_0 = 0.5 * exp(j*phi1) * J0(m)
lower_0 = 0.5 * exp(j*phi2) * J0(m)

upper_+n = 0.5 * exp(j*phi1) * Jn(m) * exp(j*n*psi)
lower_+n = 0.5 * exp(j*(phi2+n*phi)) * Jn(m) * exp(j*n*psi)

upper_-n = 0.5 * (-1)^n * exp(j*phi1) * Jn(m) * exp(-j*n*psi)
lower_-n = 0.5 * (-1)^n * exp(j*(phi2-n*phi)) * Jn(m) * exp(-j*n*psi)

total_k = upper_k + lower_k
```

其中 I路使用 `psi=0`，Q路使用GUI里的 `Q路RF相位`，默认 `psi=0 deg`。

DPMZM组合关系：

```text
I       = child_mzm(VI, VrfI, psi=0).total
Q       = child_mzm(VQ, VrfQ, psi=q_rf_phase).total
Q_after = exp(j*pi*VP/VpiP) * Q
Out     = (I + Q_after) / sqrt(2)
```

`臂分解` 模式中的上臂/下臂光谱表示进入子 MZM 输出合成前、已经带有 `0.5` 系数的复数贡献项。

## 矢量边带显示

图面不使用纵坐标表示功率，也不画 dB 谱线。每个可见边带在中间参考轴 `y=0` 上从对应阶数 `k` 出发画一个相位箭头：

- 横轴位置表示边带阶数。
- 中间灰色横线是相位矢量的出发参考轴。
- 箭头文字标注真实物理相位 `phase_deg`。
- 箭头长度是视觉显示尺寸，只轻微参考边带幅度，不是功率坐标。
- 精确场幅、相对光功率、复数值等信息通过鼠标悬停提示框或 CSV 查看。
- 相位颜色使用高对比循环映射：`0°` 为深绿色，`-90°` 为蓝色，`+90°` 为橙色，`±180°` 为同一种深红/紫红。

箭头方向大多数时候按真实相位角绘制。为了避免水平箭头不直观，接近 `0°` 和 `±180°` 的相位会特殊竖直显示：

```text
真实相位接近 0°      -> 箭头向上 90°
真实相位接近 ±180°  -> 箭头向下 -90°
真实相位 +90°       -> 箭头向上 90°
真实相位 -90°       -> 箭头向下 -90°
真实相位 +45°       -> 箭头按 +45° 倾斜显示
```

低于 `-60 dB` 的边带不会显示箭头或相位文字，也不会进入悬停目标，避免近零边带的随机相位污染画面。

CSV 只导出真实物理相位 `phase_deg`，不导出视觉箭头角度。导出内容会跟随当前显示模式：总览导出4个视图，臂分解导出6个视图。

## GUI参数

默认值：

- `显示模式=总览`
- `偏置输入方式=偏压(V)`
- `VI=0 V, VQ=0 V, VP=0 V`
- `VpiI=VpiQ=VpiP=5 V`
- `RF频率=10 GHz`
- `I/Q RF峰值电压=0.4 V`
- `Q路RF相位=0 deg`
- `边带阶数=5`

按钮功能：

- `更新图像`：按当前输入和显示模式重新计算并绘图
- `重置参数`：恢复默认参数
- `保存PNG`：保存当前显示模式下的图像，也可在文件名中选择 SVG/PDF
- `导出CSV`：导出当前显示模式下每条谱线的数据

偏置输入区可以在 `偏压(V)` 和 `相位(deg)` 间切换。切换时 GUI 会按照当前 `Vpi` 自动互转数值；相位模式只改变 I/Q/P 静态偏置输入方式，不改变 `Q路RF相位` 的含义。

鼠标悬停到可见边带时，提示框会显示视图、边带阶数、频偏、真实相位、场幅 `|Ck|`、光功率 `|Ck|²`、相对功率 dB、复数实部/虚部。

CSV 字段包括：

```text
view, order, freq_offset_ghz, magnitude, power, magnitude_db, power_db, phase_deg, real, imag
```

## 测试

```powershell
python -m unittest discover
```

测试覆盖：

- 贝塞尔展开式逐项验证
- 上臂/下臂贡献相加等于合成结果
- 零偏压推挽结构下奇数阶上/下臂非零但合成后相消
- Q路RF相对相位按边带阶数旋转
- P路相移和最终耦合关系
- 偏压/相位输入换算等价性
- 臂分解视图和CSV导出
- 相位显示映射、像素箭头长度缩放、显示阈值和 CSV 导出字段
