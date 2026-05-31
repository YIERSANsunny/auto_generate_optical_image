# DPMZM虚拟物理模型指导文档

本文档用于指导和复核当前 DPMZM 虚拟模型的实现。它描述的是程序中的相对光谱模型，而不是完整器件标定模型。

## 1. 模型目标与适用范围

本模型用于根据 DPMZM 的 I/Q/P 三路静态偏置、I/Q 两路单音 RF 驱动和边带阶数，计算各阶光谱边带的复数场系数，并以相位矢量图显示。

模型关注：

- 相对光谱边带。
- 复数场系数 `Ck`。
- 线性场幅 `|Ck|`。
- 相对光功率 `|Ck|^2`。
- 真实物理相位 `phase_deg`。
- I/Q 子 MZM 上下臂贡献和合成结果。
- 可切换的理想/非理想光谱模型。
- 非理想模型中的插入损耗和有限消光比。
- Q 路经过 P 路后的相位旋转。
- I 路与 Q 路经过 P 路后的相干耦合。

模型不包含：

- 绝对光功率标定。
- 光载频绝对值。
- 噪声、偏振、带宽滚降。
- RF 任意波形导入。
- 臂不平衡、耦合器非 3 dB 分光等更复杂实验误差项。

当前版本默认使用单音 RF 和理想推挽子 MZM；非理想模型作为可选光谱域近似模型启用。

## 2. 坐标与符号约定

边带阶数记为 `k`：

```text
k = 0      载波
k = +1     +1 阶边带
k = -1     -1 阶边带
```

频偏只显示相对频偏，不显示绝对光载频：

```text
freq_offset_ghz = k * rf_frequency_ghz
```

复数场系数记为：

```text
Ck = real + j * imag
```

每条谱线输出：

```text
magnitude = |Ck|
power = |Ck|^2
phase_deg = angle(Ck), wrapped to [-180, 180)
```

相干耦合始终对复数场 `Ck` 做加法。功率只在复数场合成之后计算，不能先用功率或 dB 重建相位。

## 3. 输入参数

`DPMZMParams` 是模型输入结构。

```text
voltage_i           I 路子 MZM 静态偏置电压 VI
voltage_q           Q 路子 MZM 静态偏置电压 VQ
voltage_p           P 路主相位偏置电压 VP
vpi_i               I 路 Vpi
vpi_q               Q 路 Vpi
vpi_p               P 路 Vpi
rf_frequency_ghz    RF 频率，单位 GHz
rf_amplitude_i_v    I 路 RF 峰值电压
rf_amplitude_q_v    Q 路 RF 峰值电压
q_rf_phase_deg      Q 路 RF 相对 I 路的时间相位 psi
sideband_order      计算边带阶数 N，输出 k=-N...+N
use_nonideal        是否启用非理想光谱模型
extinction_ratio_i_db  I 路子 MZM 消光比
extinction_ratio_q_db  Q 路子 MZM 消光比
insertion_loss_i_db    I 路子 MZM 插入损耗
insertion_loss_q_db    Q 路子 MZM 插入损耗
insertion_loss_p_db    P 路公共相位块插入损耗，只作用于 Q 路
insertion_loss_global_db 最终耦合输出后的公共插入损耗
```

当前默认值：

```text
VI = 0 V
VQ = 0 V
VP = 0 V
VpiI = VpiQ = VpiP = 5 V
rf_frequency_ghz = 10 GHz
rf_amplitude_i_v = rf_amplitude_q_v = 0.4 V
q_rf_phase_deg = 0 deg
sideband_order = 5
use_nonideal = False
extinction_ratio_i_db = extinction_ratio_q_db = 30 dB
insertion_loss_i_db = insertion_loss_q_db = insertion_loss_p_db = 6 dB
insertion_loss_global_db = 0 dB
```

GUI 支持用相位直接设置静态偏置。相位输入和偏压输入完全等价，模型内部仍然使用等效电压：

```text
phase_deg = 180 * Vbias / Vpi
Vbias = phase_deg * Vpi / 180
```

注意：`q_rf_phase_deg` 是 Q 路 RF 驱动相对 I 路 RF 驱动的时间相位，不是 Q 子 MZM 的静态偏置相位。

## 4. I/Q 子 MZM 推挽模型

每个 I/Q 子 MZM 被建模为上下两臂相位调制后相干合成。静态偏置用一个差分偏置表示：

```text
delta = pi * Vbias / Vpi
phi1 = +delta / 2
phi2 = -delta / 2
```

其中：

- `phi1` 是上臂静态相位。
- `phi2` 是下臂静态相位。
- `delta` 是上下臂静态相位差。

RF 峰值电压换算为贝塞尔调制深度：

```text
m = pi * Vrf_peak / Vpi
```

当前模型采用单音 RF 的 `sin` 时间基准。对时域模型可写成：

```text
E_upper(theta) = 0.5 * exp(j * (phi1 + m * sin(theta + psi)))
E_lower(theta) = 0.5 * exp(j * (phi2 - m * sin(theta + psi)))
E_total(theta) = E_upper(theta) + E_lower(theta)
```

其中：

- `theta = omega_rf * t`
- `psi` 是 RF 相对相位。
- I 路使用 `psi = 0`。
- Q 路使用 `psi = q_rf_phase_deg`。

推挽的含义是上下臂 RF 相移方向相反。等价到贝塞尔展开时，子 MZM 内部两臂 RF 相位差固定为：

```text
phi = 180 deg
```

## 5. 贝塞尔边带展开

对每个边带阶数 `k`，先计算上臂贡献 `upper_k` 和下臂贡献 `lower_k`，再相干相加：

```text
total_k = upper_k + lower_k
```

0 阶：

```text
upper_0 = 0.5 * exp(j * phi1) * J0(m)
lower_0 = 0.5 * exp(j * phi2) * J0(m)
total_0 = upper_0 + lower_0
```

正 n 阶，`n > 0`：

```text
upper_+n = 0.5 * exp(j * phi1) * Jn(m) * exp(j * n * psi)
lower_+n = 0.5 * exp(j * (phi2 + n * phi)) * Jn(m) * exp(j * n * psi)
total_+n = upper_+n + lower_+n
```

负 n 阶，`n > 0`：

```text
upper_-n = 0.5 * (-1)^n * exp(j * phi1) * Jn(m) * exp(-j * n * psi)
lower_-n = 0.5 * (-1)^n * exp(j * (phi2 - n * phi)) * Jn(m) * exp(-j * n * psi)
total_-n = upper_-n + lower_-n
```

这里的 `(-1)^n` 来自负阶贝塞尔关系：

```text
J_-n(m) = (-1)^n * J_n(m)
```

因此负奇数阶会额外带来 180 deg 相位翻转。臂分解图中看到的 `+45 deg`、`-135 deg`、`-45 deg`、`+135 deg` 等组合，通常正是这个符号进入复数场相位后的结果。

## 6. 上下臂分解视图

`child_mzm_components(...)` 返回：

```text
upper: dict[int, complex]
lower: dict[int, complex]
total: dict[int, complex]
```

含义：

- `upper[k]` 是上臂对第 `k` 阶边带的复数场贡献。
- `lower[k]` 是下臂对第 `k` 阶边带的复数场贡献。
- `total[k] = upper[k] + lower[k]`。

上下臂贡献已经包含 `0.5` 系数。这个 `0.5` 表示理想子 MZM 内部合成关系下的场贡献比例，不需要在 `total` 中再次除以 2。

臂分解显示模式输出 6 个视图：

```text
I上臂, I下臂, I合成
Q上臂, Q下臂, Q合成
```

臂分解主要用于观察上下臂相量如何相长或相消。

## 7. DPMZM 组合关系

总览模式输出 4 个视图：

```text
I路
Q路
Q路经过P路
耦合输出
```

计算顺序如下。

I 路：

```text
I[k] = child_mzm(VI, VrfI, psi=0).total[k]
```

Q 路：

```text
Q[k] = child_mzm(VQ, VrfQ, psi=q_rf_phase).total[k]
```

P 路只作为 Q 路进入最终耦合前的相对相位控制：

```text
phiP = pi * VP / VpiP
Q_after_P[k] = exp(j * phiP) * Q[k]
```

最终耦合：

```text
Out[k] = (I[k] + Q_after_P[k]) / sqrt(2)
```

这个式子表示理想 3 dB 耦合输出的复数场叠加。注意：

- 耦合是逐阶 `k` 进行的。
- 只有相同阶数的 I/Q 边带互相耦合。
- 耦合使用复数场，不使用 `magnitude`、`power` 或 dB。
- 耦合后的相位由 `atan2(imag, real)` 得到。

## 8. 非理想光谱模型

`use_nonideal = False` 时，模型严格使用前面的理想贝塞尔边带系数，非理想参数不参与计算。

`use_nonideal = True` 时，I/Q 子 MZM 采用 VPI 对齐的有限消光比和插入损耗形式。插入损耗是功率损耗，进入复数光场时取平方根：

```text
L = 10^(-IL_dB/10)
sqrt(L) = 10^(-IL_dB/20)
```

有限消光比残余项：

```text
delta_er = 1 / (10^(ER_dB/20) - 1)
```

I/Q 子 MZM 的 NEGATIVE 推挽传输写成：

```text
H_neg = sqrt(L) * [cos(phi/2) + delta_er * exp(j*phi/2)] / (1 + delta_er)
```

在当前贝塞尔边带系数中，理想 `upper[k]` 已经包含 `0.5` 系数，所以 `exp(j*phi/2)` 对应 `2 * upper_ideal[k]`。因此：

```text
total_nonideal[k] =
    sqrt(L)/(1+delta_er) * (total_ideal[k] + 2*delta_er*upper_ideal[k])
```

臂分解中把有限消光比残余项归入上臂显示，以保持可视化中的相干相加关系：

```text
upper_nonideal[k] = sqrt(L)/(1+delta_er) * (1+2*delta_er) * upper_ideal[k]
lower_nonideal[k] = sqrt(L)/(1+delta_er) * lower_ideal[k]
total_nonideal[k] = upper_nonideal[k] + lower_nonideal[k]
```

这个 VPI 对齐形式满足两个边界：

```text
phi = 0     -> |H_neg| = sqrt(L)
phi = pi    -> |H_neg| = sqrt(L) * 10^(-ER_dB/20)
```

P 路按 POSITIVE 公共相位块处理，位于 Q 光路上：

```text
Q_after_P[k] = sqrt(L_P) * exp(j*phiP) * Q[k]
```

最终耦合输出在非理想模式下加入公共输出损耗：

```text
Out[k] = sqrt(L_global) * (I[k] + Q_after_P[k]) / sqrt(2)
```

注意：`insertion_loss_global_db` 只作用于 `耦合输出`，不反向缩放 I/Q 中间视图。

## 9. 场幅、功率与 dB

模型对每个复数系数 `Ck` 计算：

```text
real = Re(Ck)
imag = Im(Ck)
magnitude = |Ck|
power = |Ck|^2 = real^2 + imag^2
phase_deg = angle(Ck)
```

每次输出会在当前视图集合内做全局归一化。参考值为当前输出集合中最大的 `magnitude`：

```text
max_magnitude = max(|Ck|)
max_power = max_magnitude^2
```

相对 dB：

```text
magnitude_db = 20 * log10(magnitude / max_magnitude)
power_db = 10 * log10(power / max_power)
```

由于 `power = magnitude^2`，两者数值相同：

```text
magnitude_db == power_db
```

保留 `magnitude_db` 是为了兼容早期字段；从物理命名上看，建议把 dB 读作相对功率 dB。

极小值处理：

```text
if magnitude <= 1e-15: magnitude_db = -300
if power <= 1e-30: power_db = -300
```

## 10. 输出数据结构

`SpectralLine` 表示一条谱线：

```text
view              视图名
order             边带阶数 k
freq_offset_ghz   相对频偏 k * rf_frequency_ghz
magnitude         线性场幅 |Ck|
power             光功率比例 |Ck|^2
magnitude_db      相对 dB，兼容字段
power_db          相对功率 dB
phase_deg         真实复数相位，范围 [-180, 180)
real              Ck 实部
imag              Ck 虚部
```

CSV 字段与 `SpectralLine` 一致：

```text
view, order, freq_offset_ghz, magnitude, power, magnitude_db, power_db, phase_deg, real, imag
```

CSV 只导出真实物理相位 `phase_deg`，不导出视觉箭头角度。

## 11. 可视化规则

图面不把功率作为纵坐标。每个可见边带从横轴 `y=0` 出发画一个相位箭头。

箭头方向大部分按真实相位角绘制，但为了视觉区分，特殊处理：

```text
abs(phase_deg) <= 1 deg              -> 显示角度 +90 deg
abs(abs(phase_deg) - 180) <= 1 deg   -> 显示角度 -90 deg
其他相位                              -> 显示角度 phase_deg
```

因此：

```text
真实 0 deg       显示为向上
真实 +90 deg     显示为向上
真实 -90 deg     显示为向下
真实 +/-180 deg  显示为向下
真实 +45 deg     显示为 +45 deg 方向
```

箭头长度只做轻微视觉缩放：

```text
normalized_magnitude = 10^(magnitude_db / 20)
length_px = ARROW_BASE_LENGTH_PX * (0.75 + 0.25 * normalized_magnitude)
```

箭头长度不是严格功率坐标。鼠标悬停只显示三项核心信息：频偏、真实相位、相对光功率 `|Ck|^2` 与相对功率 dB。完整场幅、功率、复数实部/虚部以 CSV 为准。

相位颜色使用高对比循环色图：

```text
-180 deg / +180 deg  深红/紫红
-90 deg              蓝色
0 deg                深绿色
+90 deg              橙色
```

`-180 deg` 与 `+180 deg` 使用同色，保证相位闭环。相位文字带白色描边，以便在白底网格上保持可读。

低功率显示阈值：

```text
PHASE_LABEL_DB_THRESHOLD = -60 dB
```

低于阈值的边带不画箭头、不标相位、不进入 hover 目标，避免近零边带的随机相位误导。

## 12. 验证方法

当前测试覆盖以下物理一致性。

贝塞尔展开逐项验证：

```text
child_mzm_coefficients(...) 与手写贝塞尔公式逐阶比较。
```

时域采样傅里叶对照：

```text
E_upper(theta) = 0.5 * exp(j * (phi1 + m * sin(theta + psi)))
E_lower(theta) = 0.5 * exp(j * (phi2 - m * sin(theta + psi)))
Ck = mean(E(theta) * exp(-j * k * theta))
```

该采样结果与 `child_mzm_components(...)` 的 `upper/lower/total` 逐阶一致。

推挽奇数阶相消：

```text
Vbias = 0 且 phi = 180 deg 时，奇数阶 upper/lower 各自非零，但 total 近似为 0。
```

Q 路 RF 相位：

```text
psi 改变时，第 k 阶边带乘以 exp(j * k * psi)。
```

P 路相移：

```text
VP 增加一个 VpiP 时，Q_after_P 整体相移约 180 deg。
```

耦合关系：

```text
Out[k] == (I[k] + exp(j * phiP) * Q[k]) / sqrt(2)
```

功率字段：

```text
power == real^2 + imag^2 == magnitude^2
power_db == 10 * log10(power / max_power)
```

非理想模型：

```text
use_nonideal = False 时输出与理想模型完全一致。
IL=0 且 ER 很大时，非理想模型近似理想模型。
bias=0, RF=0 时，子 MZM 场幅为 sqrt(L)。
bias=Vpi, RF=0 时，子 MZM 残余场幅为 sqrt(L)*10^(-ER/20)。
P 路插损只缩放 Q_after_P 和最终耦合中的 Q 分量。
非理想臂分解仍满足 upper + lower = total。
```

偏压/相位输入等价：

```text
phase_deg = 180 * Vbias / Vpi
```

用相位输入换算得到的等效偏压，与直接输入偏压得到的光谱一致。

## 13. 实现边界与后续扩展

当前模型是相对、单音 RF 光谱模型。非理想模式已经包含 I/Q 子 MZM 插入损耗、有限消光比和 P 路插入损耗，但仍不是完整实验链路。若后续要贴近实验，需要单独引入并标定：

- 输入光功率和绝对功率单位。
- 耦合器真实分光比。
- 上下臂幅度不平衡。
- RF 幅频响应。
- 光电探测器响应。
- 噪声和测量底噪。
- 多音或任意波形 RF。

这些扩展不应直接混入当前理想公式，建议通过新增参数和独立测试逐步引入。
