# ComfyUI Turbo / Base Trajectory Mixer

在同一次采样中混合 **Base 与 Turbo LoRA 轨迹**，支持四种构图 / 风格组合、轨迹比例调节和 LoRA 强度平滑。

节点接在模型加载器后，输出 `MODEL`，继续使用普通 KSampler、KSampler Advanced 或 CFGGuider + SamplerCustomAdvanced。普通 LoRA 使用 ComfyUI 原生 bypass 动态控制强度。

[English quick guide](#english)

## 快速开始

### 1. 安装

在 ComfyUI 的 `custom_nodes` 目录执行：

```sh
git clone https://github.com/chinokikiss/ComfyUI-Turbo-Base-Trajectory-Mixer.git
```

需要包含 V3 DynamicCombo、原生 LoRA bypass、ModelPatcher hooks / wrappers 的 ComfyUI；没有额外依赖。

### 2. 接线

重启 ComfyUI，添加 **Turbo / Base Trajectory Mixer**，分类为 `model/sampling`。

```text
CheckpointLoaderSimple / UNETLoader
                │ MODEL
                ▼
Turbo / Base Trajectory Mixer
                │ MODEL
                ▼
KSampler / KSampler Advanced / CFGGuider
```

把 Turbo LoRA 放在 `models/loras`，在本节点的 `turbo_lora` 中选择。CLIP、提示词、VAE 和 latent 按通常的工作流接线。

**同一份 Turbo LoRA 只在本节点中加载。** 如果提前用普通 LoRA Loader 加载它，Base 段也会包含这份 LoRA。其他风格 LoRA 可以接在本节点前。

### 3. 选择轨迹

默认选择 **Base 构图 / Base 风格**。以 12 步、比例均为 0.5 为例，轨迹为 `B3 → L6 → B3`：前 3 步 Base，中间 6 步 Turbo，最后 3 步 Base。

- `B`：输入模型，包含本节点前已加载的其他 LoRA。
- `L`：输入模型加上本节点选择的 Turbo LoRA。

## 四种方案与比例

| 构图 / 风格 | 12 步默认轨迹 | 可调比例 |
|---|---|---|
| Base / Base | `B3 → L6 → B3` | `prefix_ratio`、`suffix_ratio` |
| Base / Turbo | `B6 → L6` | `composition_ratio` |
| Turbo / Base | `L6 → B6` | `composition_ratio` |
| Turbo / Turbo | `L3 → B6 → L3` | `prefix_ratio`、`suffix_ratio` |

所有比例默认 **0.5**，范围为 **0–1**。

### 构图与风格不同

`composition_ratio` 决定前缀占**全部有效步数**的比例。前缀使用构图来源，剩余后缀使用风格来源。

例如 Base / Turbo、12 步、比例 0.25，得到 `B3 → L9`。

### 构图与风格相同

先把有效步数分成前半与后半：

- `prefix_ratio`：前半步数中，有多少比例作为构图前缀。
- `suffix_ratio`：后半步数中，有多少比例作为风格后缀。
- 前缀和后缀使用所选来源，未分配的中间步数使用另一来源。

例如 Base / Base、12 步，`prefix_ratio=0.33`、`suffix_ratio=0.67`，得到 `B2 → L6 → B4`。两个比例均为 1 时全程使用所选来源，均为 0 时全程使用另一来源。

步数取最近整数，恰好半步向上取整；奇数总步数的前半多一步，例如 5 步分为 3 + 2。比例为 0 时允许对应段消失。

**“前半决定构图、后半决定风格”是实验假设。** 方案标签描述采样意图，不能保证视觉上独立分离；图像质量由用户比较判断。

## 参数

| 参数 | 默认值 | 作用 |
|---|---|---|
| `model` | — | Base 输入模型 |
| `turbo_lora` | — | 要动态控制的 Turbo LoRA 文件 |
| `lora_strength` | `1.0` | Turbo 段的 LoRA 强度，过渡区乘以渐变权重 |
| `turbo_cfg` | `1.0` | 标准 CFG guider 的 Turbo 段 CFG；Base 段沿用采样器的 CFG |
| `scheme` | Base / Base | 选择构图与风格来源，并显示对应比例滑槽 |
| `lora_smooth_steps` | `0.0` | 每个切换点附近的 LoRA 强度过渡宽度，单位为有效采样步数 |

### 强度平滑

`lora_smooth_steps=0` 为硬切换。大于 0 时，在切换点两侧用 smoothstep S 曲线渐变，实际强度为 `lora_strength × 渐变权重`。

例如 12 步 `B3 → L6 → B3`、平滑值 2，Euler 各步的渐变权重为：

```text
0, 0, 0, 0.5, 1, 1, 1, 1, 1, 0.5, 0, 0
```

窗口以切换点为中心，宽度受两侧段长限制，避免相邻窗口重叠。全程 Base 或全程 Turbo 时没有过渡。平滑只调节 LoRA 强度，CFG 仍按原划分切换，也不会增加 denoiser 调用次数。旧工作流省略此参数时使用默认值 0。

### 调度规则

比例按**本次实际执行的 sigma 区间数**计算。图生图的 denoise、Advanced 的 start/end 切片会按切片后的有效步数重新分配。

Heun 等采样器的额外查询按 sigma 选择轨迹，并在相邻 sigma 节点间插值得到平滑位置；边界 sigma 使用下一段。采样器保留原来的 sigma 序列和多步历史，切换处不重新加噪或重启。因此结果可能与分段重启采样不同。

## 兼容性与实现

Base 与 Turbo LoRA 需要共用兼容的预测类型和采样设置。CLIP 权重不切换。

| 情况 | 使用的路径 |
|---|---|
| 普通 LoRA；标准 Linear，或 groups=1、`padding_mode="zeros"` 的 Conv1d/2d/3d | 原生 LoRA bypass |
| 包含 DoRA、LoCon mid、reshape、非标准层 / patch，或模型已有其他 injections | 整份 Turbo LoRA 使用原生权重 hook |
| DynamicVRAM 输入 | 使用宿主的 `get_non_dynamic_delegate()` 委托模型 |

**Bypass** 经 ModelPatcher injection 计算 `Wx + s × B(Ax)`，包含 alpha/rank 缩放。强度变化只改 `s`，强度为 0 时跳过 LoRA 分支。低秩权重副本只存在于本次采样中；正常结束或异常时恢复层的前向并释放副本，不缓存完整 ΔW。

**权重 hook** 保留原有 conditioning hooks，并使用 `MinVram` 模式。这个路径在强度变化时仍需重新合并权重。bypass 与合并权重的低精度舍入顺序不同，输出不保证逐位相同。

CPU / CUDA 已验证，多 GPU 做过 CPU 模拟验证。ROCm、MPS、DirectML、XPU、NPU 未做硬件验证。第三方量化 patcher、编译模型、跨模型 guider 和跳过 denoiser 的缓存插件未验证；自定义 guider 的独立 guidance 公式仍由该 guider 控制。创建 DynamicVRAM 委托可能增加加载时间和 CPU 内存。

## 示例与验证

[Anima BLB API 工作流](examples/anima_blb_api.json)：512×512、seed 0、12 步 Euler/simple、Base CFG 4、Turbo CFG 1、平滑值 0。使用前按本机文件名修改模型选择。

### 本机性能对比

2026-10-02 测量：RTX 3050 Laptop 6GB、PyTorch 2.11.0+cu128、Anima Base + Turbo LoRA v0.2；512×512、12 步 `B3 → L6 → B3`、`lora_smooth_steps=4`，DynamicVRAM 输入转原生委托。

| 路径 | 采样耗时 | 全程耗时（含加载、编码和解码） |
|---|---:|---:|
| 原权重 hook | 27.1 秒 | 34.1 秒 |
| LoRA bypass | **12.9 秒** | **19.4 秒** |

bypass 的 GPU 低秩权重约 **141.9 MiB**，完整 LoRA 权重合并次数为 0。两条路径均为 18 次模型前向；输出数值有限，清理检查通过。这是单次本机对比，不代表其他模型的性能或图像质量。

### 运行测试

在 ComfyUI 根目录使用其 Python 环境执行：

```sh
python custom_nodes/ComfyUI-Turbo-Base-Trajectory-Mixer/tests/test_mixer.py
```

已通过 22 项测试，覆盖四种方案、比例与有效区间、EPS / V prediction / flow、Euler / Heun / DPM++ 2M、平滑与 sigma 重访、既有 LoRA / hooks 叠加、bypass 数值对照、卷积、重复执行、异常清理和 V3 节点注册。

## English

Connect **Turbo / Base Trajectory Mixer** between your model loader and sampler. Select the Turbo LoRA in this node; other style LoRAs can precede it. Use the sampler's CFG for Base and `turbo_cfg` for Turbo. CLIP is unchanged.

The four composition/style modes are Base/Base (`B→L→B`), Base/Turbo (`B→L`), Turbo/Base (`L→B`), and Turbo/Turbo (`L→B→L`). Different sources expose a composition fraction of all active steps. Matching sources expose a prefix fraction of the first half and a suffix fraction of the second half; the middle uses the other source. Composition/style separation is an experimental hypothesis.

`lora_smooth_steps` defaults to **0** for hard switching. Positive values fade LoRA strength around each switch; CFG keeps its segment settings. Ratios follow the active schedule, including denoise and start/end slicing.

Ordinary LoRAs use native bypass, `Wx + s * B(Ax)`, with zero strength skipping the LoRA branch. Special adapters or models with existing injections use native weight hooks. Device copies are released after sampling or on error. DynamicVRAM inputs use the host's non-dynamic delegate. Requires ComfyUI with V3 DynamicCombo, native bypass, and ModelPatcher hooks / wrappers; no extra dependencies.
