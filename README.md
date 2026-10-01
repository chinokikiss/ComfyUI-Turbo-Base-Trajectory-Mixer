# ComfyUI Turbo / Base Trajectory Mixer

一个接在模型加载器后的 `MODEL → MODEL` 节点，用 ComfyUI 原生 LoRA bypass 和权重 hook 在同一次采样中动态控制 Turbo LoRA。继续使用普通 KSampler、KSampler Advanced 或 CFGGuider + SamplerCustomAdvanced。

不拟合 phase 映射，不修改 sigma 序列，不重新加噪，不重启采样器。`B` 表示输入模型，`L` 表示输入模型加上选定的 Turbo LoRA。

## 安装与接线

在 ComfyUI 的 `custom_nodes` 目录执行：

```sh
git clone https://github.com/chinokikiss/ComfyUI-Turbo-Base-Trajectory-Mixer.git
```

重启 ComfyUI，添加 **Turbo / Base Trajectory Mixer**（分类 `model/sampling`）。需要支持 V3 DynamicCombo、ModelPatcher wrappers 和原生权重 hooks 的新版 ComfyUI；没有额外依赖。

```text
CheckpointLoaderSimple / UNETLoader
                │ MODEL
                ▼
Turbo / Base Trajectory Mixer
                │ MODEL
                ▼
KSampler / KSampler Advanced / CFGGuider
```

CLIP、正负提示词、VAE、latent 按原工作流接线。Turbo LoRA 放在 `models/loras`，在本节点内选择；不要预先用普通 LoRA Loader 加载同一份 Turbo LoRA，否则 Base 段也会包含它。其他风格 LoRA 可以接在本节点前。

## 四种方案

以下为 **12 步、所有可见比例默认 0.5** 的示例：

| scheme（构图 / 风格） | 轨迹 | 显示的滑槽 |
|---|---|---|
| Base / Base | `B3 → L6 → B3` | `prefix_ratio`、`suffix_ratio` |
| Base / Turbo | `B6 → L6` | `composition_ratio` |
| Turbo / Base | `L6 → B6` | `composition_ratio` |
| Turbo / Turbo | `L3 → B6 → L3` | `prefix_ratio`、`suffix_ratio` |

- **构图与风格不同**：`composition_ratio` 是全部有效采样步中前缀所占比例，前缀使用构图来源，剩余后缀使用风格来源。
- **构图与风格相同**：`prefix_ratio` 是前半步数中前缀所占比例；`suffix_ratio` 是后半步数中后缀所占比例。前缀、后缀使用所选来源，中间使用另一来源。两个滑槽的 1.0 都表示**对应半段的全部**。
- 步数取最近整数，恰好半步向上取整。奇数总步数的前半多一步，例如 5 步分成 3 + 2。0 允许某段消失；两个滑槽均为 1 时全程使用所选来源，均为 0 时全程使用另一来源。
- Base / Base、12 步时，`prefix_ratio=0.33`、`suffix_ratio=0.67` 对应 `B2 → L6 → B4`。

“前半决定构图、后半决定风格”是实验假设。标签描述采样意图，不能保证构图与风格在视觉上独立。最终质量由用户比较图像决定。

## 参数与调度

- `lora_strength`：Turbo 段的扩散模型 LoRA 强度，过渡区乘以渐变权重；CLIP 权重不切换。
- `lora_smooth_steps`：每个切换点附近的 LoRA 强度过渡宽度，单位为有效采样步数，默认 0。0 保留硬切换；例如 2 表示在切换点前后共 2 步内，用 smoothstep S 曲线渐变。实际强度为 `lora_strength × 渐变权重`，权重在 0 与 1 之间变化。过渡窗口以切换点为中心，最大宽度受两侧段长限制，避免相邻窗口重叠；纯 Base / Turbo 轨迹不产生渐变。旧工作流省略此参数时沿用硬切换。
- `turbo_cfg`：标准 CFG guider 的 Turbo 段 CFG，默认 1；Base 段沿用 KSampler / CFGGuider 的 CFG。自定义 guider 若使用自己的 guidance 公式或独立字段，其 guidance 仍由该 guider 决定。
- 比例依据**本次实际执行的 sigma 区间数**。图生图、Advanced 的 start/end 切片会按切片后的有效步数重新分配；不是整个原始时间轴的绝对百分比。
- Heun 等采样器的额外模型查询按查询 sigma 选择轨迹，边界 sigma 使用下一段；不会按模型调用次数误计采样步数。多步采样器保留其原有历史，切换处不清空历史，因此不会承诺与分段重启采样得到相同结果。

平滑仅调节 LoRA 强度，CFG 仍按原划分切换。例如 12 步 `B3→L6→B3`、`lora_smooth_steps=2` 时，Euler 各步 LoRA 权重为 `0, 0, 0, 0.5, 1, 1, 1, 1, 1, 0.5, 0, 0`。Heun 等额外查询在相邻 sigma 节点之间插值得到连续步进位置。平滑不会增加 denoiser 调用次数。

## 动态 LoRA 与兼容范围

LoRA 文件在节点执行时读取一次，通过原生 LoRA key mapping/conversion 匹配模型。普通 LoRA 使用原生 `BypassForwardHook`，经 ModelPatcher injection 在层的前向中计算 `Wx + s × B(Ax)`（含 alpha/rank）；强度变化只修改 `s`，强度为 0 时直接执行原层。硬切换和平滑均使用 bypass，不生成或缓存完整 ΔW，不把 Turbo LoRA 静态合并到输入模型。

低秩权重在本次采样中按 patcher 的加载设备放置，原始 CPU adapter 不变。正常结束、异常或部分注入失败时，恢复层的前向、移除本节点 injection，并释放设备副本。多 GPU 副本分别使用自己的模型和加载设备；没有跨采样的 GPU 缓存。

当整份 Turbo LoRA 包含 DoRA、LoCon mid、reshape、非标准层/patch，或模型已有其他 injections 时，保留原生权重 hook 路径。bypass 支持标准 Linear，以及 groups=1、zero padding 的 Conv1d/2d/3d。特殊路径的强度变化仍会重新合并权重。原有 conditioning hooks 保留并组合；权重 hook 使用原生 `MinVram` 模式。bypass 与合并权重的低精度舍入顺序不同，不保证逐位相同的输出。

支持具有原生 ModelPatcher / LoRA 权重 hook 接口的模型，不绑定 Anima、flow latent 或特定网络结构。Base 和 Turbo LoRA 必须能够共用相同的预测类型与采样设置；本节点不会把要求不同采样参数的 LoRA 自动转换成兼容模型。第三方量化 patcher、编译模型、跨模型 guider、跳过 denoiser 的缓存插件未验证。

当前宿主的 `ModelPatcherDynamic` 尚未实现权重 hook；输入为动态显存 patcher 时，使用 ComfyUI 自带的 `get_non_dynamic_delegate()`，与宿主处理 conditioning 权重 hooks 的方式一致。GPU 上仍只采样一个委托模型，但创建委托可能增加加载时间和 CPU 内存。CPU / 普通 CUDA patcher 已测试；ROCm、MPS、DirectML、XPU、NPU 沿用宿主设备逻辑，未进行硬件验证。

## 示例与验证

`examples/anima_blb_api.json` 是 512×512、seed 0、12 步 Euler/simple、Base CFG 4、Turbo CFG 1 的 API 工作流。先按本机文件名修改模型选择。

在 ComfyUI 根目录使用它的 Python 环境运行：

```sh
python custom_nodes/ComfyUI-Turbo-Base-Trajectory-Mixer/tests/test_mixer.py
```

测试使用小型真实 ModelPatcher / CFGGuider 和原生 LoRA adapter，检查四方案、奇数与端点、有效区间切片、重复执行、EPS / V prediction / flow、Euler / Heun / DPM++ 2M、既有 LoRA 叠加、mask、异常清理、V3 注册及条件滑槽，以及平滑强度、过渡连续性、短段、额外 sigma 查询和反向重访。bypass 测试检查零强度跳过、权重不合并、alpha/负强度与原生合并的数值对照、卷积 stride/dilation、DoRA 与其他 injections 的回退、部分注入失败及多模型清理。

本机验证（2026-10-02）：22 项测试通过，示例 API 工作流通过宿主验证。RTX 3050 Laptop 6GB、PyTorch 2.11.0+cu128、Anima Base + Turbo LoRA v0.2、512×512、12 步 `B3→L6→B3`，旧权重 hook 路径硬切换时普通 KSampler 采样约 14.5–15.0 秒；包含加载、编码和解码约 22 秒。低显存拆开 CFG 正负分支后共 18 次前向，没有额外搜索调用。时间仅代表这台机器。

相同设置、`lora_smooth_steps=4` 时，窗口受两侧段长限制为 3 步，实际权重为 `0, 0, 0.074, 0.5, 0.926, 1, 1, 1, 0.926, 0.5, 0.074, 0`。真实 DynamicVRAM 输入转原生委托后，bypass 实测采样约 **12.9 秒**、全程约 **19.4 秒**；旧权重 hook 路径分别约 27.1 / 34.1 秒。GPU 上低秩权重约 **141.9 MiB**，完整 LoRA 权重合并次数为 0，仍为 18 次模型前向。输出数值有限，权重 hook 与 bypass 均清理正常。多 GPU 仅做 CPU 模拟验证。质量由用户判定。

## English

Connect **Turbo / Base Trajectory Mixer** after a model loader and send its MODEL output to your existing sampler. Ordinary Turbo LoRAs use native ComfyUI bypass through ModelPatcher injections: `Wx + s * B(Ax)`. Strength changes update `s`; zero strength skips the LoRA branch. CLIP is unchanged. Use the sampler's CFG for Base and `turbo_cfg` for Turbo with standard CFG guiders.

The four modes are Base/Base (`B→L→B`), Base/Turbo (`B→L`), Turbo/Base (`L→B`), and Turbo/Turbo (`L→B→L`). Different sources expose one composition fraction of all active steps. Matching sources expose a prefix fraction of the first half and a suffix fraction of the second half; the middle uses the other source. Ratios follow the active sampler schedule, including denoise/start/end slicing. No phase mapping, sigma replacement, latent conversion, extra noise, or sampler restart is introduced. Composition/style separation is a hypothesis, not a quality guarantee.

`lora_smooth_steps` controls the smoothstep transition width around each switch in active sampling steps. Default 0 preserves hard switching and existing workflows; try a value such as 2 for a two-step transition. The width is limited by adjacent segment lengths to prevent overlap. LoRA strength fades between 0 and `lora_strength`; CFG retains its segment settings. Intermediate sigma queries interpolate the step position. No extra denoiser evaluations are added.

Device copies and injections live only for the sampling call and are released on completion or error. Original CPU adapters remain unchanged. No full delta matrices are cached. If the Turbo adapter includes DoRA, mid/reshape weights, unsupported layers/patches, or the model already has other injections, the whole Turbo adapter uses native weight hooks instead. Low-precision bypass and weight merging can differ due to rounding order.

Requires modern ComfyUI with V3 DynamicCombo, native bypass, and ModelPatcher hooks/wrappers. Dynamic-VRAM inputs retain the host's non-dynamic delegate for compatibility with weight hooks. Model/LoRA pairs must share compatible prediction and sampling settings. Nonstandard guiders and third-party quantized patchers have not been verified.
