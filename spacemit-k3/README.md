# 端侧文生图 · SpacemiT K3

在 **SpacemiT K3 Pico-ITX**（Bianbu 4.0.7 / riscv64）上部署的文生图服务。
全部文件都在同一个目录：`/home/bianbu/image-generation/`。

实测：**512×512，SD-Turbo，1 步约 102 秒，4 步约 188 秒**，输出质量与 diffusers 参考实现一致。

> **关于本仓库里的路径**：设备上所有文件是**扁平**摊在一个目录里的；仓库里为了与 Jetson 那套分开，把这个项目收进了 `spacemit-k3/` 子目录，于是 `imagegen/` 变成 `spacemit-k3/runtime/imagegen/`、`generate.py` 变成 `spacemit-k3/runtime/generate.py`。
> 代码里的相对路径（`models/gguf/...`、`output/`）都是按**设备上**的布局写的，把 `spacemit-k3/` 下的内容拷回设备根目录即可原样运行，不需要改。
> `export/`、`tools/` 是 PC 侧用的，不参与设备上的运行。
>
> 权重与 ONNX 导出包不在仓库里（可重新下载 / 重新导出），见上一级的 **[MANIFEST.md](../MANIFEST.md)**。

---

## 快速开始

```bash
cd /home/bianbu/image-generation

# 出图（默认 ONNX 引擎，SD-Turbo，4 步）
python3 generate.py -p "a red fox sitting in a snowy forest, soft morning light" -o output/fox.png
python3 generate.py -p "..." -s 1          # 1 步：约 102 秒

# Web UI + HTTP API
python3 server.py                          # 浏览器打开 http://<设备IP>:8080

# GGUF 引擎（stable-diffusion.cpp）
python3 generate.py -e gguf -m models/gguf/sd15-q8.gguf -p "..." -s 8

# 后端体检
python3 validate.py                        # 逐组件与 PyTorch 参考张量比对
./run.sh devices                           # 列出可用后端与模型
```

`run.sh` 是便捷入口：

```bash
./run.sh generate "一只在雪地森林里的红狐狸"   # ONNX 引擎
./run.sh generate-cpu "..."                    # GGUF 引擎
./run.sh serve 8080                            # 起服务
./run.sh bench                                 # 逐组件计时
```

---

## 硬件侦查结论（决定了整个方案）

设备：16 核（8× `spacemit,a100` @1.8GHz + 8× `spacemit,x100` @2.15GHz），15GiB 内存，无 swap。
ISA：`rv64imafdcvh` + RVV 1.0（`zve64d` / `zvfh` / `zvfbfwma` / `zvkned`）。

**Linux 只能用 0–7 号核（X100）。** PID 1 自身就是 `Cpus_allowed_list: 0-7`，
连 root 都无法把亲和性扩到 8–15，`isolcpus` 也没开。A100 不是通过 Linux 调度器使用的，
而是由 SPERT / Spine 运行时当作协处理器驱动，入口是 `/dev/ai_dma`。

### A100 有两条入口，只有一条通

**① ggml / SPERT 路径 —— 不通。**
Bianbu 预装的 `llama.cpp-tools-spacemit` 带一套打过补丁的 ggml，注册了 `CPU_RISCV64_SPACEMIT`
后端，自报 `num_perfer_cores: 8, use_ime2: 1, cpu_mask: ff00`（即想用 8–15 号 A100 核）。
实测拿不到收益：TCM 初始化失败（`open(/dev/tcm_sync_mem) failed`，退化为 heap），
且进程被限制在 0–7 核，`ff00` 掩码落空。
`llama-bench`（Qwen3-0.6B Q4_0，8 线程）只有 **pp64 444 t/s、tg32 34 t/s**，就是 8 个 X100 核的水平。

> 注：该包里的 `llama-diffusion-cli` 是 **LLaDA 文本扩散**示例，不是 stable-diffusion.cpp。
> 把 `ggml/src/ggml-cpu/spacemit/`（约 1.5 万行）移植进 stable-diffusion.cpp 也走不通：
> sd.cpp 依赖自家 `leejet/ggml` fork 的私有算子（`ggml_mul_mat_i8_tensorwise`），
> 与系统 ggml 0.16.0 不兼容，`-DSD_USE_SYSTEM_GGML=ON` 编译直接挂在
> `static_assert(GGML_MAX_NAME >= 160)` 与私有算子缺失上。

**② ONNX Runtime + SpaceMiT EP —— 通路存在，但 Python 绑定不可靠。**

`import spacemi_ort` 之后，`providers=["SpaceMITExecutionProvider"]` 会把算子编译到 A100，
不支持的算子自动回落 CPU。纯前馈 CNN 上效果很好，这些都是 Python 路径测出来的：

| 模型（官方 model zoo） | CPU | NPU | 加速比 |
|---|---|---|---|
| MobileNetV3-Small fp16 | 23.10 ms | 2.71 ms | **8.5×** |
| YOLO26n fp16（Conv/SiLU/Concat/Resize） | 530 ms | 14.1 ms | **37.5×** |
| PPLiteSeg | 3495 ms | 667 ms | **5.2×** |

**但要小心：Python 这条路径会不稳定地段错误，而且同一个脚本、同一个模型，时好时坏。**
今天的实测：

- 11:47 `npu_test.py` 测 mobilenet → 正常，2.85 ms
- 15:03 同一个脚本再跑 → 正常，2.85 ms
- 15:20 去掉所有会话选项的等价脚本 → **SIGSEGV**

排除过的因素：`provider_options`（传不传、传空的都崩，但**最初能跑通的脚本恰好没传**）、
`intra_op_num_threads`（default/1/2/4/8 全崩）、opset 17 vs 20、ONNX external data、
关闭图优化、算子过滤（崩溃发生在 EP 认领算子之前，过滤来不及生效）。

**同一时刻的对照是关键：**

| 通道 | 同一模型、同一时刻 | 结果 |
|---|---|---|
| SpacemiT 官方 C++ `run_demo` | mobilenet_v3_small.fp16.onnx | **正常**，init 63ms / 推理 8.67ms |
| 本仓库的 C++ 探针 `npu_probe.cc` | 同上 | **正常**，init 64.7ms / 推理 8.53ms |
| 官方 `onnxruntime_perf_test -e spacemit` | 同上 | EP 加载成功（只因缺测试输入退出） |
| Python `spacemi_ort` wheel | 同上 | **SIGSEGV** |

**所以 A100 和 EP 都是好的，坏的是 Python 绑定层。** 走 C++ 通道，SD 的组件也能建会话：

```
./npu_probe models/onnx/sd-turbo/vae_decoder.onnx 3
[1] model  : models/onnx/sd-turbo/vae_decoder.onnx
[2] init   : 111.2 ms          <-- 无段错误
[3] input  : latent shape=[1,4,64,64,] elem_type=1
[4] run ...                    <-- 编译+执行很慢（>8 分钟未出结果，已中止）
```

**待办（未完成）**：VAE 首次推理的耗时没测出来（EP 编译这一算子图非常慢）。
在测完 UNet 和文本编码器之前，不能断言「NPU 能加速扩散模型」——但同样**也不能再断言它不能**。

构建探针：

```bash
P=/tmp/spacemit-ort.riscv64.2.0.2          # 来自 archive.spacemit.com/spacemit-ai/onnxruntime/
g++ -O2 -fPIC -std=c++17 npu_probe.cc -o npu_probe     -I$P/include -L/usr/lib -lonnxruntime -lspacemit_ep
```

注意 `SessionOptionsSpaceMITEnvInit` 在 `libspacemit_ep.so` 里（不在 libonnxruntime），
且必须 `-fPIC`；官方 `run_demo` 链接的就是系统库。

---

## A100 加速：能不能吃到？（实测结论）

**能吃到，但扩散模型吃不到 —— 卡在 SpacemiT 的 kernel 覆盖面上。**

### 先确认 A100 后端到底有多快

用同一台设备、同一模型，对比上游原版 llama.cpp 与 Bianbu 预装的 SpacemiT 版：

| 模型 | 测试 | 上游 llama.cpp | SpacemiT 版 | 加速比 |
|---|---|---|---|---|
| Qwen3-4B Q4_K_M | pp64 | 14.89 t/s | **62.22 t/s** | **4.2×** |
| Qwen3-4B Q4_K_M | tg32 | 3.60 t/s | **8.28 t/s** | **2.3×** |
| Qwen3-0.6B Q4_0 | pp64 | 119.87 t/s | **444 t/s** | **3.7×** |
| Qwen3-0.6B Q4_0 | tg32 | 19.30 t/s | **48.81 t/s** | **2.5×** |

**SpacemiT 的 ggml 补丁带来 2.3–4.2 倍加速**（`CPU_RISCV64_SPACEMIT` 后端，IME2 + SPERT）。
上游基线只有 ~120 t/s（约 143 GFLOPS），说明 8 个 X100 核本身到不了 444 t/s —— 这个加速是真实存在的。

### 但 llama.cpp 跑不了扩散模型

`llama.cpp-tools-spacemit` 是 **LLM 推理引擎**，只实现了 LLM 架构。实测：

```
$ llama-cli -m z_image_turbo-Q4_K_M.gguf -p test -n 4
E llama_model_load_from_file_impl: failed to load model        # 34 毫秒就退出
```

GGUF 只是容器格式，llama.cpp 和 stable-diffusion.cpp 都用它，但里面装的东西完全不同：
扩散模型还需要**噪声调度器、去噪循环、VAE 解码**，llama.cpp 三样都没有。
（包里那个 `llama-diffusion-cli` 是 **LLaDA 文本扩散**，生成文字不是图片。）

### 真正的路径：让 stable-diffusion.cpp 用 SpacemiT 的 ggml

sd.cpp 用的就是 ggml，所以把它的 ggml 换成 SpacemiT 那套即可。这条路**已经打通到编译成功**：

1. 系统 `libggml.so.0.16.0` 就是带 A100 后端的版本，但它的 `GGML_MAX_NAME=64`，
   而 sd.cpp 要求 ≥160 —— **ABI 不兼容**，不能直接链。
2. 从 `spacemit-com/llama.cpp` 源码重编 ggml，关键是：

   ```bash
   cmake -B build-ggml -G Ninja -DCMAKE_BUILD_TYPE=Release      -DGGML_MAX_NAME=160 -DGGML_CPU_RISCV64_SPACEMIT=ON      -DSPERT_DIR=<含 include/,lib/ 的目录>      -DGGML_RVV=ON -DGGML_RV_ZFH=ON -DGGML_RV_ZVFH=ON      -DGGML_RV_ZBA=ON -DGGML_RV_ZICBOP=ON -DGGML_RV_ZVFBFWMA=ON
   ```

   注意：`GGML_RV_*` 这些选项是必须的 —— ggml-cpu 自己拼 `-march` 串并覆盖全局 flag，
   只靠 `CMAKE_CXX_FLAGS=-march=...` 不生效；`-march=native` 这个 binutils 也不认。
   最终 march 串会是 `rv64gcv_zfh_zvfh_zvfbfwma_zicbop_zihintpause_zba_xsmtvdotii`。
3. sd.cpp 与上游 ggml 的差距**只有 5 项**（用 `cmake --build -- -k 0` 全量收集得到）：

   | 缺失项 | 处理 |
   |---|---|
   | `ggml_mul_mat_i8_tensorwise` | 打桩（只在 sd.cpp 的 int8-tensorwise 量化路径上调用，Q8_0/fp16 不走） |
   | `ggml_quantize_i8_convrot` | 同上 |
   | `GGML_TYPE_F8_E4M3` / `F8_E5M2` | 0.16 的枚举正好停在 43，43/44 空着，直接加 |
   | `GGML_MAX_NAME >= 160` | 编译宏，两边都要给 |

   补丁脚本：`patch_ggml_shims.py`；cmake 桥：`src/ggml-a100/ggml-config.cmake`。

4. **编译成功**，`sd-cli` 正确链接到 `build-ggml/bin/libggml*.so` + `libspert.so.1`，
   启动时打出 `CPU_RISCV64_SPACEMIT: ... use_ime2: 1` 横幅。

### 但一跑扩散就 abort

```
spacemit_kernels::rvv::forward_binary<GGML_OP_ADD, float>
  → ggml_abort("fatal error")        # rvv_kernels.cpp 的形状分派 else 分支
```

SpacemiT 的 RVV 内核是**为 LLM 图写的**（1D/2D、连续张量）。扩散图的典型形态是
4D NCHW + 广播加法（把 timestep embedding 加到特征图上），落进未支持分支后
它选择 `GGML_ABORT` 而**不是回退到通用内核**。这套后端没有「不支持就降级」的机制。

**所以结论是：A100 加速是真的（LLM 上 2.3–4.2 倍，已实测），链路也打通到能编译，
但 SpacemiT 的 kernel 覆盖面不含扩散模型的张量形态，而且失败方式是硬 abort。
要让扩散模型吃到这个加速，需要 SpacemiT 侧补齐 NCHW/广播类算子 —— 不是我们这边能补的。**

## 试过但不行的：Z-Image-Turbo（6B，2025）

你指定的 `jayn7/Z-Image-Turbo-GGUF` 的 Q4_K_M。三份文件都下齐了（主模型 4.98GB + Qwen3-4B 文本编码器 2.50GB + Flux `ae.safetensors` 0.34GB，共 7.3GB），
`bin/sd` 也确实支持 Z-Image（`docs/z_image.md`，源码里有 `z_image_name_map`），但**这台设备跑不动**，两个独立原因：

**1. 内存不够 —— 被内核 OOM 杀掉。** 两次都被杀，dmesg 明确：

```
Out of memory: Killed process (sd)
  total-vm:8373292kB, anon-rss:7534076kB      # 峰值 7.2GB
```

权重本身 4.75GB（DiT）+ 2.31GB（Qwen3-4B）+ VAE，加载完 RSS 就到 7.5GB。
设备 15GiB 且**无 swap**，桌面 + 常驻服务占掉约 2GB，剩余不够。

**2. 就算内存够，也太慢 —— 474 秒/步。**

```
[INFO] get_learned_condition completed, taking 3.99s      # 文本编码（Qwen3-4B）没问题
       |======>  | 1/8 - 474.74s/it                     # DiT 每步 474 秒
```

8 步 = **约 63 分钟/张**（对比 SD-Turbo 4 步 188 秒，慢约 20 倍）。
sd.cpp 的 s/it 是真实每步耗时（SD1.5 那次第 1 步 65.16s、稳态 64.2s，只差 1.5%），
所以这不是「首步预热的假象」。

**根因是 6B DiT 的算力需求**，不是量化精度问题 —— 换 Q3_K 只省约 1GB 内存、少 10~20% 时间，仍然要 50 分钟以上一张。
G4 量化改变不了这台设备有效算力只有约 12–26 GFLOPS 的事实。

结论：**Z-Image / FLUX / SD3.5 这类 2025 年的 6B+ 模型，在只有 8 个 X100 核、无 GPU、无 NPU 可用的 K3 上没有实用价值。**
这台设备上「能跑的」上限大约就是 SDXL（2.6B UNet）级别。

## 实测性能（512×512）

### ONNX 引擎（默认，X100）

| 步数 | 文本编码 | UNet | VAE 解码 | 合计 |
|---|---|---|---|---|
| **1 步** | 1.5 s | 28.8 s | 71.8 s | **102 s** |
| **4 步** | 1.5 s | 114.8 s | 72.1 s | **188 s** |

VAE 解码占了三分之一以上 —— 它是 fp32 且在 512×512 上做重卷积。

### GGUF 引擎（X100，stable-diffusion.cpp）

SD1.5 Q8_0，8 线程：**64.2 秒/步**；文本编码 0.86 s；20 步约 21 分钟。
只在需要某个特定 GGUF 模型时用。

### 与 diffusers 参考实现的数值对拍（`export/debug_parity.py`）

| 阶段 | 相对误差 |
|---|---|
| token ids | 0（完全一致） |
| prompt_embeds | 7.1e-4 |
| UNet 输出（第 1–4 步） | 2.0e-3 → 2.4e-2 |
| 最终图像像素 | 8.9e-2 |

误差随步数累积，来源是 UNet 的 fp16 推理。图像内容与参考实现一致（肉眼看不出差别）。

---

## 踩过的两个坑（都会让出图变成噪点，且不报错）

1. **`pad_token` 不是 `<|endoftext|>`。**
   sd-turbo 的 `tokenizer_config.json` 把 `pad_token` 设成 `"!"`，**id 是 0**。
   按常规用 49407 填充，文本编码器会看到 77 个 EOS，出图变成棕色抽象斑块。
   另外 `"!"` 是 *added token*，HF 会绕过 BPE 直接映射到 0；走 BPE 会得到 `"!</w>"`=256。
   中文还要按 BERT BasicTokenizer 逐字拆开（且只拆中文区，不含假名/谚文）。
   `imagegen/tokenizer.py` 已全部对齐，15 条中英混合提示词与 HF 输出逐位一致。

2. **`EulerDiscreteScheduler` 会缩放 UNet 输入。**
   `scale_model_input` 要除以 `sqrt(sigma²+1)`。漏掉它，第 1 步 UNet 的输入尺度就差约 14.6 倍，
   出图同样是噪点。`imagegen/scheduler.py` 已修正（Euler-ancestral 不做这个缩放，两者不能混）。

---

## 内存

设备原本被 Bianbu 自带助手套件吃掉 8.4GiB。执行 `scripts/free-memory.sh` 后：

| | 已用 | 可用 |
|---|---|---|
| 之前 | 8.4 GiB | 7.2 GiB |
| 之后 | **2.7 GiB** | **12 GiB** |

停用并 disable 的单元：`bianbu-agent-memory{,-adapter,-embedding}d?`、`bianbu-agent-runtime@bianbu`、
`bianbu-agent-network-learning`、`bibit-*`、`file2md`、`searxng`。
**没动**桌面环境（sddm / labwc / VSCodium）。

还原：

```bash
sudo systemctl enable --now bianbu-agent-memoryd bianbu-agent-memory-adapter \
  bianbu-agent-memory-embedding bianbu-agent-runtime@bianbu file2md searxng
```

---

## 目录结构

```
/home/bianbu/image-generation/
├── bin/sd, bin/sd-server        # stable-diffusion.cpp（自编译，X100）
├── models/
│   ├── gguf/sd15-q8.gguf        # SD1.5 Q8_0（GGUF 引擎）
│   └── onnx/sd-turbo/           # SD-Turbo ONNX 全套（ONNX 引擎）
│       ├── text_encoder.onnx    #   OpenCLIP-H 文本编码器 fp16（681MB）
│       ├── unet.onnx(+.data)    #   860M UNet fp16（1.73GB）
│       ├── vae_decoder.onnx(+.data)  # VAE 解码器 fp32（198MB）
│       ├── meta.json            #   scaling factor + 预计算 sigma 表（1–12 步）
│       └── tokenizer/           #   CLIP BPE 词表 + tokenizer_config
├── imagegen/                    # 纯 numpy + onnxruntime 推理管线（无需 torch）
│   ├── tokenizer.py             #   CLIP BPE 纯 Python 实现
│   ├── scheduler.py             #   Euler / Euler-ancestral 采样器
│   └── pipeline.py              #   三组件编排
├── generate.py                  # 统一 CLI
├── server.py                    # HTTP API + Web UI
├── validate.py                  # 逐组件数值校验
├── probe_npu.py                 # NPU EP 崩溃点定位工具
├── run.sh                       # 便捷入口
├── imagegen.service             # systemd 单元（可选常驻）
├── src/stable-diffusion.cpp/    # 引擎二的源码与 build/
├── scripts/free-memory.sh
├── output/  logs/  bench/
```

设备上**不需要** torch / diffusers / transformers —— 全部推理是 numpy + onnxruntime。

---

## HTTP API

```bash
curl -X POST http://127.0.0.1:8080/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a lighthouse on a cliff at sunset","steps":1,"seed":7,"engine":"onnx"}'
```

```json
{"ok": true,
 "image": "data:image/png;base64,...",
 "info": {"total_s": 102.2, "steps": 1, "timings": {"text_encoder": 1.53, "unet": 28.8, "vae_decoder": 71.8},
          "engine": "onnx", "file": "img_....png"}}
```

其他端点：`GET /`（Web UI）、`GET /health`、`GET /images/<file>`。

常驻服务：

```bash
sudo cp imagegen.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now imagegen
```

---

## 还能再快吗

当前瓶颈：UNet 28.8 秒/步、VAE 解码 72 秒（都是 ORT CPU provider 的卷积）。
按收益排序：

1. **VAE 解码换 fp16**（`export_sd_onnx.py --vae-fp16`）—— X100 有 `zvfh`，值得一试，能省最多时间。
2. **VAE tiling** —— 降峰值内存，不降算力，收益有限。
3. **等 SpacemiT 修 EP 的注意力支持** —— 一旦能跑，UNet/VAE 都有 5–37 倍的先例。
4. **用 ONNX Runtime 的其他 EP**（如 XNNPACK 类 RISC-V 优化）—— 需要自行编译。

设备访问不到 huggingface / hf-mirror；可用 `archive.spacemit.com`（约 37MB/s）、
`github.com`（约 5MB/s）、`modelscope.cn`。PC 与设备同网段，可用 SFTP 推送（约 8–11MB/s）。
