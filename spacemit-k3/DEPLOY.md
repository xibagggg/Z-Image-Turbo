# 端侧文生图部署复现指南

目标设备：**SpacemiT K3 Pico-ITX**，Bianbu 4.0.7 (riscv64)，16GiB 内存，117GiB 系统盘。
全部文件位于设备上的单个目录：`/home/bianbu/image-generation/`。

---

## 1. 硬件与软件事实（实测）

| 项目 | 实测值 |
|---|---|
| SoC | SpacemiT K3 Pico-ITX |
| 内核 | Linux 6.18.3-generic riscv64，Bianbu 4.0.7 (resolute) |
| CPU | 16 核：8× `spacemit,a100` @1.8GHz（AI 核）+ 8× `spacemit,x100` @2.15GHz（应用核） |
| ISA | `rv64imafdcvh` + RVV 1.0 (`zve64d`/`zvfh`/`zvfbfwma`/`zvkned`…) |
| Linux 可用核 | **仅 0–7（X100）**。PID 1 自身 `Cpus_allowed_list: 0-7`，root 也无法把亲和性扩到 8–15 |
| 内存 | 15GiB 可见，无 swap |
| AI 加速入口 | `/dev/ai_dma`、`/dev/aidma_list`（NPU/A100 计算），`/dev/tcm` |
| NPU 软件栈 | `onnxruntime 1.24.2+spacemit.a1` + `spacemit_ort` + `libspacemit_ep.so.2.0.7` + `libspert.so.0.6.2`（Spine 运行时） |

### 关键结论：A100 有两条入口，只有一条可用

1. **ggml / SPERT 路径（不可用）**
   Bianbu 预装的 `llama.cpp-tools-spacemit` 附带打过补丁的 ggml，注册了 `CPU_RISCV64_SPACEMIT`
   后端，声称 `num_perfer_cores: 8, use_ime2: 1, cpu_mask: ff00`（即 8–15 号 A100 核）。
   实测它拿不到收益：
   - `alloc_chunk: open(/dev/tcm_sync_mem) failed, errno=2`，TCM 初始化失败退化为 heap；
   - 进程被内核限制在 0–7 核，`ff00` 掩码无法生效。
   `llama-bench`（Qwen3-0.6B Q4_0，8 线程）只有 **pp64 444 t/s / tg32 34 t/s**，就是 8 个 X100 核的水平。
   把 `ggml/src/ggml-cpu/spacemit/`（约 1.5 万行）移植到 stable-diffusion.cpp 自带的 ggml 0.19
   也走不通：sd.cpp 依赖其 `leejet/ggml` fork 的私有算子（`ggml_mul_mat_i8_tensorwise` 等），
   与系统 ggml 0.16.0 ABI 不兼容（`-DSD_USE_SYSTEM_GGML=ON` 编译失败：`GGML_MAX_NAME>=160` 断言 + 私有算子缺失）。

2. **ONNX Runtime SpaceMIT EP（可用，本项目采用）**
   通过 `import spacemit_ort` 打补丁后，`providers=["SpaceMITExecutionProvider"]` 就会把算子编译到
   A100 上，**不支持的算子自动回落 CPU**，即单次推理天然横跨 A100 + X100。

   | 模型 | CPU 延迟 | NPU 延迟 | 加速比 |
   |---|---|---|---|
   | MobileNetV3-Small fp16 | 23.10 ms | 2.71 ms | **8.5×** |
   | YOLO26n fp16（Conv/SiLU/Concat/Resize） | 529.99 ms | 14.14 ms | **37.5×** |
   | PPLiteSeg | 3495 ms | 667 ms | **5.2×** |

   YOLO 的算子构成与扩散 UNet/VAE 高度重合（Conv + SiLU + Concat + Resize），所以扩散模型上 NPU 是可行的。

---

## 2. 释放内存

设备原本被 Bianbu 自带的助手套件占用 8.4GiB。执行 `scripts/free-memory.sh`：

| | 已用 | 可用 |
|---|---|---|
| 之前 | 8.4 GiB | 7.2 GiB |
| 之后 | **2.7 GiB** | **12 GiB** |

停用并 disable 的单元：`bianbu-agent-memory{,-adapter,-embedding}d?`、`bianbu-agent-runtime@bianbu`、
`bianbu-agent-network-learning`、`bibit-*`、`file2md`、`searxng`。**未动**桌面（sddm/labwc/VSCodium）。

还原：

```bash
sudo systemctl enable --now bianbu-agent-memoryd bianbu-agent-memory-adapter \
  bianbu-agent-memory-embedding bianbu-agent-runtime@bianbu file2md searxng
```

---

## 3. 引擎一：stable-diffusion.cpp（X100 / RVV，吃 GGUF）

```bash
sudo apt-get install -y cmake ninja-build build-essential libcurl4-openssl-dev git
cd /home/bianbu/image-generation/src
git clone --depth 1 https://github.com/leejet/stable-diffusion.cpp.git
cd stable-diffusion.cpp && git submodule update --init --depth 1
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSD_BUILD_EXAMPLES=ON
cmake --build build -j 8            # riscv64，自带 ggml 0.19 + RVV
cp build/bin/sd-cli ../bin/sd
```

实测（SD1.5 Q8_0，512×512，8 线程）：**64.2 秒/步**，文本编码 0.86s。

---

## 4. 引擎二：ONNX Runtime + SpaceMIT EP（A100 NPU）

### 4.1 PC 侧导出 ONNX

设备的 `spacemi_ort` 需要静态形状，因此全部按 batch=1 / 512×512 固定导出。

```bash
# 独立 venv，避免污染系统 Python（系统是 transformers 4.44 + hub 1.31，互相冲突）
python -m venv --system-site-packages .venv
./.venv/Scripts/python.exe -m pip install "huggingface_hub==0.35.3" "transformers==4.44.2" "diffusers==0.31.0"

HF_ENDPOINT=https://hf-mirror.com ./.venv/Scripts/huggingface-cli download stabilityai/sd-turbo \
  --include "*.json" "tokenizer/*" "text_encoder/*.fp16.safetensors" \
            "unet/*.fp16.safetensors" "vae/*.fp16.safetensors" --local-dir build/sd-turbo

./.venv/Scripts/python.exe export_sd_onnx.py --model build/sd-turbo --out build/sd-turbo-onnx
```

**踩过的坑：**

- sd-turbo 是 **SD2.1 蒸馏**，文本编码器是 OpenCLIP-H，**cross-attention 维度是 1024 而非 768**。
  导出脚本从 `text_encoder.config.hidden_size` 读取，不要硬编码。
- sd-turbo 只发布 fp16 权重，`from_pretrained` 必须显式传 `variant="fp16"`。
- `do_constant_folding=True` 会让 1.7GB 的 UNet 导出卡住 20 分钟以上，**设为 False**。
- **参考前向必须在 fp32 下算**。PyTorch 的 CPU fp16 卷积极慢，用它算 `ref_unet_out` 会让导出
  看起来像卡死。脚本里改成 `unet.float()` 算参考、再 `unet.half()` 导出。
- 导出器默认用 **dynamo**（UNet 约 90 秒，opset 20 + external data）。legacy 导出器产出
  opset 17 单文件、更贴近官方 model zoo 的格式，但 UNet 要 20 分钟以上，只在需要时用
  `--exporter legacy`。
- 版本必须配对：diffusers 0.40 需要 transformers ≥4.46（`Dinov2WithRegistersConfig`），
  而 diffusers 0.31 才配 transformers 4.44.2。

### 4.1.1 两个会让出图变成噪点的运行时坑

导出正确不代表出图正确 —— 这两处错了都**不报错**，只是吐一堆棕色抽象斑块：

1. **`pad_token` 是 `"!"`（id 0），不是 `<|endoftext|>`（49407）。**
   sd-turbo 的 `tokenizer_config.json` 就是这么配的。填充错了，文本编码器会看到 77 个 EOS。
   另外 `"!"` 在 HF 里是 *added token*，直接映射到 0、不参与 BPE；当普通标点走 BPE 会得到
   `"!</w>"`=256。中文还需按 BERT BasicTokenizer 逐字拆开，且只拆中文区（不含假名/谚文）。
   验证方法：拿 15 条中英混合提示词与 `transformers.CLIPTokenizer` 逐位比对，要求 0 差异。

2. **`EulerDiscreteScheduler.scale_model_input` 要除以 `sqrt(sigma²+1)`。**
   漏掉它，第 1 步 UNet 的输入尺度就差约 14.6 倍。注意 Euler-ancestral 不做这个缩放，
   两者不能共用一份实现。

配套的逐阶段对拍脚本见 `export/debug_parity.py`：它同时跑 diffusers 参考与本地 ONNX 管线，
比较 token ids / prompt_embeds / 每一步 UNet 的输入与输出 / 最终像素。
调好后本次实测的误差是：ids 完全一致、embeddings 7.1e-4、UNet 输出 2.0e-3→2.4e-2、最终图像 8.9e-2。


### 4.2 传到设备

设备访问不到 huggingface / hf-mirror，也回连不了 PC 的 HTTP 端口（Windows 防火墙）。
可行通道有两条：设备直连 `archive.spacemit.com`（约 37MB/s）与 `github.com`（约 5MB/s），
或由 PC 用 SFTP 推送（约 8.1 MB/s）。本项目用后者：

```bash
python tools/rcmd.py -u export/build/sd-turbo-onnx/unet.onnx \
    /home/bianbu/image-generation/models/onnx/sd-turbo/unet.onnx
```

### 4.3 设备侧运行

```bash
python3 validate.py                       # 逐组件与 PyTorch 参考张量比对
python3 generate.py -p "..." -o output/x.png   # 默认 ONNX 引擎，SD-Turbo
python3 server.py                         # Web UI + HTTP API，默认 :8080
```

实测（512×512）：1 步 102 秒（文本编码 1.5s + UNet 28.8s + VAE 71.8s），4 步 188 秒。

NPU 通道（`--device npu`）当前不可用：SpacemiT EP 在含自注意力的图上建会话时段错误，
连官方 model zoo 自带的 ViT / CLIP 文本编码器也一样崩，详见 README 的对比表与复现工具
`probe_npu.py`。

`imagegen/` 是纯 numpy + onnxruntime 实现，设备上**不需要 torch / diffusers / transformers**：
- `tokenizer.py`：CLIP BPE 纯 Python 实现（`tokenizers` 无 riscv64 wheel）
- `scheduler.py`：Euler / Euler-ancestral 采样器，sigma 表由 PC 侧预计算写入 `meta.json`，
  避免设备端重算导致与参考实现漂移
