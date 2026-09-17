# 归档清单：这个仓库里有什么，没有什么，以及怎么把它重新跑起来

这个仓库是**源码归档**，不含大文件。下面把"哪些东西在仓库里、哪些不在、不在的东西怎么拿回来、要花多久"逐条写清楚，照着做可以在一台新机器上从零恢复出可运行的服务。

所有 SHA-256 都是**实测值**：设备上现存的三个权重文件算出来的哈希，与 ModelScope / HF 发布页公布的哈希**逐字节一致**，所以按下面的地址下载得到的文件与当前在跑的那套完全相同。

---

## 一、结论速览

| 资产 | 大小 | 在仓库里吗 | 怎么拿回来 | 实测耗时 |
| --- | --- | --- | --- | --- |
| 全部手写源码（105 个文件） | 1.4 MB | ✅ **在** | clone 即可 | < 1 min |
| Z-Image 三个权重 | 7.8 GB | ❌ 不在 | `deploy/download-weights.sh`（走 ModelScope，自动校验哈希） | 25–60 min（单流 2–6 MiB/s） |
| stable-diffusion.cpp 源码 | 96 MB | ❌ 不在 | `git clone` + 锁定 commit `59c23bc` | 1–3 min |
| `sd-cli` / `sd-server` 二进制 | 446 MB | ❌ 不在 | 上面那份源码编译，命令见下 | **8.1 min**（实测 393 个目标） |
| Python venv（导出用） | 58 MB | ❌ 不在 | `docs/requirements-export.txt` | 2–5 min |
| K3 的 ONNX 导出包 | 2.9 GB | ❌ 不在 | 跑 `spacemit-k3/export/export_sd_onnx.py` 重新导出 | 15–30 min |
| K3 用的 sd-turbo 原始权重 | 4.9 GB | ❌ 不在 | `HF_ENDPOINT=https://hf-mirror.com` 拉取 | 20–40 min |

**一句话：源码一份不落全在仓库里；剩下的都是"能按图索骥下载或重算"的东西，不是丢失就找不回来的。** 唯一需要留心的是 K3 那套 ONNX 包——它能重新导出，但过程最麻烦（见第五节）。

---

## 二、为什么权重不进 Git

不是嫌麻烦，是 Git 本身不适合存这个：

- GitHub 单个文件硬上限 **100 MB**；`z_image_turbo-Q4_K_M.gguf` 是 4.64 GiB。
- 要放就得走 Git LFS。而 LFS 的免费额度是 **1 GB 存储 + 1 GB/月流量**，7.8 GB 的权重推一次、以后每克隆一次都算流量。
- 更关键的是**没有必要**：这三个文件在公开渠道都能下到，而且**哈希与原文件完全一致**。把它们再存一份，等于用掉配额换一个"多一份副本"，收益很低。

---

## 三、Z-Image 权重（Jetson 用）

一条命令搞定，脚本会自己校验哈希，哈希对不上就报错退出，不会留下半个坏文件：

```bash
./deploy/download-weights.sh            # 默认下到 models/zimage/
./deploy/download-weights.sh /data/zi   # 也可以指定目录
```

脚本里固化的三个文件（`deploy/download-weights.sh:22-26`）：

| 文件 | 大小 | SHA-256 | 来源 |
| --- | --- | --- | --- |
| `z_image_turbo-Q4_K_M.gguf` | 4981532736 | `745ec270db042409fde084d6b5cfccabf214a7fe5a494edf994a391125656afd` | ModelScope `jayn7/Z-Image-Turbo-GGUF` |
| `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | 2497281120 | `3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597` | ModelScope `unsloth/Qwen3-4B-Instruct-2507-GGUF` |
| `ae.safetensors` | 335304388 | `afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38` | ModelScope `black-forest-labs/FLUX.1-schnell` |

### 为什么走 ModelScope 而不是 Hugging Face

- 构建这套环境的两条网络（PC 和 Orin 设备）**都连不上 huggingface.co**。设备侧是直接 `curl` 超时（`000`，0.02 s 就断），PC 侧 DNS 能解析到 `47.88.58.234` 但 TLS 握不上手。
- `hf-mirror.com` 从 PC 上是通的（实测 5/5 次 200），但**不稳且更慢**：同一个 Qwen3 文件，hf-mirror 实测 2.7 MiB/s，ModelScope 实测 6.0 MiB/s。
- 前两个文件在两个渠道都有；但 `ae.safetensors` 在 hf-mirror 上只返回 **183 字节**（FLUX.1-schnell 是 gated 仓库，未登录拿不到），**只有 ModelScope 能下**。所以统一走 ModelScope，少一套凭据。

### 备用渠道（两个文件可用）

```bash
HF=https://hf-mirror.com
curl -fL -C - -o z_image_turbo-Q4_K_M.gguf \
  $HF/jayn7/Z-Image-Turbo-GGUF/resolve/main/z_image_turbo-Q4_K_M.gguf
curl -fL -C - -o Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  $HF/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf
```

两个 `Content-Length` 与上表逐字节一致（`4981532736` / `2497281120`），下完仍建议核一遍 SHA-256。

---

## 四、stable-diffusion.cpp（推理引擎）

仓库里**没有** vendored 这份外部源码（96 MB，含自己的 `.git`），因为它是别人的项目且我们一行没改——设备上 `git status` 是干净的。恢复方式就是 clone 到那个 commit：

```bash
git clone https://github.com/leejet/stable-diffusion.cpp
cd stable-diffusion.cpp
git checkout 59c23bce0d82be3a922023ab811194f05b3e2faa    # 2026-09-15
```

**这个 commit 号要锁死。** 实测跑通的是 `59c23bc`；sd.cpp 迭代很快，`master` 上的新 commit 与本文档记录的参数默认值、`z_image.hpp` 的 forward 签名都不保证一致。

编译（实测 **8.1 分钟**，393 个编译目标，12 核并行）：

```bash
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSD_BUILD_EXAMPLES=ON \
  -DSD_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build -j$(nproc)
```

`CMAKE_CUDA_ARCHITECTURES=87` 必须写对（Orin 是 sm_87），否则编出来的 SASS 跑不了。产物 `sd-cli` 233 MB、`sd-server` 234 MB。

> 编译产物（446 MB）**不适合放 Releases**：它绑定 `sm_87` + CUDA 12.6，换台机器就得重编；而重编只要 8 分钟。存它没有意义。

---

## 五、SpacemiT K3 那套（`spacemit-k3/`）

设备上 `models/onnx/sd-turbo/` 那 2.9 GB 导出包**不在仓库里**，但它可以从零重算，源材料都拿得到：

**1. 装导出环境**（PC 侧，见 `docs/requirements-export.txt`）

```bash
python -m venv --system-site-packages .venv
./.venv/Scripts/python.exe -m pip install -r docs/requirements-export.txt
```

`--system-site-packages` 是有意的：torch 装了 2 GB 多，留在系统解释器里；venv 只覆盖真正冲突的那几个（系统的 `transformers`/`hub` 与 `diffusers` 对不上）。

**2. 拉 sd-turbo 的 diffusers 版权重**

```bash
export HF_ENDPOINT=https://hf-mirror.com
./.venv/Scripts/huggingface-cli download stabilityai/sd-turbo \
  --include "*.json" "tokenizer/*" "text_encoder/*.fp16.safetensors" \
            "unet/*.fp16.safetensors" "vae/*.fp16.safetensors" \
  --local-dir build/sd-turbo
```

hf-mirror 上有完整的 diffusers 目录结构，逐个文件实测可下：`model_index.json` 616 B、`unet/config.json` 1867 B、`tokenizer/vocab.json` 1059962 B、`unet/diffusion_pytorch_model.fp16.safetensors` 1731904736 B。

> ModelScope 上也有 `stabilityai/sd-turbo`，但**只有单文件** `sd_turbo.safetensors`（4.9 GB），没有 diffusers 需要的目录结构；`export_sd_onnx.py` 走的是 `UNet2DConditionModel.from_pretrained` 这条路，所以这里必须用 hf-mirror。

**3. 导出 + 传上设备**

```bash
./.venv/Scripts/python.exe export/export_sd_onnx.py --model build/sd-turbo --out build/sd-turbo-onnx
python tools/rcmd.py -u export/build/sd-turbo-onnx/unet.onnx \
    /home/bianbu/image-generation/models/onnx/sd-turbo/unet.onnx
# ...其余文件同理；注意大文件带 sidecar .data
```

**4. 上设备验证**

```bash
python3 validate.py            # 逐组件与 PyTorch 参考张量比对
```

实测误差：ids 完全一致、embeddings 7.1e-4、UNet 输出 2.0e-3→2.4e-2、最终图像 8.9e-2。

K3 侧的 NPU 依赖是 **apt 包，不是 pip 包**（riscv64 没有 wheel）：

```bash
sudo apt install python3-spacemit-ort     # 2.0.7+2
# 提供 onnxruntime 1.24.2+spacemit.a1 / spacemit_ort /
#      libspacemit_ep.so.2.0.7 / libspert.so.0.6.2
```

---

## 六、从零恢复一台新 Orin：完整顺序

```bash
# 1) 源码
git clone https://github.com/xibagggg/Z-Image-Turbo && cd Z-Image-Turbo

# 2) 权重（7.8 GB，25–60 分钟）
./deploy/download-weights.sh

# 3) 引擎（8 分钟）
git clone https://github.com/leejet/stable-diffusion.cpp /tmp/sdcpp
cd /tmp/sdcpp && git checkout 59c23bc && \
  cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSD_BUILD_EXAMPLES=ON \
    -DSD_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=87 && cmake --build build -j$(nproc)
mkdir -p ../bin && cp build/bin/sd-cli build/bin/sd-server ../bin/
cd ..

# 4) 起服务
./deploy/run.sh serve                      # sd-server :8080
SD_API=http://127.0.0.1:8080 UI_PORT=8081 python3 deploy/webui.py   # UI :8081

# 5) 开机自启（改掉单元里的路径和 User=）
sudo cp deploy/imagegen.service deploy/imageui.service /etc/systemd/system/
sudo systemctl enable --now imagegen imageui
```

**总耗时约 40–80 分钟**，其中权重下载占绝大部分，编译只要 8 分钟。

---

## 七、目录对应关系

| 仓库内路径 | 设备上的位置 |
| --- | --- |
| `deploy/*.sh`、`deploy/webui.py`、`deploy/web/` | `/home/bianbu/image-generation/` |
| `deploy/*.service` | `/etc/systemd/system/` |
| `models/zimage/*`（需自行下载） | `/home/bianbu/image-generation/models/zimage/` |
| `spacemit-k3/` | K3 设备上 `/home/bianbu/image-generation/`（扁平铺开，不是子目录） |
| `spacemit-k3/samples/` | K3 上 `output/` 里挑出来的代表图 |

设备原目录是**扁平**的（`generate.py`、`server.py`、`validate.py` 都在根下）。仓库里为了把两个项目分开放进 `spacemit-k3/` 子目录，代码里的相对路径（`models/gguf/...`、`output/`）按 K3 上的布局写，直接拷回设备根目录即可运行。
