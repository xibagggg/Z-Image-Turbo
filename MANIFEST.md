# 归档清单：这个仓库里有什么，没有什么，以及怎么把它重新跑起来

这个仓库是**源码归档**，不含大文件。下面把"哪些东西在仓库里、哪些不在、不在的东西怎么拿回来、要花多久"逐条写清楚，照着做可以在一台新机器上从零恢复出可运行的服务。

所有 SHA-256 都是**实测值**：设备上现存的三个权重文件算出来的哈希，与 ModelScope / HF 发布页公布的哈希**逐字节一致**，所以按下面的地址下载得到的文件与当前在跑的那套完全相同。

---

## 一、结论速览

| 资产 | 大小 | 在仓库里吗 | 怎么拿回来 | 实测耗时 |
| --- | --- | --- | --- | --- |
| 全部手写源码（46 个文本文件） | 2.8 MB | ✅ **在** | clone 即可 | < 1 min |
| 示例出图（19 张 PNG） | 16.6 MB | ✅ **在** | clone 即可 | < 1 min |
| Z-Image 三个权重 | 7.8 GB | ❌ 不在 | `deploy/download-weights.sh`（走 ModelScope，自动校验哈希） | 25–60 min（单流 2–6 MiB/s） |
| stable-diffusion.cpp 源码 | 96 MB | ❌ 不在 | `git clone` + 锁定 commit `59c23bc` | 1–3 min |
| `sd-cli` / `sd-server` 二进制 | 446 MB | ❌ 不在 | 上面那份源码编译，命令见下 | **8.1 min**（实测 393 个目标） |
| Python venv（导出用） | 58 MB | ❌ 不在 | `docs/requirements-export.txt` | 2–5 min |
| K3 的 ONNX 导出包 | 2.9 GB | ❌ 不在 | 跑 `spacemit-k3/export/export_sd_onnx.py` 重新导出 | 15–30 min |
| K3 用的 sd-turbo 原始权重 | 4.9 GB | ❌ 不在 | `HF_ENDPOINT=https://hf-mirror.com` 拉取 | 20–40 min |
| 设备上 `output/` 图库（21 张测试图） | 24 MB | ❌ 不在 | 图自带完整参数元数据，可按元数据重跑（见第八节） | — |

**一句话：源码一份不落全在仓库里；剩下的都是"能按图索骥下载或重算"的东西，不是丢失就找不回来的。** 需要留意的只有两处：K3 那套 ONNX 包能重新导出但过程最麻烦（第五节）；设备 `output/` 里那 21 张测试图没有外部来源，但每张图自带完整参数，能按元数据重跑（第八节）。

### 仓库里那份源码与设备上跑的是否一致

逐个比对过 md5，**完全一致**：

| 文件 | 仓库内 md5 | 设备上 md5 |
| --- | --- | --- |
| `deploy/run.sh` | `01e9f55e…` | `01e9f55e…` |
| `deploy/webui.py` | `57711e0e…` | `57711e0e…` |
| `deploy/web/index.html` | `b3db63d3…` | `b3db63d3…` |
| `deploy/gpu_watch.sh` | `f02f9e4b…` | `f02f9e4b…` |
| `deploy/steps_seed_demo.sh` | `c274f2a7…` | `c274f2a7…` |

不是"抄了一份差不多的"，是同一份字节。把 `deploy/` 拷回设备即可覆盖，不会引入差异。

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

---

## 八、设备上 `output/` 那批图

上面所有"不在仓库里"的资产，都有下载地址或重算路径。设备上 `output/` 里的图是唯一没有外部来源的一批：

```
output/          24 MB
├── zimage_1024.png, zimage_512.png, zimg_1024_s4.png, zimg_1024_s8.png   # 命令行基准出图
└── ui/          21 张 PNG，名字是「时间戳-提示词-种子」
                  20260917-093517-a-red-fox-sitting-in-a-snowy-forest-soft-morning-777.png
                  20260917-143817-一-主体-主体是一位古风女性角色-…-1669265377.png
```

好消息是**这批图自带完整配方**。每一张 PNG 里都有 sd.cpp 写进去的 `parameters` 文本块，提示词全文、步数、CFG、seed、分辨率、用的哪个模型全在里面：

```
a red fox sitting in a snowy forest, soft morning light
Steps: 4, CFG scale: 1.000000, Guidance: 3.500000, Eta: inf, Seed: 777,
Size: 512x512, RNG: cuda, Sampler: NONE,
TE: Qwen3-4B-Instruct-2507-Q4_K_M.gguf, Unet: z_image_turbo-Q4_K_M.gguf,
VAE: ae.safetensors, Version: stable-diffusion.cpp
```

所以图丢了也**能按元数据重跑出来**——文件名里的提示词被截断到 48 字符，真正完整的那份在 PNG 内部。仓库里 `README.md` 和 `docs/` 引用的 6 张（`samples/`）已归档；这里剩下的 21 张没有，因为它们是逐次调试的中间产物。

要备份就手动拷出来：

```bash
scp -r bianbu@192.0.2.10:/home/bianbu/image-generation/output ./device-output-backup
```

仓库根的 `.gitignore` 里有 `output/`，这类生成物不会被误提交进 Git——这是有意的，免得仓库被测试图撑大。

---

## 九、工作区里那三个"看起来该备份"的目录

本地工作区除了源码，还有三个体量很大的目录。逐个查过之后，**结论是三个都不需要归档**，原因各不相同——尤其第一个，它根本不是你想的那个东西。

### `src/sdcpp/` —— 是个坏掉的半成品克隆，没有任何内容

第一眼是 79 MB 的 `.git`，像是"外部克隆的 stable-diffusion.cpp 源码"。实际上：

```
src/sdcpp/.git/
├── objects/pack/tmp_pack_4KauoG    78.4 MB   ← 没下载完的临时包
└── shallow.lock                    41 B      ← 残留的锁文件
```

**没有 `HEAD`、没有 `config`、没有 `refs/`、没有任何工作区文件。** 这是 `git clone` 下到一半被中断留下的残骸：对象包连名字都还叫 `tmp_pack_*`（正式名字应该是 `pack-<sha>.pack`），`shallow.lock` 是进程异常退出时没清掉的锁。这个目录里一行业务代码都没有，上传上去只是 79 MB 垃圾。

真正在跑的那份在**设备上**，而且**与上游一字不差**：

```
$ cd /home/bianbu/image-generation/src/stable-diffusion.cpp
$ git status --short          # 空
$ git log -1 --format='%H %ad'
59c23bce0d82be3a922023ab811194f05b3e2faa Tue Sep 15 02:37:52 2026 +0800
$ git remote -v
origin  https://github.com/leejet/stable-diffusion.cpp.git
```

零本地改动，所以"归档"它等于复制一份别人的仓库，而且会归档到一个**过时且不完整**的快照。正确做法是锁 commit 后重新 clone，**实测 1–3 分钟**，命令见第四节。

### `export/.venv/` —— Windows 专用，而且被 `requirements-export.txt` 完全覆盖

58 MB、2801 个文件。两个问题：

- `pyvenv.cfg` 里 `home = C:\Users\<你的用户名>\AppData\Local\Programs\Python\Python311`、`command = ... python.exe -m venv --system-site-packages ...`。**路径是硬编码的本机绝对路径**，换机器、换用户、换系统都不能用，venv 本来就不该跨机复制。
- 它是 `--system-site-packages` 建的，真正的重活（`torch 2.13.0+cpu`，2 GB 多）**不在 venv 里**，在系统解释器里。venv 里只有 4 个包：`diffusers 0.31.0`、`huggingface_hub 0.35.3`、`pip`、`setuptools`。所以就算把它拷走，换个环境一样跑不起来。

`docs/requirements-export.txt` 里记的是**完整的** 11 个包及其精确版本（含 venv 之外的 torch / transformers / onnx / onnxruntime / numpy / Pillow / protobuf / tokenizers / safetensors），重建命令也写在文件头。这才是可移植的归档形式。

### `models/` —— 3.4 GB，但三个部分的性质完全不同

| 文件 | 大小 | 能拿回来吗 |
| --- | --- | --- |
| `comfy-vae/split_files/vae/ae.safetensors` | 335 MB | ✅ **能** —— SHA-256 `afc8e28272cd15db…`，**与 Jetson 上那个 VAE 是同一个文件**，走 `deploy/download-weights.sh` 就会下到 |
| `sd15-q8.gguf` | 1.88 GB | ⚠️ 源头没记录，但能从 base 模型重新转换 |
| `sdxl-turbo-q8.gguf` | 1.30 GB | ⚠️ 同上 |
| ONNX 探针模型 ×6（mobilenet / resnet50 / vit / pp_liteseg） | 0.27 GB | ⚠️ 诊断用，源头没记录；标准 torchvision / ModelScope 导出可替代 |

**两个 GGUF 为什么"源头没记录"**：GGUF 头部的 `kv_count = 0`，即转换时没写入任何 `general.source.url` / `general.author` 之类的溯源字段，文件里查不到出处：

```
sd15-q8.gguf:        ver=3 tensors=1130 kv=0
sdxl-turbo-q8.gguf:  ver=3 tensors=2641 kv=0
```

（顺带确认了后者**确实是 SDXL**：它有 1680 个 `model.` 张量和 `model.diffusion_model.label_emb`，这是 SDXL 的 size/AR 条件嵌入，SD1.5 没有。）

也比对过公开的 GGUF 转换版：`second-state/stable-diffusion-v1-5-GGUF` 的 `Q8_0` 是 1763578176 字节，而本地这个是 1881241504 字节——**不是同一个文件**，是本地另行转换的。

要复现，从原始 safetensors 转即可：

```bash
# 原始权重（ModelScope 上都有，已验证可达）
#   stabilityai/sdxl-turbo                → sd_xl_turbo_1.0_fp16.safetensors   6938081905 字节（6.46 GB）
#   AI-ModelScope/stable-diffusion-v1-5   → v1-5-pruned-emaonly.safetensors   4265146304 字节（3.97 GB）
git clone https://github.com/leejet/stable-diffusion.cpp && cd stable-diffusion.cpp
git checkout 59c23bc
python3 convert.py --model-type sdxl --fp16 <原始.safetensors> --outtype q8_0 --output sdxl-turbo-q8.gguf
```

本地那两份的 SHA-256（想验证重转结果或找到别处副本时用得上）：

```
sd15-q8.gguf         1881241504  a51037fac577d133fe3b83a4a0c2a7605615786f07bdec397378143fe0361f32
sdxl-turbo-q8.gguf   1302343680  0bbedbf92031504c4a89c2f302ea03f9eedb8a31227f587dae1bad3d3d6820dc
```

转出来的字节未必与本地那份逐字节相同（量化器版本、参数都可能差），但**功能等价**——而且 K3 的 GGUF 引擎本来就是备用路径（主路径是 ONNX），见 `spacemit-k3/README.md` 的实测性能一节：GGUF 64.2 秒/步，只在该用某个特定 GGUF 模型时才用。

### 那到底要不要把 `models/` 传上去？

**可以传，但不该传到这个仓库。** 原因是这个仓库是**公开**的（无凭据 `HTTP 200`，README 和示例图任何人都能看）。

把它放进 GitHub Release 会让这 3.4 GB 对全世界可下载——那是**发布**，不是"个人保存"。而且模型权重各有各的许可证（SD1.5 是 CreativeML OpenRAIL-M、SDXL-Turbo 是 Stability AI Community License、FLUX VAE 是 Apache-2.0），重新分发要附带许可条款。

**如果目的是"怕本地文件被删"，更合适的做法**：

1. 先确认哪些真的不可替代——`ae.safetensors` 不用管（随时能下），两个 GGUF 和 ONNX 探针可以重转重导出。真正一份不落地只有你自己生成的东西。
2. 要离线留一份，存到私有介质：移动硬盘、NAS、或者私有对象存储。Git 不适合放 2 GB 的文件，GitHub 的公开仓库更不适合放"私人物品"。
3. 如果确实想借 GitHub 存，那就**换成私有仓库**（Private repo 的 Release 也只有你能看到），或者用 Git LFS 放到私有仓库——但 LFS 免费额度是 1 GB 存储 + 1 GB/月流量，3.4 GB 会超。

判断依据：这台机器上**真正独一份**的是 `output/` 里那 21 张测试图（第八节），不是这些能从上游复原的模型。别把备份精力花错地方。
