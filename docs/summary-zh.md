# 端侧文生图部署总结（Z-Image-Turbo on Jetson AGX Orin）

> 本文里的 `192.0.2.x` 是 RFC 5737 的文档专用地址段，实际部署时替换成你的设备 IP。

> 设备：`192.0.2.10`　·　完成日期：2026-09-17
> 网页 UI：**http://192.0.2.10:8081**　·　API：`http://192.0.2.10:8080`

---

## 一、一句话结论

在一台 Jetson AGX Orin 上把 **Z-Image-Turbo（6B 文生图模型，GGUF Q4_K_M 量化）** 跑在了 CUDA 上，1024×1024 / 8 步约 **59 秒**出一张图，512×512 / 8 步约 **12 秒**；配了一个自带图库的网页 UI，**中文提示词直接可用**，两个 systemd 服务开机自启。

作为对比：同一个模型之前在另一台 SpacemiT K3 设备上每步要 **474 秒**、还会被 OOM 杀掉——这台是 **6.42 秒/步**，快约 **73 倍**，差异是「能跑」和「不能跑」的区别。

---

## 二、软硬件环境

| 项目 | 配置 |
| --- | --- |
| 设备 | NVIDIA Jetson AGX Orin |
| 系统 | L4T R36.4.7（JetPack 6.x）/ Ubuntu 22.04.5 LTS |
| 内核 | `5.15.148-tegra` |
| CPU | 12× Cortex-A78AE (aarch64)，最高 2.2 GHz |
| 内存 | 29 GiB RAM + 14 GiB Swap |
| 存储 | NVMe 467 GB（已用 343 GB，78%） |
| CUDA | 12.6，compute capability **8.7**（Ampere） |
| 显存 | 统一内存架构，GPU 可见 **30696 MiB**（与系统内存同一物理池） |

⚠️ **关键认知**：Orin 是**核显 + 统一内存**，没有独立显存。GPU 用的内存就是系统内存本身，所以「显存占用」和「进程 RSS」是同一笔账。这也直接导致了 `nvidia-smi` 在这台机器上不可用（见第八节）。

---

## 三、选型过程与结论

### 为什么是 Z-Image-Turbo

需求是「效果好的新模型 + 下载量高 + 能在端侧跑」。最终选 **阿里通义 Z-Image-Turbo**，理由：

- **6B DiT**，在同量级里出图质量好，且是 guidance-distilled 版本，**8 步就能出图**（普通模型要 20–30 步）
- 用了 **Qwen3-4B** 做文本编码器 → **中文原生支持**，这是对比 SDXL / Flux 系的最大优势
- 社区有现成的 **GGUF 量化**，Q4_K_M 把 7.8 GB 的模型压到端侧可承受的范围

模型来自：`https://www.modelscope.cn/models/jayn7/Z-Image-Turbo-GGUF/files`

### 为什么不用 NPU / llama.cpp-tools-spacemit

设备上确实装得了 `llama.cpp-tools-spacemit`，但这套是给 **SpacemiT 的 RISC-V NPU（A100 核）** 用的，而 `192.0.2.10` 是 **ARM + NVIDIA GPU**，两者指令集与加速器完全不兼容。这台机器正确的加速路径就是 **CUDA**。

> 补充：`192.0.2.20` 才是 SpacemiT K3（有 x100/A100 核）。两台设备的技术栈是分开的。

### 为什么用 stable-diffusion.cpp

- 单文件 C++ 实现，**原生支持 GGUF 量化**、原生 CUDA 后端
- 自带 `sd-server`，提供 **OpenAI 兼容的 HTTP 接口**，方便接前端
- 不需要 Python 生态（这台设备没有也不该装一堆 pip 依赖）

版本：`leejet/stable-diffusion.cpp`，commit **`59c23bc`**（2026-09-15）

---

## 四、模型与运行参数

### 模型文件（`models/zimage/`，合计 7.8 GB）

| 文件 | 大小 | 作用 |
| --- | --- | --- |
| `z_image_turbo-Q4_K_M.gguf` | 4.98 GB | **6B DiT** 主模型（去噪） |
| `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | 2.50 GB | 文本编码器（理解提示词） |
| `ae.safetensors` | 0.34 GB | Flux VAE（潜变量 ↔ 像素） |

架构是 **FLOW 模式**：提示词 → Qwen3 编码 → DiT 迭代去噪 → VAE 解码成图。

### 启动参数

```bash
./bin/sd-server \
  --diffusion-model models/zimage/z_image_turbo-Q4_K_M.gguf \
  --vae             models/zimage/ae.safetensors \
  --llm             models/zimage/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  --cfg-scale 1.0 \
  --diffusion-fa \
  --steps 8 -W 1024 -H 1024 \
  --listen-ip 0.0.0.0 --listen-port 8080
```

两个不能动的参数：

- **`--cfg-scale 1.0`**：Z-Image-Turbo 是 guidance-distilled 模型，引导强度已经烧进权重。给大于 1 的值**只会更慢**，没有任何收益。网页 UI 里索性不暴露这个选项。
- **`--diffusion-fa`**：开 FlashAttention，省显存也更快，必开。

### 这两个参数为什么不能动（1024²/8 步，预热后交替 A/B）

| 配置 | 采样耗时 | 说明 |
| --- | --- | --- |
| `--cfg-scale 1.0 --diffusion-fa` | **53.4 s** | 生产配置 |
| 去掉 `--diffusion-fa` | 74.0 s | 慢 38%，且计算缓冲变大，一次运行还撞上过 `cudaMalloc failed: out of memory` |
| `--cfg-scale 3.0 --diffusion-fa` | 23.9 s（512²/8 步，对比 1.0 的 12.8 s） | 只要 >1 就多跑一遍无条件前向，约 1.9 倍耗时；`3.0` 和 `7.0` 耗时完全相同（23.9 s），说明代价来自「是否开 CFG」而不是数值大小 |

README 里经常被忽略的一点：**`--guidance`（distilled guidance）对 Z-Image 是空操作**。`src/model/diffusion/z_image.hpp` 的 `ZImageModel::forward` 签名里根本没有 guidance 输入（全文 0 处 `guidance`），`sd_version_is_flux()` 也不含 `VERSION_Z_IMAGE`，所以那份 `guidance_tensor` 传不进 Z-Image。实测 `--guidance 3.5` 与 `--guidance 10.0` 出图**逐像素完全相同**（262144 个像素 0 处不同）。PNG 元数据里那行 `Guidance: 3.500000` 只是「请求参数的回执」，不代表被消费了。

### 显存分布（实测）

```
total params memory size = 7288.11MB (VRAM 7288.11MB, RAM 0.00MB)
  ├─ text_encoders    2375.91MB (VRAM)
  ├─ diffusion_model  4752.20MB (VRAM)
  └─ vae               160.00MB (VRAM)
```

**`RAM 0.00MB`** —— 全部权重都在显存里，系统内存里一份都没留。另外还有 DiT 计算缓冲 690 MB、VAE 计算缓冲 6657 MB（1024² 时）。

---

## 五、实测性能

### 出图耗时（实测）

| 分辨率 | 步数 | 命令行 `sd` | 服务常驻 `sd-server` |
| --- | --- | --- | --- |
| 512×512 | 8 | 17.3 s | **12.3 s** |
| 512×512 | 4 | — | **6.8 s** |
| 1024×1024 | 4 | 41.9 s | — |
| 1024×1024 | 8 | 62.4 ~ 62.8 s | **57.7 ~ 59.2 s** |

服务比命令行快一截，是因为 `sd-server` 会把图切分方案缓存下来（日志里的 `build cached graph cut plan`），后续请求省掉这部分开销。

单步成本：**512² 约 1.38 s/步，1024² 约 6.42 s/步**。

### 1024²/8 步的阶段分解

| 阶段 | 耗时 |
| --- | --- |
| 文本编码（Qwen3） | 0.94 s（常驻后降到 **0.05 s**） |
| **采样（DiT ×8 步）** | **54.18 s** ← 占了 95% |
| VAE 解码 | 7.25 s |

结论：**瓶颈完全在 DiT 采样**，优化方向只有降步数或降分辨率。

### 步数对照实验（512²，同 seed=42，中文提示词）

| 步数 | 耗时 | 效果 |
| --- | --- | --- |
| 1 | 2.64 s | 糊，只有大致轮廓和色块 |
| 2 | 4.01 s | 能认出主体，毛发糊成一片 |
| 4 | 6.77 s | 锐利可用，**性价比拐点** |
| 8 | 12.28 s | 毛发质感、背景虚化最干净 |

耗时**线性增长**。低于 4 步开始出结构错误（肢体错位、糊脸）。8 步以上收益很小，16 步只是白花一倍时间。

### 与 192.0.2.20（SpacemiT K3）对比

| 指标 | 192.0.2.20 | 192.0.2.10 |
| --- | --- | --- |
| 单步耗时 | 474.74 s | **6.42 s** |
| 出图能力 | 两次 OOM 被杀 | 稳定连续出图 |
| 相对速度 | 1× | **约 73×** |

---

## 六、网页 UI（Z-Image Studio）

**地址：http://192.0.2.10:8081**

单个 `web/index.html`（约 41 KB），**纯原生 JS/CSS，零外部依赖**（无 CDN，断网可用，手机可开）。

### 架构

```
浏览器 ──8081──> webui.py（代理 + 图库）──8080──> sd-server（sd.cpp）
```

`webui.py`（约 13 KB，**只用 Python 标准库**，设备上不需要 pip 装任何东西）负责三件事：

1. **参数注入** —— 绕开 sd.cpp 接口的参数黑洞（见第七节）
2. **结果回读** —— 解析 PNG 里的 `parameters` chunk，以**服务器自报的参数**为准
3. **图库落盘** —— 图片存设备 `output/ui/`，不用浏览器 localStorage（后者配额约 5 MB，两张 1024² 就爆）

### 功能

| 区域 | 说明 |
| --- | --- |
| 提示词 | 6 个一键预设、实时字数、`Ctrl`+`Enter` 快速生成 |
| 分辨率 | 512² / 768² / 1024² / 自定义，切换时实时显示该档 s/步 |
| 采样步数 | 1–20 滑杆，标注「最快 / 推荐 / 更精细」 |
| 种子 | 手填或 🎲 随机，同种子可复现 |
| 连续生成 | 1–4 张，种子依次 +1，方便挑图 |
| 进度 | 环形进度条按实测 s/步 估算阶段（文本编码 → 采样 Step k/N → VAE 解码） |
| 作品集 | 瀑布流、悬停浮层、大图看完整参数、复用/下载/复制/删除 |

大图底部那行「**服务器记录**」是从 PNG 元数据里读出来的真实参数，不是前端自己写的数字：

```
服务器记录：Steps: 8, CFG scale: 1.000000, Guidance: 3.500000, Eta: inf,
Seed: 42, Size: 512x512, RNG: cuda, TE: Qwen3-4B-Instruct-2507-Q4_K_M.gguf, ...
```

注意这行是**请求参数的回执**，不等于都被消费了：`CFG scale` 和 `Steps` 确实生效，而 `Guidance: 3.5` 对 Z-Image 无效（原因见第四节）。

### 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | UI 页面 |
| GET | `/api/health` | 健康检查（前端用它探测服务在线） |
| GET | `/api/gallery` | 图库索引 |
| GET | `/images/<name>` | 单张图（带路径穿越防护） |
| POST | `/api/generate` | `{prompt, width, height, steps, seed}` |
| DELETE | `/api/gallery/<name>` | 删图 |

---

## 七、中文提示词

**可以直接用中文写，不需要翻译成英文。**

文本编码器是 **Qwen3-4B-Instruct-2507**（通义千问系列），中文是它的母语级能力。

### 实测：中英文耗时一致

| 提示词 | 耗时 |
| --- | --- |
| `一只红色的狐狸坐在雪地里，柔和的晨光，细节丰富，电影感` | 12.29 s |
| `a red fox sitting in a snowy forest, soft morning light, highly detailed, cinematic` | 12.28 s |

中文的 token 开销在这个量级上可以忽略，**不用担心"中文会慢"**。

### 中文特有概念的优势

> `春天的苏州园林，白墙黛瓦，桃花纷飞，午后柔和的阳光洒在回廊上，胶片摄影质感`

出的是白墙、黛瓦、木质回廊、漏窗花窗、屋顶瓦当，**建筑语汇全部正确**。这类题材写英文反而容易画成日式庭院。

### 写法建议

和英文逻辑一样，堆「**主体 + 光线 + 材质细节 + 镜头风格**」四类词效果最好：

```
<主体>，<光线>，<材质/细节>，<镜头或风格>
```

中英混写也可以（中文描述画面 + 英文补 `cinematic lighting, 85mm`）。

### UI 内置的 6 个中文预设

🦊 雪地红狐 · 🏮 苏州园林 · 🏔️ 晨雾山脉 · 🖌️ 水墨山水 · 🐞 微距昆虫 · 🌊 悬崖灯塔

---

## 八、踩过的坑与解法（最有价值的部分）

### 坑 1：sd.cpp 的 OpenAI 接口会**静默忽略**大部分参数 ⭐

**现象**：请求 `{"steps":4, "width":512, "height":512}`，返回的却是 1024×1024、跑了 8 步的图，耗时 57 秒。

**原因**：读源码 `examples/server/routes_openai.cpp` 的 `build_openai_generation_request()` 发现，它**只读 5 个字段**：

```
prompt / n / size / output_format / output_compression
```

**其他参数一律取服务启动时的 CLI 默认值，客户端传了也不报错，直接丢掉。**

**解法**：用 sd.cpp 的隐藏后门——把参数塞进 prompt 里：

```
<sd_cpp_extra_args>{"seed":42,"sample_params":{"sample_steps":8}}</sd_cpp_extra_args>
```

服务端有个 `extract_and_remove_sd_cpp_extra_args()` 会用正则把这段抠出来，交给 `SDGenerationParams::from_json_str()` 解析，同时把这段文字从 prompt 里删掉。分辨率则走顶层 `size: "WxH"` 字段。

**验证方式**：PNG 里嵌了 `parameters` tEXt chunk，前半行是人类可读摘要，后半行是 `SDCPP: {...}` JSON，这是**唯一可信的运行记录**。代理直接读它来覆盖回传值。

顺带做了防护：用户 prompt 里如果自己写了 `<sd_cpp_extra_args>` 块，会被先剥掉，绕不过参数校验。

### 坑 2：中文文件名的图片 404 ⭐

**现象**：中文提示词生成的图片，文件名保留中文（这是刻意设计，方便一眼看出用了什么提示词），但在网页里**显示不出来**。

**原因**：HTTP 请求行里的中文是 percent-encoded 的（`%E6%98%A5...`），而代理直接拿原始路径去拼磁盘路径，自然找不到文件。

**解法**：统一在 `_route()` 里 `unquote`。⚠️ **顺序很关键**：

```python
# 必须先解码、再做路径穿越检查
# 否则 %2e%2e%2f 解码后就是 ../，能绕过 "/" in name 的防护
return unquote(self.path.split("?", 1)[0])
```

### 坑 3：`nvidia-smi` 在 Jetson 上没用 ⭐

用 `nvidia-smi` 查显存和利用率，得到的是：

```
|   0  Orin (nvgpu)  N/A | Not Supported | N/A |
| No running processes found |
```

因为 Orin 是核显 + 统一内存，`nvidia-smi` 那套为独显设计的读法在这里读不到东西。

**正确工具**：

```bash
sudo jtop                                        # 推荐，交互式，CPU/GPU/内存/功耗/温度全有
sudo tegrastats --interval 1000                  # 原始数据，每秒一行
```

重点看 **`GR3D_FREQ`**（GPU 利用率）和 **`VDD_GPU_SOC`**（GPU 功耗）。

### 坑 4：`backend_fit.cpp:444 auto-fit: no GPU devices` 是假警报

日志里有这条 warning，看着像没用上 GPU，**实际推理走的就是 CUDA0**（`GR3D_FREQ 99%` 可证）。这是自动后端选择的探测逻辑没识别到 CUDA，纯噪声，不影响性能和结果。

### 坑 5：内存和 CPU「看着没动静」，是正常的 ⭐

这是最容易被误判的一点。实测一次 1024²/8 步生成（57.3 秒），逐秒采样：

| 阶段 | RAM | GR3D |
| --- | --- | --- |
| 采样中（约 54 s） | 17594 → 17601 MB（**只飘 7 MB**） | **99%** |
| VAE 解码（约 4 s） | 跳到 ~21084 MB（**+3.5 GB**） | 短暂 0% |
| 结束后 | 回落 17600 MB | 0% |

**为什么内存不动**：权重在首次请求时就全部加载进显存常驻了，`sd-server` 的 RSS 稳定在 **8.6 GB**，之后每张图只是复用，不再重新分配。

**为什么 CPU 不动**：12 个核里大部分 0–6%，但**总有一个核在 93–100%**——那是驱动 GPU 的宿主线程（下发 kernel、做同步）。真正算数的是 GPU。

**判断 GPU 有没有在干活，看这两个数就够了**：

- `GR3D_FREQ`（71 个采样里 53 个是 99%）
- `VDD_GPU_SOC`（采样时 **33.9 W**，空闲时 **5.4 W**）

### 坑 6：首次请求要等 127 秒

第一次请求触发模型加载（7.8 GB 从 NVMe 读进显存），约 127 秒。客户端超时要放宽到 600 秒以上。之后的请求都是常驻内存，秒级响应。

---

## 九、目录结构

```
/home/bianbu/image-generation/          (9.1 GB)
├── bin/
│   ├── sd                233 MB   命令行出图
│   └── sd-server         234 MB   HTTP 服务
├── models/zimage/
│   ├── z_image_turbo-Q4_K_M.gguf               4.98 GB
│   ├── Qwen3-4B-Instruct-2507-Q4_K_M.gguf      2.50 GB
│   └── ae.safetensors                          0.34 GB
├── src/stable-diffusion.cpp/        源码 + 编译目录
├── web/index.html                   41 KB  单页 UI
├── webui.py                         13 KB  代理 + 图库服务
├── run.sh                           便捷入口（generate / serve / bench）
├── bench/                           压测与监控脚本
│   ├── gpu_watch.sh                 生成时采样 tegrastats
│   └── steps_seed_demo.sh           步数/种子对照实验
├── output/ui/                       图库（PNG + index.json）
├── service.log                      sd-server 日志
├── webui.log                        代理日志
└── README.md                        完整部署文档（含本总结的技术细节）
```

编译命令（如需重建）：

```bash
cd src/stable-diffusion.cpp
cmake -B build -DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DSD_BUILD_EXAMPLES=ON
cmake --build build -j4
# 产物在 build/bin/sd 和 build/bin/sd-server
```

---

## 十、运维手册

### 服务管理

```bash
sudo systemctl status  imagegen    # sd-server，0.0.0.0:8080
sudo systemctl status  imageui     # webui.py，   0.0.0.0:8081
sudo systemctl restart imageui
sudo systemctl restart imagegen
```

两个单元都已 `enabled`，**开机自启**。

| 服务 | 关键配置 |
| --- | --- |
| `imagegen` | `Environment=STEPS=8 WIDTH=1024 HEIGHT=1024 PORT=8080` |
| `imageui` | `Environment=SD_API=http://127.0.0.1:8080 UI_PORT=8081` |

### 看日志

```bash
tail -f /home/bianbu/image-generation/service.log   # 推理日志，含每步 s/it
tail -f /home/bianbu/image-generation/webui.log     # HTTP 访问日志
```

### 看 GPU

```bash
sudo jtop
sudo tegrastats --interval 1000 | grep --line-buffered -oE "GR3D_FREQ [0-9]+%|RAM [0-9]+/[0-9]+MB"
/home/bianbu/image-generation/bench/gpu_watch.sh    # 生成一次并采样全程轨迹
```

### 命令行直接出图

```bash
cd /home/bianbu/image-generation
./run.sh generate "一只红色的狐狸坐在雪地里，柔和的晨光" out.png
STEPS=4 WIDTH=512 HEIGHT=512 ./run.sh generate "a red fox" small.png
```

### 调 API

```bash
curl -X POST http://192.0.2.10:8081/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"一只红色的狐狸坐在雪地里","width":512,"height":512,"steps":4,"seed":42}'
```

---

## 十一、已知问题

| 问题 | 影响 | 处置 |
| --- | --- | --- |
| `backend_fit.cpp:444 auto-fit: no GPU devices` warning | **无**，纯噪声 | 忽略，实际走 CUDA0 |
| 第 3 张连续出图时偶发 `NvMapMemAllocInternalTagged error 12` | **无**，该次解码仍成功 | 观察即可；如高频出现可加 `--vae-tiling` |
| 首次请求约 127 秒 | 客户端需放宽超时到 600 s | 保持服务常驻，不要频繁重启 |
| 1024² 下 VAE 解码瞬时 +3.5 GB 内存 | 系统有 12 GB available，安全 | 避免与其他大内存服务同时跑 |

---

## 十二、性能优化结论（如果还要更快）

按收益排序：

1. **降步数** —— 8 → 4 步直接省一半时间（57.7 s → 41.9 s），画质仍可用
2. **降分辨率** —— 1024² → 512² 快约 5 倍（59 s → 12 s），试提示词阶段必用
3. **`--diffusion-fa`** —— 已开，必开项
4. **`--cfg-scale` 保持 1.0** —— 调高只会更慢更差

**不建议的方向**：加步数（>8 步收益极小）、上更大分辨率（显存和 VAE 缓冲会顶到上限）。

瓶颈在 DiT 采样（占 95% 时间），且已经跑满 GR3D（99%），所以**没有软件层面的优化空间了**——要更快只能换硬件或换更小的模型。

---

## 附录：关键数字速查

| 指标 | 数值 |
| --- | --- |
| GPU | Jetson AGX Orin，CUDA 12.6，CC 8.7 |
| 模型 | Z-Image-Turbo 6B DiT + Qwen3-4B TE + Flux VAE |
| 量化 | Q4_K_M，合计 7.8 GB |
| 显存占用 | 权重 7288 MB + DiT 缓冲 690 MB + VAE 缓冲 6657 MB |
| 常驻 RSS | 8.6 GB |
| 出图速度 | 512²/8步 12.3 s　·　1024²/8步 57.7~59.2 s |
| 单步成本 | 512² 1.38 s/步　·　1024² 6.42 s/步 |
| GPU 利用率 | 99%（GR3D） |
| GPU 功耗 | 33.9 W（生成）/ 5.4 W（空闲） |
| 采样期内存波动 | 仅 7 MB |
| 可复现性 | 同 seed + 同参数 → PNG SHA256 完全一致 |
