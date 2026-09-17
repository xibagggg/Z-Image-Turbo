# Z-Image-Turbo 部署说明（Jetson AGX Orin ）

> 本文里的 `192.0.2.x` 是 RFC 5737 的文档专用地址段，实际部署时替换成你的设备 IP。

> 提示词**中英文都可以**，直接写中文效果很好（详见文末《中文提示词》一节）。
>
> 想通读整个项目的来龙去脉（选型、性能、踩过的坑、运维命令、关键数字速查），看同目录的 **《端侧文生图部署总结.md》**。

## 硬件与软件

| 项目 | 值 |
|---|---|
| 机型 | Jetson AGX Orin（`orin-agx-01`） |
| OS / BSP | Ubuntu 22.04.5 LTS，L4T / JetPack R36.4.7（aarch64） |
| CPU | 12× Cortex-A78AE @2.2GHz |
| 内存 | 29 GiB RAM + 14 GiB swap（**统一内存**，显存即内存） |
| 存储 | 467 GB NVMe，可用约 110 GB |
| GPU | Orin iGPU，`compute capability 8.7`，CUDA 12.6 |
| 显存 | `nvidia-smi` / ggml 报告 30696 MiB（与系统内存共享） |

所有文件集中在 `/home/bianbu/image-generation/`。

## 目录结构

```
/home/bianbu/image-generation/
├── bin/sd                  # sd-cli（stable-diffusion.cpp，带 CUDA 后端）
├── bin/sd-server           # HTTP 服务
├── models/zimage/
│   ├── z_image_turbo-Q4_K_M.gguf              4.98 GB   DiT 主模型
│   ├── Qwen3-4B-Instruct-2507-Q4_K_M.gguf     2.50 GB   文本编码器
│   └── ae.safetensors                         0.34 GB   Flux VAE
├── output/                 # 出图
├── src/stable-diffusion.cpp/   # 源码与 build 目录
├── run.sh                  # 便捷入口
├── imagegen.service        # systemd 单元
└── README.md
```

模型来源：
- 主模型：`https://www.modelscope.cn/models/jayn7/Z-Image-Turbo-GGUF` 的 `z_image_turbo-Q4_K_M.gguf`
- 文本编码器：`unsloth/Qwen3-4B-Instruct-2507-GGUF` 的 `Q4_K_M`
- VAE：`black-forest-labs/FLUX.1-schnell` 的 `ae.safetensors`

## 编译方法

```bash
git clone https://github.com/leejet/stable-diffusion.cpp
cd stable-diffusion.cpp
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSD_BUILD_EXAMPLES=ON \
  -DSD_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build -j$(nproc)
```

关键点：
- `CMAKE_CUDA_ARCHITECTURES=87`：Orin 是 sm_87，不指定会编译出错误的 SASS。
- 产物约 233 MB（静态链了 CUDA runtime）。

## 实测性能

命令行单次出图（`sd-cli`，包含模型加载）：

| 分辨率 | 步数 | 采样 s/it | 采样总耗时 | VAE 解码 | 端到端 |
|---|---|---|---|---|---|
| 512×512 | 8 | 1.38 | 14.2 s | 1.6 s | **17.3 s** |
| 1024×1024 | 4 | 6.41 | 25.6 s | ~6 s | **41.9 s** |
| 1024×1024 | 8 | 6.46 | 54.2 s | 7.3 s | **62.4～62.8 s** |

分解（1024×1024 / 8 步）：
- 文本编码（Qwen3-4B）：0.94 s
- 采样：54.18 s（8 × 6.4 s）
- VAE 解码：7.25 s

HTTP 服务（模型常驻，`/v1/images/generations`，1024×1024 / 8 步）：
- 首次请求 127 s（含加载），之后 **稳定 57.7 / 59.2 s**
- 文本编码降到 **0.05 s**（KV/嵌入已常驻）

常驻资源占用（`tegrastats` 采样）：
- 峰值 RAM 使用 **19.2 GB / 30.7 GB**
- 峰值 `GR3D` 占用率 **99%**
- 权重占用（服务日志）：文本编码器 2376 MB + DiT 4752 MB + VAE 160 MB = **7288 MB VRAM**
- DiT 计算缓冲 690 MB，VAE 计算缓冲 6657 MB

## 与 K3对比

| | K3 Pico-ITX (192.0.2.20) | Jetson AGX Orin (192.0.2.10) |
|---|---|---|
| Z-Image-Turbo 1024×1024 | **474.74 s/步**（约 63 分钟/张） | **6.46 s/步**（62 秒/张） |
| 能否跑起来 | 两次被 OOM killer 杀死（峰值 7.2 GB，15 GiB 不够） | 正常，峰值 19.2 GB |
| 另一基线：SD-Turbo 512×512 | 1 步 102 s / 4 步 188 s | — |

即同一条 prompt、同一个模型，Orin 比 K3 快约 **73 倍**，K3 上是直接 OOM 跑不出来的状态。

## 用法

```bash
cd /home/bianbu/image-generation

# 出图（1024x1024，8 步，约 62 秒）
./bin/sd --diffusion-model models/zimage/z_image_turbo-Q4_K_M.gguf \
         --vae models/zimage/ae.safetensors \
         --llm models/zimage/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
         -p 'a red fox sitting in a snowy forest, soft morning light' \
         --cfg-scale 1.0 --steps 8 -W 1024 -H 1024 --seed 42 --diffusion-fa \
         -o output/fox.png

# 或走便捷脚本
./run.sh generate "your prompt"
./run.sh serve        # 起 HTTP 服务（:8080）
./run.sh devices      # 看后端识别情况
```

### HTTP API

```bash
curl -X POST http://192.0.2.10:8080/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a red fox in snow","steps":8,"width":1024,"height":1024,"seed":1}'
```

返回 `b64_json` 格式的 PNG（约 2.1 MB），落盘可直接 `base64 -d`。

## 参数建议

- **`--diffusion-fa`**：开 FlashAttention，1024×1024 下省显存也更快，建议常开。
- **`--cfg-scale 1.0`**：Z-Image-Turbo 是蒸馏模型，不需要 CFG；给大于 1 的值只会变慢且画质变差。
- **步数**：4 步已经可用（41.9 s），8 步画质更稳（62.4 s）。低于 4 步开始出结构错误。
- **分辨率**：512×512 只要 17.3 s，一次性批量试 prompt 时用这个更划算。
- **`-W/-H` 必须写全**，不传会用默认值导致构图与预期不符。

## 已知问题

- `sd-server` 日志里有 `backend_fit.cpp:444 - auto-fit: no GPU devices; using the default backend` 的 warning。这只是自动后端选择的探测逻辑没识别到 CUDA，**实际推理仍然走 CUDA0**（日志中 `params backend buffers ... on CUDA0`、`GR3D 99%` 可证），不影响性能和结果。
- 服务连续出第 3 张图时出现过一条 `NvMapMemAllocInternalTagged: 1075072515 error 12`（统一内存下的临时分配失败），但该次解码仍成功完成（日志 `latent 1 decoded, taking 6.97s`）。若后续高频调用，建议给 `--vae-tiling` 或降低并发；`--diffusion-fa` 已经是必开项。
- 首次请求会触发模型加载，约 127 秒，客户端超时要放宽到 600 秒以上。

---

# 网页 UI（Z-Image Studio）

浏览器直接打开：**http://192.0.2.10:8081**

单页应用，无需安装任何东西，手机上也能开。整个 UI 只有一个 `web/index.html`（约 40 KB，纯原生 JS + CSS，无外部 CDN 依赖，断网也能用）。

## 架构

```
浏览器 ──8081──> webui.py（本地代理 + 图库）──8080──> sd-server（sd.cpp）
```

为什么要中间加一层 `webui.py`？因为 `sd-server` 的 OpenAI 兼容接口有个硬限制：
`examples/server/routes_openai.cpp` 里的 `build_openai_generation_request()` **只读**
`prompt` / `n` / `size` / `output_format` / `output_compression` 这几个字段，
其余参数一律取服务启动时的 CLI 默认值，客户端传了也会被静默忽略。

所以 `webui.py` 做了三件事：

1. **参数注入**：把 `steps` / `seed` 拼进 prompt 里，走 sd.cpp 的隐藏开关
   `<sd_cpp_extra_args>{"seed":N,"sample_params":{"sample_steps":S}}</sd_cpp_extra_args>`；
   分辨率走顶层 `size: "WxH"` 字段。用户在网页上调的每一个参数都真的生效。
2. **结果回读**：生成的 PNG 里 sd.cpp 会写一段 `parameters` tEXt chunk，
   前半行是人类可读摘要、后半行是 `SDCPP: {...}` JSON。代理把它解出来，
   以**服务器自报的参数**为准覆盖回传字段（而不是断言我们请求了什么）。
   点开大图底部那行「服务器记录：Steps: 4, CFG scale: 1.000000, …」就是它。
3. **图库落盘**：图片和索引存设备上的 `output/ui/`（`index.json`，上限 500 条），
   不用浏览器 localStorage —— 后者配额只有约 5 MB，两张 1024² PNG 就爆了。

## 功能

| 区域 | 说明 |
| --- | --- |
| 提示词 | 6 个一键预设；实时字数；`Ctrl`+`Enter` 直接出图 |
| 分辨率 | 512² / 768² / 1024² / 自定义，切换时下方实时显示该档位的 s/步 |
| 采样步数 | 1–20 滑杆，标了「最快 / 推荐 / 更精细」 |
| 种子 | 可手填，或 🎲 随机；同种子可复现（已用 SHA256 比对验证过） |
| 连续生成 | 1–4 张，种子依次 +1，方便挑图 |
| 进度 | 环形进度条按实测 s/步 估算阶段（文本编码 → 采样中 Step k/N → VAE 解码） |
| 作品集 | 瀑布流网格，悬停浮层，点开大图看完整参数，支持复用参数 / 下载 / 复制 / 删除 |

CFG 固定在 1.0 且**不在 UI 上暴露**：Z-Image-Turbo 是 guidance-distilled 模型，
调高只会更慢更差。

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | UI 页面 |
| GET | `/api/health` | 健康检查，同时给前端做服务在线探测 |
| GET | `/api/gallery` | 图库索引 |
| GET | `/images/<name>` | 单张图（带路径穿越防护） |
| POST | `/api/generate` | `{prompt, width, height, steps, seed}`，返回带 `applied` 的结果 |
| DELETE | `/api/gallery/<name>` | 删图 |

`webui.py` 会把用户 prompt 里自带的 `<sd_cpp_extra_args>` 块先剥掉，防止绕过参数校验。

## 服务管理

两个 systemd 单元，都已 `enable`，开机自启：

```bash
sudo systemctl status imagegen   # sd-server，0.0.0.0:8080
sudo systemctl status imageui    # webui.py，   0.0.0.0:8081

sudo systemctl restart imageui
tail -f /home/bianbu/image-generation/webui.log
```

`imageui.service` 里通过 `Environment=SD_API=http://127.0.0.1:8080` 指向后端，
`UI_PORT` / `GEN_TIMEOUT` 也可用环境变量覆盖。

## 前端时间估算的实测常量

进度环不是"死循环转"，是按这台上实测出来的曲线插值，所以进度和真实进度基本对得上：

```js
const stepSec = (s) => s <= 512 ? 1.38 * (s / 512) ** 2
                      : s <= 1024 ? 1.38 + (6.42 - 1.38) * (s - 512) / 512
                      : 6.42 * (s / 1024) ** 2;
const vaeSec  = (s) => s <= 512 ? 1.60 * (s / 512) ** 2
                      : s <= 1024 ? 1.60 + (7.25 - 1.60) * (s - 512) / 512
                      : 7.25 * (s / 1024) ** 2;
const TEXT_SEC = 1.0;
```

---

# 中文提示词

**可以直接用中文写，不需要翻译成英文。**

Z-Image-Turbo 的文本编码器是 **Qwen3-4B-Instruct-2507**（通义千问系列），中文本来就是它的母语级能力，不是"凑合能用"。网页 UI 里的 6 个预设已经全部换成中文，方便直接改。

## 实测

同样参数（512×512、8 步、seed 42）分别用中英文跑：

| 提示词 | 耗时 |
| --- | --- |
| `一只红色的狐狸坐在雪地里，柔和的晨光，细节丰富，电影感` | 12.29 s |
| `a red fox sitting in a snowy forest, soft morning light, highly detailed, cinematic` | 12.28 s |

耗时基本一致——中文的 token 开销在这个量级上可以忽略，不用担心"中文会慢"。

质量上两张都能出，中文那张毛发光影、构图、景深都很稳。更值得说的是模型对中国特有概念的先验：

> `春天的苏州园林，白墙黛瓦，桃花纷飞，午后柔和的阳光洒在回廊上，胶片摄影质感`

出的是白墙、黛瓦、木质回廊、漏窗花窗、屋顶瓦当，建筑语汇全部正确——这类题材用英文反而容易画成日式庭院。

## 写法建议

中文和英文的写法逻辑一样，堆"画面 + 光线 + 质感 + 镜头感"这四类词效果最好：

```
<主体>，<光线>，<材质/细节>，<镜头或风格>
```

例如：

- `一只红色的狐狸坐在雪地里，柔和的晨光，毛发细节丰富，浅景深，电影感`
- `春天的苏州园林，白墙黛瓦，桃花纷飞，午后柔和的阳光洒在回廊上，胶片摄影质感`
- `中国水墨画，远山如黛，云雾缭绕，大量留白，意境悠远`

中英混写也可以（例如中文描述画面 + 英文写 `cinematic lighting, 85mm`），Qwen3 对混合输入处理得很好。

## 两个实现细节

- 文件名会保留中文（`slugify` 的正则里带 `\u4e00-\u9fff` 区间），所以图库里一眼能看出每张图用的什么提示词。例如
  `20260917-095712-春天的苏州园林-白墙黛瓦-桃花纷飞-午后柔和的阳光洒在回廊上-胶片摄影质感-43.png`
- 代理需要对 URL 做 percent-decode：HTTP 请求行里的中文文件名是 `%E6%98%A5...` 形式，
  如果直接拿去拼磁盘路径就会 404。`webui.py` 里统一在 `_route()` 里 `unquote`，
  且**先解码再做出路径穿越检查**（否则 `%2e%2e%2f` 能绕过去）。
