# Z-Image-Turbo on Jetson AGX Orin

在 **Jetson AGX Orin** 上用 [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) 部署 **Z-Image-Turbo**（6B DiT）文生图服务，配一个自带图库的网页 UI，支持**中文提示词**。

本仓库只放**部署代码、脚本与文档**，不含模型权重（下面有下载地址）。

## 实测数据

| 分辨率 | 步数 | 命令行 `sd` | 服务常驻 `sd-server` |
| --- | --- | --- | --- |
| 512×512 | 8 | 17.3 s | 12.3 s |
| 1024×1024 | 4 | 41.9 s | — |
| 1024×1024 | 8 | 62.4 s | 57.7 ~ 59.2 s |

1024×1024 / 8 步的耗时拆解：

| 阶段 | 耗时 | 占比 |
| --- | --- | --- |
| Qwen3-4B 文本编码 | 0.05 s（热） / 0.94 s（冷） | ~0.1% |
| DiT 迭代去噪 | **54.18 s** | **~95%** |
| VAE 解码 | 5.47 ~ 7.25 s | ~10% |

GPU 在这 54 秒里 `GR3D_FREQ` 有 53/71 个采样点跑到 **99%**，`VDD_GPU_SOC` 从空闲 5.4 W 拉到 33.9 W。**瓶颈完全在 GPU 算力上，软件层面没有优化空间**。

权重全部驻留显存，系统内存里一份副本都不留：

```
total params memory size = 7288.11MB (VRAM 7288.11MB, RAM 0.00MB)
  ├─ text_encoders    2375.91MB (VRAM)   ← Qwen3-4B
  ├─ diffusion_model  4752.20MB (VRAM)   ← 6B DiT
  └─ vae               160.00MB (VRAM)   ← ae.safetensors
```

## 示例

下面这些图都是用本仓库的网页 UI 在这台 Orin 上生成的。每张 PNG 都保留了 sd.cpp 写入的 `parameters` 元数据块，可以看到真实的 prompt / steps / cfg / seed。

| | | |
| --- | --- | --- |
| ![雪地红狐](samples/fox-1024.png)<br>`a red fox sitting in a snowy forest, soft morning light`<br>1024² · 8 步 | ![苏州园林](samples/suzhou-garden.png)<br>`春天的苏州园林，白墙黛瓦，桃花纷飞，午后柔和的阳光洒在回廊上，胶片摄影质感`<br>512² · 8 步 | ![微距瓢虫](samples/macro-ladybug.png)<br>`微距摄影，一只瓢虫停在沾满露珠的叶子上，浅景深，自然光`<br>1024² · 8 步 |
| ![晨雾山脉](samples/misty-mountains.png)<br>`misty mountain ridge at sunrise, layered peaks, volumetric fog`<br>1024² · 8 步 | ![小熊猫](samples/red-panda.png)<br>`a red panda eating bamboo, soft studio light, shallow depth of field`<br>512² · 8 步 | ![春日庭院](samples/courtyard-spring.png)<br>`a quiet Japanese courtyard in spring, cherry blossoms falling`<br>1024² · 8 步 |

**中文提示词可以直接用，不需要翻译成英文。** 文本编码器是 Qwen3-4B-Instruct-2507（通义千问系列），中文是它的母语级能力。同样参数下中文 12.29 s / 英文 12.28 s，耗时基本一致。而且模型对中国特有概念的先验更准——上面那张苏州园林出的是白墙、黛瓦、木质回廊、漏窗花窗、屋顶瓦当，建筑语汇全部正确，这类题材用英文反而容易画成日式庭院。

## 硬件与软件

| 项目 | 值 |
| --- | --- |
| 机型 | Jetson AGX Orin |
| OS / BSP | Ubuntu 22.04.5 LTS，L4T / JetPack R36.4.7（aarch64） |
| CPU | 12× Cortex-A78AE @2.2GHz |
| 内存 | 29 GiB RAM + 14 GiB swap（**统一内存**，显存即内存） |
| GPU | Orin iGPU，`compute capability 8.7`，CUDA 12.6 |
| 推理框架 | stable-diffusion.cpp `59c23bc`（2026-09-15） |

## 模型

| 文件 | 大小 | 作用 | 来源 |
| --- | --- | --- | --- |
| `z_image_turbo-Q4_K_M.gguf` | 4.98 GB | 6B DiT 主模型 | [jayn7/Z-Image-Turbo-GGUF](https://www.modelscope.cn/models/jayn7/Z-Image-Turbo-GGUF) |
| `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | 2.50 GB | 文本编码器 | [unsloth/Qwen3-4B-Instruct-2507-GGUF](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF) |
| `ae.safetensors` | 0.34 GB | Flux VAE | [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) |

统一放到设备上的 `models/zimage/` 下。

## 快速开始

### 1. 编译 stable-diffusion.cpp

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

`CMAKE_CUDA_ARCHITECTURES=87` 必须写对，Orin 是 **sm_87**，不指定会编译出错误的 SASS。产物约 233 MB。

### 2. 起服务

```bash
./deploy/run.sh serve          # sd-server 监听 :8080
./deploy/run.sh generate "一只红色的狐狸坐在雪地里，柔和的晨光"
./deploy/run.sh devices        # 看后端识别情况
```

`run.sh` 里的参数是调好的，**两个参数不要动**：

```bash
--cfg-scale 1.0     # Z-Image-Turbo 是 guidance-distilled 模型，引导已烧进权重
--diffusion-fa      # 开 FlashAttention，1024² 下快 38% 且避免 OOM
```

### 3. 起网页 UI

```bash
SD_API=http://127.0.0.1:8080 UI_PORT=8081 python3 deploy/webui.py
```

打开 `http://<设备IP>:8081`。UI 是纯 stdlib 实现（设备上没有想依赖的 pip 包），功能包括预设提示词、分辨率/步数/种子控制、进度状态、服务端图库（masonry 网格 + 单图灯箱）。

用 systemd 常驻：把 `deploy/imagegen.service` 和 `deploy/imageui.service` 拷到 `/etc/systemd/system/`，改掉里面的路径和 `User=`，然后 `systemctl enable --now imagegen imageui`。

## 两个必须知道的坑

### 1. API 只认 prompt/n/size，其他参数要靠"走私"

stable-diffusion.cpp 的 OpenAI 兼容路由（`examples/server/routes_openai.cpp`）**只读** `prompt` / `n` / `size` / `output_format` / `output_compression`，其余全部来自服务端 CLI 默认值。唯一的逃生口是在 prompt 里塞一个：

```
<sd_cpp_extra_args>{"seed":42,"sample_params":{"sample_steps":8}}</sd_cpp_extra_args>
```

这个块会被剥离出文本并解析进 `SDGenerationParams`。`webui.py` 就是靠它传递 seed 和步数的。

### 2. `--cfg-scale` 忘了写不会报错，会默默慢一倍

`sd_sample_params_init()` 里 `txt_cfg` 的默认值是 **7.0**。而 `resolve_guidance()` 的逻辑是：

```cpp
if (guidance->img_cfg != guidance->txt_cfg) {
    *use_uncond = true;      // 只要 cfg != 1.0 就多跑一遍无条件前向
}
```

也就是**任何不等于 1.0 的 cfg 都会让每一步计算量翻倍**。实测（512²/8 步，预热后交替 A/B）：

| `--cfg-scale` | 采样耗时 |
| --- | --- |
| 1.0 | 12.8 s |
| 3.0 | 23.9 s |
| 7.0 | 23.9 s |

注意 3.0 和 7.0 耗时完全相同——代价来自"是否开启无条件分支"这个开关，与数值大小无关。

顺带一提，`--guidance`（distilled guidance）**对 Z-Image 是空操作**：`src/model/diffusion/z_image.hpp` 全文 0 处 `guidance`，`Z_image` 的 forward 签名里根本没有这个输入。实测 `--guidance 3.5` 与 `--guidance 10.0` 出图逐像素完全相同（262144 个像素 0 处不同）。PNG 元数据里那行 `Guidance: 3.500000` 只是请求参数的回执，不代表它生效了。

## 目录结构

```
.
├── deploy/
│   ├── webui.py                 # 网页 UI + 图库服务（stdlib only）
│   ├── web/index.html           # 单页前端
│   ├── run.sh                   # 便捷入口（generate / serve / devices / bench）
│   ├── imagegen.service         # sd-server 的 systemd 单元
│   ├── imageui.service          # webui.py 的 systemd 单元
│   ├── gpu_watch.sh             # 生成一次并采样全程 tegrastats 轨迹
│   ├── steps_seed_demo.sh       # 步数 / 种子 对照实验
│   ├── make_contact_sheet.py    # 把对照实验拼成带标注的对比图
│   ├── probe_api.py             # 探测 API 到底认哪些参数
│   └── probe_meta.py            # 读 PNG 元数据，确认服务端实际用了什么
├── docs/
│   ├── deployment-jetson-orin.md  # 部署说明（含完整参数建议与已知问题）
│   └── summary-zh.md              # 完整的选型 / 性能 / 踩坑总结
├── tools/
│   └── rcmd.py                  # 通过 SSH 在设备上跑命令、传文件
└── samples/                     # 示例出图（保留 PNG 参数元数据）
```

## 文档

- **[docs/deployment-jetson-orin.md](docs/deployment-jetson-orin.md)** —— 部署说明：编译、参数建议、系统服务管理、已知问题。
- **[docs/summary-zh.md](docs/summary-zh.md)** —— 完整总结：选型过程与结论、实测性能、中文提示词实测、6 个踩过的坑及解法、运维手册、关键数字速查。

## 已知问题

- **`backend_fit.cpp:444 - auto-fit: no GPU devices; using the default backend`** —— 这是自动后端选择的探测逻辑没识别到 CUDA，**属于误报**。实际推理仍走 CUDA0（日志里 `params backend buffers ... on CUDA0` 和 `GR3D 99%` 可证），不影响性能和结果。
- **`NvMapMemAllocInternalTagged: 1075072515 error 12`** —— 连续出第 3 张图时偶尔出现的一条统一内存临时分配失败，但该次解码仍成功完成。若高频调用建议降低并发。
- **首次请求约 127 秒** —— 含模型加载。客户端超时要放宽到 600 秒以上。
- **`nvidia-smi` 在这台机器上没用**（集成 GPU + 统一内存，只会显示 "Not Supported"）。要看 GPU 请用 `tegrastats`，关注 `GR3D_FREQ`（利用率）和 `VDD_GPU_SOC`（功耗），或者装 `jtop`。

## 许可

[MIT](LICENSE)。模型权重各自的许可请以原始发布页为准。
