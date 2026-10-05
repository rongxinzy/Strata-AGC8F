> This source is the isolated upstream `6f32ec0` integration checkpoint (243). Historical measurements below belong to their stated older commits; host validation alone does not establish CUDA parity or performance.

# Strata-AGC8F

本分支用于 RongxinZY agc8f 的源码优化与实测。上游为
[Niko1221/Strata](https://github.com/Niko1221/Strata)，当前实验代码基于
`1678de333d0e0711bc414ad992b640e1a37dd814`（0.1.34）。`main` 保留上游快照，
`agc8f` 保存本机已测改造；新优化经独立分支和设备验收后集成。MIT 许可及上游署名保留。

## 目标设备与约束

- 8 × RTX 4060 Ti 16GB，Hygon C86-4G 16物理核/32线程，125GiB RAM。
- AVX2/FMA 可用，AVX-512/VNNI 不可用。八卡共享 PCIe4×8 CPU 上行。
- 实际八卡同时 H2D/D2H 总带宽约13.48/13.54GB/s，不能将单卡带宽乘8。
- GSQ-RCO Qwen3.8-Flash-Next IQ3_S，主模型、MTP权重与量化保持固定。
- 服务器仅监听 localhost。性能实验保持FP16 KV、32768上下文、spec4/min_p0.5、
  static cache、256输出预算；不通过删除专家、PLE或替换模型获得速度。

## 已实现：完整 prefill 流水

`src/prefill/prefill.cpp` 和 `include/strata/prefill/pipeline.hpp` 实现请求级stage流水：
每个stage一个owner、容量2的FIFO、既有两个pinned交接槽。消费者H2D完成后ack，
生产者才可覆写；请求结束join/drain。默认关闭，精确环境值开启：

```sh
STRATA_PREFILL_PIPELINE=1
```

stage的chunk不一致时退回原路径。`STRATA_PREFILL_PIPELINE_TRACE=1`记录host事件；
`STRATA_PREFILL_PIPELINE_AUDIT=1`记录部分状态指纹。两者只用于诊断，不用于正式计时。
诊断dump共享静态状态的旧开关与新流水不兼容，源码会拒绝组合。

2026-10-03，未用于本轮选参的9个输入做A/B/B/A重启复测，三个语料×三输入档，
每臂每档6请求。A为原有路径chunk8192，B为流水chunk2048，同一个独立二进制：

| 输入 | A端到端均值 | B端到端均值 | 耗时减少 |
|---|---:|---:|---:|
| 1K | 5.659s | 5.587s | 1.3% |
| 4K | 8.216s | 6.989s | 14.9% |
| 16K | 17.823s | 10.306s | 42.2% |

同chunk1024仅改变流水开关的独立校准中，16K prefill为20.752→6.727s，
decode基本不变。最终A/B同时改变chunk和buffer策略：A末卡借专家槽，B无loan；
原有chunk选参来自早先原构建/profile开启结果，不宣称全局最优。
完整均值、接受率和文本一致率见 [结果JSON](agc8f/2026-10-03-pipeline-results.json)。

这些是运行耗时诊断。五题sanity仍4/5，算式137×29+41错答4032（正确4014）；
A/B跨臂输出文本只有10/36一致。部分state比较只覆盖chunk1024的GDN、活动PLE、
各stage末有效residual一行，不覆盖QSA/MTP KV或全部activation。
GPU取消、cache/checkpoint恢复、非零起点、并发原型与长期稳定性尚未验收。
CPU绑核候选出现跨启动部分state差异，未集成。

## 新候选：末 stage batched MTP prefill

多卡 serve 原先逐 token 填充 MTP KV。新路径复用实际末 stage 的
`Prefill::draft_kv`，按值捕获对象指针；只在精确环境值下启用：

```sh
STRATA_MTP_BATCH_MULTI=1
```

默认保持原多卡行为；已有 `STRATA_MTP_BATCH=0` 仍禁用 batch。
不支持的条件回退，真实错误直接传播。主模型与 MTP 权重、量化和 KV 编码保持固定；
batch 与逐 token 计算顺序不同，不能假定逐位一致。

`STRATA_MTP_PREFILL_AUDIT=1` 配合每次运行独占的
`STRATA_MTP_PREFILL_AUDIT_DIR`，导出本次 callback 新写的活动 MTP K/V。
只支持未旋转、resident paged FP16，按真实页表提取，排除无效早期行和页 padding；
每次 D2H 不超过64KiB，记录真实小端 FP16 bits、区间、页映射和有限值统计。
先写 partial、最后发布 complete 元数据；已存在的 chunk 目录拒绝覆盖。
审计同步、复制和写盘全部排除正式计时。

同新构建的八卡两臂审计中，长输入22个 chunk 确认从 token 切到 batched，
K/V 全部有限；最大相对 L2 差异分别约0.058%和0.254%。单独2112/2113/2114输入
确认2048后63 token尾部回退、64/65 token尾部走batch。长输入诊断中覆盖的5请求主状态指纹一致；单独tail诊断4/4请求在末卡
GDN45/46及residual/GDN summary指纹有差异，文本4/4相同，原因未定位。
不能称全面状态通过。原五题 sanity 仍4/5。此处不覆盖主 QSA KV、
cache/checkpoint恢复、CUDA sanitizer、取消或完整质量评估。

性能校准18请求与 A/B/B/A 留出36请求使用独立新语料，同二进制、八卡、
pipeline=1、chunk2048，只有新开关不同；关闭所有审计/profile/trace。每臂每档
三种语料各两次，实际输出均256 token、prompt reuse为0。

| 输入 | Prefill均值(s)，OFF → ON | 减少 | 客户端总耗时均值(s)，OFF → ON | 减少 | Decode均值(s)，OFF → ON |
| --- | ---: | ---: | ---: | ---: | ---: |
| 约1K | 1.3081 → 1.2664 | 3.19% | 5.6056 → 5.4053 | 3.57% | 4.1350 → 4.1306 |
| 约4K | 2.6942 → 2.6168 | 2.87% | 6.9882 → 6.9651 | 0.33% | 4.2687 → 4.3223 |
| 约16K | 5.7261 → 5.4266 | 5.23% | 10.0775 → 9.6935 | 3.81% | 4.2581 → 4.1709 |
| 合计 | 3.2428 → 3.1033 | 4.30% | 7.5571 → 7.3546 | 2.68% | 4.2206 → 4.2079 |

总体prefill减少4.30%、客户端总耗时减少2.68%；decode只变化−0.30%，小于
两次相同开关启动间−2.68%/+1.50%的变化，不能宣称稳定decode提升。
1K OFF两次启动首可见延时变化24.02%，短请求收益也须保留该波动。
留出文本OFF自一致2/9、ON3/9，跨开关16/36相同；接受率及输出内容会影响decode。
这是一轮ABBA的运行耗时诊断，不是相同输出/全面质量受控加速。
实现构建固定`cb22b89933cce7189b21b99ce59ba5e668707959`，后续仅补文档，
[原始精度汇总及限制](agc8f/2026-10-03-mtp-results.json)。不能将这次增量与
不同语料、不同基线的此前42.2%流水结果相乘或直接相加。

## 各 stage 的 decode profile

`STRATA_VERIFY_PROFILE` 在构建 graph 前设置。服务在请求的 decode 边界 reset/drain
所有 Verifier，输出 `STRATA_VERIFY_STAGE_PROFILE`：stage/device、层范围、窗口数、
已采样窗口、G2遗漏和 coverage。报告在 decode 计时及 commit wait 之后进行；
关闭 profile 不增加 GPU 工作。

类别沿用原时间戳口径，`hc0 norm` 在主 stage 包含 PLE/历史及其它工作，
不能当纯 norm kernel。不同 GPU 的局部时间不能推成跨卡完整关键路径。
本轮只有一轮开启新候选的独立 profile，不宣称 profile A/B 加速。
排除warmup后4请求、3520个stage窗口：verify占decode计时桶87.67%，draft10.01%。
VRAM专家混合链占23.98%、HC第二读/router混合跨度10.68%、末卡head混合跨度5.45%。
这些指向后续细分内核边界；各桶不能当单个kernel或完整跨卡关键路径。

## 构建与运行

使用CUDA支持构建，ggml固定在 `3cf03257f219afbe7334045ff7c6a06ac68c627d`，
不要原地更新已有实验构建。以下路径须替换为本机的独立目录：

```sh
cmake -S . -B build-agc8f -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_BUILD_TESTS=OFF -DSTRATA_GGML_DIR=/path/to/pinned-llama.cpp
cmake --build build-agc8f --target strata -j 8
```

原版setup下载的release二进制不含本分支改造。已有模型完成pack和MTP准备后，
API配置的`exe`须指向本分支构建，`gpu`为0–7，`layer_split`为`auto`，`env`中设
`STRATA_PREFILL_PIPELINE=1`、`STRATA_VERIFY_DEVICE_PLAN=0`；CLI参数使用
`--prefill 2048 --kv fp16 --spec 4 --spec-min-p 0.5 --adapt-swaps 0 --no-spec-split`。
所有prompt/conversation cache关闭是本次测量条件，不是生产推荐。

```sh
taskset -c 0-15 python -m serve.server --engine strata \
  --config /path/to/agc8f-config.json --host 127.0.0.1 --port 18080
```

正式计时不设置TIMING/VERIFY_PROFILE/AUDIT/TRACE。部分profile开关按环境变量存在判断，
设0仍启用，应unset。仅停止自己创建且身份已核对的进程。

## 回归与下一步

队列/slot/取消模拟及audit helper已通过host普通、ASan/UBSan、TSan；CUDA构建和有限设备
对照完成。host测试不能代替GPU内存安全、数值或服务恢复验收：

```sh
cmake -S . -B build-agc8f-host -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_AGC8F_TESTS=ON
cmake --build build-agc8f-host --target prefill_pipeline_test prefill_audit_test mtp_prefill_audit_test
ctest --test-dir build-agc8f-host -R '^(prefill_(pipeline|audit)|mtp_prefill_audit)_test$' --output-on-failure
```

后续先定位tail末卡状态指纹差异，再细分native专家gate/up、SwiGLU、量化、down，
以及HC/router和末卡head混合跨度；随后验收小T融合、native投影裁剪与GPU路由分组。P2P需核实switch/ACS实际路径后设计；既有13.5GB/s共享上行不能靠软件
变成八条独立通路。新候选默认关闭，数值诊断与性能测量分离，所有失败保留。
