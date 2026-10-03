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
cmake --build build-agc8f-host --target prefill_pipeline_test prefill_audit_test
ctest --test-dir build-agc8f-host -R '^prefill_(pipeline|audit)_test$' --output-on-failure
```

后续源码工作先测末stage MTP和逐stage decode，再验证batch MTP、native投影裁剪与
GPU路由分组。P2P需核实switch/ACS实际路径后设计；既有13.5GB/s共享上行不能靠软件
变成八条独立通路。新候选默认关闭，数值诊断与性能测量分离，所有失败保留。
