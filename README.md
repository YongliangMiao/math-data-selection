# LLaMA-Factory / Select2Reason 使用说明（README）

---

## 目录

* [快速开始](#快速开始)
* [详细步骤](#详细步骤)

  * [1. 克隆与进入目录](#1-克隆与进入目录)
  * [2. 创建虚拟环境](#2-创建虚拟环境)
  * [3. 安装依赖](#3-安装依赖)
  * [4. 下载模型（ModelScope）](#4-下载模型modelscope)
  * [5. 配置训练 YAML](#5-配置训练-yaml)
  * [6. 训练](#6-训练)
  * [7. 测评](#7-测评)
* [常见问题](#常见问题)
* [参考目录结构](#参考目录结构)

---


## 快速开始

```bash
# 进入工程目录
cd LLaMA-Factory

# 1) 创建环境
conda create -n math python=3.10 -y
conda activate math

# 2) 安装工程与依赖
pip install -e .
pip install modelscope
MAX_JOBS=4 pip install flash-attn --no-build-isolation
pip install deepspeed==0.16.9
# 仅在测评阶段需要
pip install vllm

# 3) 下载模型（将 <MODEL_CACHE_DIR> 替换为你的模型缓存目录）
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-Math-7B-Instruct', cache_dir='<MODEL_CACHE_DIR>')"

# 4) 编辑 YAML：examples/train_full/select2reason.yaml
#    - model_name_or_path: <MODEL_CACHE_DIR>/Qwen/Qwen2.5-Math-7B-Instruct/
#    - output_dir: <YOUR_OUTPUT_DIR>  ##训练模型的保存目录
#    - logging_dir: <YOUR_LOG_DIR>     ##训练模型的保存目录 + /logs 就好
#    - dataset_dir: /select2reason/  # 保持不变

# 5) 训练（8卡示例，按需调整 CUDA_VISIBLE_DEVICES / --nproc_per_node）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.run \
  --nproc_per_node=8 \
  --master_addr=127.0.0.1 \
  --master_port=29505 \
  src/llamafactory/launcher.py \
  examples/train_full/select2reason.yaml

# 6) 测评（将 <CKPT_DIR> 替换为训练权重输出目录）
python eval.py --model_path <CKPT_DIR>
```

---

## 详细步骤

### 1. 克隆与进入目录

```bash
cd LLaMA-Factory
```

> 若尚未克隆，请先将仓库拉取到本地/服务器，再进入根目录。

### 2. 创建虚拟环境

```bash
conda create -n math python=3.10 -y
conda activate math
```

### 3. 安装依赖

在项目根目录执行：

```bash
pip install -e .
pip install modelscope
MAX_JOBS=4 pip install flash-attn --no-build-isolation
pip install deepspeed==0.16.9
# 测评阶段：
pip install vllm
```

> **说明**
>
> * `-e .` 以开发者模式安装 LLaMA-Factory。
> * `flash-attn` 安装时通过 `MAX_JOBS=4` 降低并行编译任务数，缓解内存压力。
> * `deepspeed==0.16.9` 为已验证版本。
> * 仅在评测推理时需要 `vllm`。

### 4. 下载模型（ModelScope）

将模型下载至服务器指定目录（例如 `<MODEL_CACHE_DIR>`）：

```bash
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-Math-7B-Instruct', cache_dir='<MODEL_CACHE_DIR>')"
```

下载完成后，模型路径通常形如：

```
<MODEL_CACHE_DIR>/Qwen/Qwen2.5-Math-7B-Instruct/
```

### 5. 配置训练 YAML

示例 YAML：`examples/train_full/select2reason.yaml`

需要修改以下字段（**根据你的服务器路径替换**）：

* `model_name_or_path`: `在4中下载的模型目录`
* `output_dir`: 训练权重保存目录（例如：`/mnt/public/<user>/select2reason/train/`）
* `logging_dir`: 训练日志目录（例如：`/mnt/public/<user>/select2reason/train/logs`）
* `dataset_dir`: **保持为** `/select2reason/` （需要迁移这个文件夹的话这里也要修改）

> DeepSpeed 配置：`examples/deepspeed/ds_z2_config.json`（**不需要修改**）。

### 6. 训练

以 8 卡为例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.run \
  --nproc_per_node=8 \
  --master_addr=127.0.0.1 \
  --master_port=29505 \
  src/llamafactory/launcher.py \
  examples/train_full/select2reason.yaml
```

> * 如单机多卡，`--master_addr` 设为本机地址即可。
> * 若 GPU 数量不同，请同步调整 `CUDA_VISIBLE_DEVICES` 和 `--nproc_per_node`。

### 7. 测评

训练完成后，使用保存的权重目录进行评测：

```bash
python eval.py --model_path <CKPT_DIR>
```

> 将 `<CKPT_DIR>` 替换为步骤 6 中 `output_dir` 生成的最终模型路径。

---

## 常见问题

1. **flash-attn 编译失败/内存崩溃**

   * 使用：`MAX_JOBS=4 pip install flash-attn --no-build-isolation` , 设置max jobs限制

2. **DeepSpeed 版本冲突**

   * 使用已验证版本：`pip install deepspeed==0.16.9`

3. **多 GPU 训练卡住**

   * 检查 `--master_addr`/`--master_port` 是否被占用；
   * `--nproc_per_node` 与可见 GPU 数量一致；
   * 网络/防火墙设置允许本机通信。
---

## 参考目录结构

```
LLaMA-Factory/
├─ examples/
│  ├─ train_full/
│  │  └─ select2reason.yaml    # 训练配置（需修改路径）
│  └─ deepspeed/
│     └─ ds_z2_config.json     # ZeRO-2 配置（无需改）
├─ src/llamafactory/launcher.py
├─ eval.py                      # 评测脚本：python eval.py --model_path <CKPT_DIR>
└─ eval_data
└─ data_reason                  #训练数据文件夹
```

---

