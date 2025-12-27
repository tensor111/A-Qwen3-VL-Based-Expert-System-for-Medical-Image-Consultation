import os
import io
import torch
from PIL import Image
from datasets import Dataset
from transformers import (
    CLIPProcessor,
    CLIPModel,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)

class CLIPTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 忽略多余参数（如 num_items_in_batch）
        outputs = model(**inputs, return_loss=True)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


# ----------------配置区域----------------
MODEL_NAME = "/mnt/workspace/clip-vit-base-16patch"  # 原始 CLIP 模型
DATA_PATH = "/mnt/workspace/data/cleantrain-00000-of-00010.parquet" # 你的 Parquet 文件路径
OUTPUT_DIR = "/mnt/workspace/output"        # 保存路径
BATCH_SIZE = 64                                # CLIP 对比学习需要较大的 Batch Size 效果才好
NUM_EPOCHS = 5                                 # 只是对齐特征，不需要训练太久
LEARNING_RATE = 5e-5
MAX_LEN = 77                                   # CLIP 文本最大长度

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------1. 数据准备----------------
def load_and_process_data(data_path, processor):
    print(f"正在加载数据: {data_path}")
    dataset = Dataset.from_parquet(data_path)
    
    # 定义预处理函数
    def transform(examples):
        images = []
        texts = []
        
        for img_data, text in zip(examples['image'], examples['text']):
            # 1. 图片处理
            if isinstance(img_data, dict) and 'bytes' in img_data:
                image = Image.open(io.BytesIO(img_data['bytes'])).convert("RGB")
            elif isinstance(img_data, bytes):
                image = Image.open(io.BytesIO(img_data)).convert("RGB")
            else:
                image = img_data.convert("RGB")
            images.append(image)
            
            # 2. 文本处理 (Prompt Engineering)
            # 这一步很重要：将单纯的标签变成描述性句子，帮助模型理解语境
            text = f"A brain CT scan showing {text}" 
            texts.append(text)
        
        # 使用 Processor 同时处理图文
        # padding="max_length" 确保 batch 内维度一致
        batch_out = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN
        )
        
        # Processor 返回的是 pixel_values 和 input_ids 等
        # 这里的 key 需要直接对应模型的 forward 参数
        return {
            "input_ids": batch_out["input_ids"],
            "attention_mask": batch_out["attention_mask"],
            "pixel_values": batch_out["pixel_values"]
        }

    # 应用预处理 (batched=True 加速)
    print("正在预处理数据...")
    processed_dataset = dataset.map(
        transform, 
        batched=True, 
        batch_size=32,
        remove_columns=dataset.column_names, # 移除原始列，只保留 tensor
    )
    
    return processed_dataset

# ----------------2. 模型加载----------------
print("正在加载 CLIP 模型...")
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME, use_fast=True)

# ----------------3. 准备数据集----------------
train_dataset = load_and_process_data(DATA_PATH, processor)

# 划分验证集 (可选)
split_dataset = train_dataset.train_test_split(test_size=0.1)
train_ds = split_dataset["train"]
eval_ds = split_dataset["test"]

print(f"训练集大小: {len(train_ds)}, 验证集大小: {len(eval_ds)}")

# ----------------4. 训练设置----------------
# CLIP 的 forward 函数会自动计算 contrastive loss (如果提供了 input_ids 和 pixel_values)
# 只要 return_loss=True (Trainer 默认会处理)
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.1,
    save_strategy="epoch",
    eval_strategy="epoch",  # <--- 将 evaluation_strategy 改为 eval_strategy
    logging_steps=10,
    fp16=True, 
    dataloader_num_workers=4,
    remove_unused_columns=False, 
    report_to="none"
)

# 使用默认 Collator 即可，因为我们已经处理成了 tensor
trainer = CLIPTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=DefaultDataCollator(),
)

# ----------------5. 开始微调----------------
print("开始 CLIP 微调...")
trainer.train()

# ----------------6. 保存模型----------------
print(f"保存微调后的 CLIP 模型到: {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print("完成！")