import torch
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error



# 假设已经有Speed和OD数据集
Speed = np.load("../data/Speed_完整批处理25.3.14.npy")
OD = np.load("../data/OD_完整批处理25.3.14.npy")

# 自定义数据集类
class TrafficODDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        label_sample = self.labels[idx]
        return {
            'input': torch.tensor(input_sample, dtype=torch.long),
            'label': torch.tensor(label_sample, dtype=torch.float32)
        }

torch.cuda.empty_cache()

# 数据集划分
train_size = int(0.6 * len(Speed))
val_size = int(0.2 * len(Speed))
test_size = len(Speed) - train_size - val_size

train_inputs, temp_inputs, train_labels, temp_labels = train_test_split(Speed, OD, train_size=train_size, random_state=42)
val_inputs, test_inputs, val_labels, test_labels = train_test_split(temp_inputs, temp_labels, test_size=0.5, random_state=42)

# 创建数据集和数据加载器
train_dataset = TrafficODDataset(train_inputs, train_labels)
val_dataset = TrafficODDataset(val_inputs, val_labels)
test_dataset = TrafficODDataset(test_inputs, test_labels)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



torch.cuda.empty_cache()

model_name = "deepseek-ai/deepseek-moe-16b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

custom_config = {
    "num_hidden_layers": 2,
    # "vocab_size": 110,
    # 其他自定义参数...
}


# model = AutoModelForCausalLM.from_pretrained(cache_dir, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda:1",torch_dtype=torch.bfloat16,
                                             trust_remote_code=True,local_files_only=False,**custom_config)

model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

model.generation_config.num_hidden_layers = 2
model.generation_config.vocab_size = 110
model.config.num_hidden_layers = 2
model.config.vocab_size = 110

# 定义损失函数和优化器
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# 训练模型
num_epochs = 40
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("device:",device)
# model.to(device)
print(model.device)

torch.cuda.empty_cache()

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    torch.cuda.empty_cache()
    for batch in train_loader:
        inputs = batch['input'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        # last_hidden_s = outputs.last_hidden_state

        # print("Input shape",inputs.shape)
        # print(f"Last hidden state shape: {last_hidden_s.shape}")
        # print("input",inputs)
        # print("output",outputs)

        print("outputs",outputs.logits.shape)
        print(outputs.logits[0:,0:,:10])
        # print("Label shape",labels.shape)

        loss = criterion(outputs.logits[:,:,:110].reshape(-1), labels.reshape(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['input'].to(device)
            labels = batch['label'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs.logits[:,:,:110].reshape(-1), labels.reshape(-1))
            val_loss += loss.item()

    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}')



torch.cuda.empty_cache()


# 测试模型
model.eval()
test_predictions = []
test_true_labels = []
with torch.no_grad():
    for batch in test_loader:
        inputs = batch['input'].to(device)
        labels = batch['label'].to(device)
        outputs = model(inputs)
        test_predictions.extend(outputs.logits[:,:,:110].reshape(-1).cpu().numpy())
        test_true_labels.extend(labels.reshape(-1).cpu().numpy())

# 计算评估指标
rmse = np.sqrt(mean_squared_error(test_true_labels, test_predictions))
mae = mean_absolute_error(test_true_labels, test_predictions)
mape = np.mean(np.abs((np.array(test_true_labels) - np.array(test_predictions)) / np.array(test_true_labels))) * 100

print(f'RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}%')

