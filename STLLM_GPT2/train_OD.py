
import torch
import numpy as np
import pandas as pd
import argparse
import time
import util
import os
from util import *
import random
from model_ST_LLM import ST_LLM, ST_LLM_DS
from ranger21 import Ranger
import torch.optim as optim
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:21'

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:3", help="")
parser.add_argument("--data", nargs="+", type=str, default=["bike_drop"], help="data path")
parser.add_argument("--input_dim", type=int, default=3, help="input_dim")
parser.add_argument("--channels", type=int, default=64, help="number of features")
parser.add_argument("--num_nodes", type=int, default=266, help="number of nodes")
parser.add_argument("--input_len", type=int, default=1, help="input_len")
parser.add_argument("--output_len", type=int, default=1, help="out_len")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--lrate", type=float, default=0.05, help="learning rate")
parser.add_argument("--U", type=int, default=2, help="unfrozen attention layer")
parser.add_argument("--epochs", type=int, default=500, help="500")
parser.add_argument("--print_every", type=int, default=50, help="")
parser.add_argument("--wdecay", type=float, default=0.0001, help="weight decay rate")
parser.add_argument("--es_patience", type=int, default=20, help="quit if no improvement after this many iterations")
parser.add_argument("--sample_days", type=int, default=None, help="Number of days to sample for training (None for full sample)")
args = parser.parse_args()

data_str = "_".join(args.data)
if args.sample_days is not None:
    data_str += f"-{args.sample_days}"

args.save = f'./logs/{str(time.strftime("%Y-%m-%d-%H:%M:%S"))}-{data_str}/'

class trainer:
    def __init__(self, scaler, lrate, wdecay, input_dim, channels, num_nodes, input_len, output_len, U, device):
        # self.model = ST_LLM_DS(device, input_dim, channels, num_nodes, input_len, output_len, U)
        self.model = ST_LLM(device, input_dim, channels, num_nodes, input_len, output_len, U)

        # 新增用于存储所有预测OD矩阵的列表
        self.all_predictions = []


        self.model.to(device)
        self.optimizer = Ranger(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.MSE_torch
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(input)



        real = real_val
        predict = output

        # predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, None)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.MAE_torch(predict, real,None).item()
        mape = util.MAPE_torch(predict, real, 0.0).item()
        rmse = util.RMSE_torch(predict, real, None).item()
        wmape = util.WMAPE_torch(predict, real, 0.0).item()
        return loss.item(),rmse,mae,mape

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)

        real = real_val
        predict = output


        loss = self.loss(predict, real, None)
        mae = util.MAE_torch(predict, real, None).item()
        mape = util.MAPE_torch(predict, real, 0.0).item()
        rmse = util.RMSE_torch(predict, real, None).item()
        wmape = util.WMAPE_torch(predict, real, 0.0).item()
        return loss.item(), rmse, mae, mape

    def test(self, input, real_val):
        self.model.eval()
        output = self.model(input)

        # 收集预测结果
        self.all_predictions.append(output.detach().cpu().numpy())  # 转换为numpy并保存到列表


        print("Input",input.shape)
        print("Output",output.shape)
        print("Real",real_val.shape)
        real = real_val
        predict = output
        print(f"测试集的预测shape：{predict.shape}")

        vmin = min(output[-2].min(), real_val[-2].min())
        vmax = max(output[-2].max(), real_val[-2].max())

        # true_max = output[-2].max()
        # pred_max = real_val[-2].max()
        # print(f"真实最大值{true_max},预测最大值，{pred_max}")
        #
        # real_val_np = real_val.cpu().numpy()
        # output_np = output.cpu().numpy()
        #
        # plt.figure(figsize=(15, 7))
        #
        # # 绘制真实 OD 热力图
        # plt.subplot(1, 2, 1)
        # sns.heatmap(real_val_np[-2], cmap="Blues", cbar=True, vmin=0, vmax=vmax)
        # plt.title("True OD Matrix", fontsize=14)
        # plt.xlabel("Destination Zones")
        # plt.ylabel("Origin Zones")
        #
        # # 绘制预测 OD 热力图
        # plt.subplot(1, 2, 2)
        # sns.heatmap(output_np[-2], cmap="Blues", cbar=True, vmin=0, vmax=vmax)
        # plt.title("Predicted OD Matrix", fontsize=14)
        # plt.xlabel("Destination Zones")
        # plt.ylabel("Origin Zones")
        #
        # # 调整布局并显示
        # plt.tight_layout()
        # plt.savefig(f"GPT2-predict.png")
        # # plt.show()






        loss = self.loss(predict, real, None)
        mae = util.MAE_torch(predict, real, None).item()
        mape = util.MAPE_torch(predict, real, 0.0).item()
        rmse = util.RMSE_torch(predict, real, None).item()
        wmape = util.WMAPE_torch(predict, real, 0.0).item()
        return loss.item(), rmse, mae, mape

def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def get_sample_data(data, sample_days):
    if sample_days is None:
        return data  # 如果没有选择 sample_days，返回完整数据

    total_time_steps = data["x_train"].shape[0]
    samples_per_day = 48  # 一天的时间步数

    if sample_days not in [1, 3, 7, 30]:
        raise ValueError("Unsupported sample days value")

    # 计算选择的时间步
    end = min(total_time_steps, samples_per_day * sample_days)

    # 截取数据集，只保留前 `end` 个时间步的数据
    data["x_train"] = data["x_train"][:end, ...]
    data["y_train"] = data["y_train"][:end, ...]

    # 重新生成训练数据加载器
    data["train_loader"] = DataLoader(data["x_train"], data["y_train"], batch_size=args.batch_size)

    return data


def main():
    log_filename = f"./logs/DS_{args.data}.log"

    seed_it(42)
    folders = args.data
    data = "_".join(folders)

    if "bike" in data:
        args.num_nodes = 250
    elif "taxi" in data:
        args.num_nodes = 266
    elif "Speed" in data:
        args.num_nodes = 110
    elif "MCM" in data:
        args.num_nodes = 110


    device = torch.device(args.device)

    # data_paths = [f"{folder}/" for folder in folders]
    # dataloader = util.load_dataset(data_paths, args.batch_size, args.batch_size, args.batch_size)
    # dataloader = get_sample_data(dataloader, args.sample_days)  # 获取只筛选后的训练集数据

    # 加载OD和Speed数据
    dataloader = { }
    train_loader, val_loader, test_loader, scaler = util.load_data(data)
    dataloader["train_loader"] = train_loader
    dataloader["val_loader"] = val_loader
    dataloader["test_loader"] = test_loader

    loss = float("inf")
    test_log = float("inf")
    epochs_since_best_mae = 0

    his_loss = []
    val_time = []
    train_time = []
    result = []
    test_result = []
    print(args)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    engine = trainer(
        scaler,
        args.lrate,
        args.wdecay,
        args.input_dim,
        args.channels,
        args.num_nodes,
        args.input_len,
        args.output_len,
        args.U,
        device
    )

    print("start training...", flush=True)

    # 注释掉训练代码
    # count = 0
    #
    # for i in range(1, args.epochs + 1):
    #     train_loss = []
    #     train_mape = []
    #     train_rmse = []
    #     train_wmape = []
    #     train_mae = []
    #
    #     t1 = time.time()
    #     for iter, (x, y) in enumerate(dataloader["train_loader"]):
    #         trainx = torch.Tensor(x).to(device)
    #
    #         trainy = torch.Tensor(y).to(device)
    #
    #         metrics = engine.train(trainx, trainy)
    #         train_loss.append(metrics[0])
    #         train_rmse.append(metrics[1])
    #         train_mae.append(metrics[2])
    #         train_mape.append(metrics[3])
    #
    #     t2 = time.time()
    #     # log = "Epoch: {:03d}, Training Time: {:.4f} secs"
    #     # print(log.format(i, (t2 - t1)))
    #     train_time.append(t2 - t1)
    #
    #     valid_loss = []
    #     valid_mape = []
    #     valid_wmape = []
    #     valid_rmse = []
    #     valid_mae = []
    #
    #     s1 = time.time()
    #     for iter, (x, y) in enumerate(dataloader["val_loader"]):
    #         testx = torch.Tensor(x).to(device)
    #
    #         testy = torch.Tensor(y).to(device)
    #
    #         metrics = engine.eval(testx, testy)
    #         valid_loss.append(metrics[0])
    #         valid_rmse.append(metrics[1])
    #         valid_mae.append(metrics[2])
    #         valid_mape.append(metrics[3])
    #
    #     s2 = time.time()
    #     # log = "Epoch: {:03d}, Inference Time: {:.4f} secs"
    #     # print(log.format(i, (s2 - s1)))
    #     val_time.append(s2 - s1)
    #
    #     mtrain_loss = np.mean(train_loss)
    #     mtrain_mape = np.mean(train_mape)
    #     mtrain_wmape = np.mean(train_wmape)
    #     mtrain_rmse = np.mean(train_rmse)
    #     mtrain_mae = np.mean(train_mae)
    #
    #     mvalid_loss = np.mean(valid_loss)
    #     mvalid_mape = np.mean(valid_mape)
    #     mvalid_wmape = np.mean(valid_wmape)
    #     mvalid_rmse = np.mean(valid_rmse)
    #     mvalid_mae = np.mean(valid_mae)
    #
    #     his_loss.append(mvalid_loss)
    #     print("-----------------------")
    #
    #     train_m = dict(
    #         train_loss=mtrain_loss,
    #         train_rmse=mtrain_rmse,
    #         train_mape=mtrain_mape,
    #         train_wmape=mtrain_wmape,
    #         train_mae=mtrain_mae,
    #         valid_loss=mvalid_loss,
    #         valid_rmse=mvalid_rmse,
    #         valid_mape=mvalid_mape,
    #         valid_wmape=mvalid_wmape,
    #         valid_mae=mvalid_mae
    #     )
    #     train_m = pd.Series(train_m)
    #     result.append(train_m)
    #
    #     log = "Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAE: {:.4f}, Train MAPE: {:.4f}, "
    #     print(log.format(i, mtrain_loss, mtrain_rmse, mtrain_mae, mtrain_mape), flush=True)
    #
    #     log = "Epoch: {:03d}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}"
    #     print(log.format(i, mvalid_loss, mvalid_rmse, mvalid_mae, mvalid_mape), flush=True)
    #
    #     if mvalid_loss < loss:
    #         print("### Update tasks appear ###")
    #         loss = mvalid_loss
    #         count += 1
    #         if count == 1 or loss < 100:
    #             torch.save(engine.model.state_dict(), args.save + "best_model.pth")
    #         bestid = i
    #         epochs_since_best_mae = 0
    #         print("Updating! Valid Loss:{:.4f}".format(mvalid_loss), end=", ")
    #         print("epoch: ", i)
    #     else:
    #         epochs_since_best_mae += 1
    #
    #     if epochs_since_best_mae >= args.es_patience:
    #         print("Early stopping triggered.")
    #         break
    #
    # print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    # print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
    #
    # print("Training ends")
    # print("The epoch of the best result：", bestid)
    # print("The valid loss of the best model", str(round(his_loss[bestid - 1], 4)))


    # Final Test
    print("开始测试")
    # engine.model.load_state_dict(torch.load(args.save + "best_model.pth"))


    # ds直接测试
    # engine.model.load_state_dict(torch.load("./logs/DS-Best-MCM/best_model.pth"))

    # gpt2直接测试
    engine.model.load_state_dict(torch.load("./logs/GPT2-Best-MCM/best_model.pth"))


    test_loss = []
    test_mape = []
    test_wmape = []
    test_rmse = []
    test_mae = []

    s1 = time.time()
    with torch.no_grad():
        for iter, (x, y) in enumerate(dataloader["test_loader"]):
            testx = torch.Tensor(x).to(device)

            testy = torch.Tensor(y).to(device)

            metrics = engine.test(testx, testy)
            test_loss.append(metrics[0])
            test_rmse.append(metrics[1])
            test_mae.append(metrics[2])
            test_mape.append(metrics[3])

    s2 = time.time()

    mtest_loss = np.mean(test_loss)

    mtest_mape = np.mean(test_mape)
    mtest_wmape = np.mean(test_wmape)
    mtest_rmse = np.mean(test_rmse)
    mtest_mae = np.mean(test_mae)
    log = "Final Test: RMSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}"
    print(log.format(mtest_rmse, mtest_mae, mtest_mape), flush=True)
    with open(log_filename, 'a') as log_file:
        log_file.write(f"Lr = {args.lrate}, Epochs = {args.epochs}, Test Loss: {mtest_loss:.4f} RMSE: {mtest_rmse:.4f} "
                       f"MAE: {mtest_mae:.4f} MAPE: {mtest_mape:.4f} -Time: {(s2-s1):.4f}s\n")


    # 收集所有测试预测结果
    all_od_predictions = np.concatenate(engine.all_predictions, axis=0)  # 按批次拼接


    # 保存为npy文件
    np.save("./Pred_GPT2-MCM.npy", all_od_predictions)
    # print(f"已保存所有预测OD矩阵Pred_DeepSeek-MOE-MCM.npy,形状为{all_od_predictions.shape}")

if __name__ == "__main__":
    torch.cuda.empty_cache()  # 清除 GPU 缓存
    t1 = time.time()  # 记录开始时间

    main()  # 执行主函数
    t2 = time.time()  # 记录结束时间
    print("Total time spent: {:.4f}".format(t2 - t1))  # 打印总耗时
