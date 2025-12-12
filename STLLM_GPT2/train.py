
import torch
import numpy as np
import pandas as pd
import argparse
import time
import util
import os
from util import *
import random
from model_ST_LLM import ST_LLM
from ranger21 import Ranger

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:21'

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:3", help="")
parser.add_argument("--data", nargs="+", type=str, default=["bike_drop"], help="data path")
parser.add_argument("--input_dim", type=int, default=3, help="input_dim")
parser.add_argument("--channels", type=int, default=64, help="number of features")
parser.add_argument("--num_nodes", type=int, default=266, help="number of nodes")
parser.add_argument("--input_len", type=int, default=12, help="input_len")
parser.add_argument("--output_len", type=int, default=12, help="out_len")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--lrate", type=float, default=1e-3, help="learning rate")
parser.add_argument("--U", type=int, default=2, help="unfrozen attention layer")
parser.add_argument("--epochs", type=int, default=300, help="500")
parser.add_argument("--print_every", type=int, default=50, help="")
parser.add_argument("--wdecay", type=float, default=0.0001, help="weight decay rate")
parser.add_argument("--es_patience", type=int, default=100, help="quit if no improvement after this many iterations")
parser.add_argument("--sample_days", type=int, default=None, help="Number of days to sample for training (None for full sample)")
args = parser.parse_args()

data_str = "_".join(args.data)
if args.sample_days is not None:
    data_str += f"-{args.sample_days}"

args.save = f'./logs/{str(time.strftime("%Y-%m-%d-%H:%M:%S"))}-{data_str}/'

class trainer:
    def __init__(self, scaler, lrate, wdecay, input_dim, channels, num_nodes, input_len, output_len, U, device):
        self.model = ST_LLM(device, input_dim, channels, num_nodes, input_len, output_len, U)
        self.model.to(device)
        self.optimizer = Ranger(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.MAE_torch
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val):
        print("每次训练中传入的输入和真值",input.shape,real_val.shape)

        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input)
        print("训练中模型输出的预测值shape",output.shape)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        print("每次训练计算loss时的预测值和真值",predict.shape,real.shape)
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.MAPE_torch(predict, real, 0.0).item()
        rmse = util.RMSE_torch(predict, real, 0.0).item()
        wmape = util.WMAPE_torch(predict, real, 0.0).item()
        return loss.item(), mape, rmse, wmape

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.MAPE_torch(predict, real, 0.0).item()
        rmse = util.RMSE_torch(predict, real, 0.0).item()
        wmape = util.WMAPE_torch(predict, real, 0.0).item()
        return loss.item(), mape, rmse, wmape

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
    seed_it(6666)
    folders = args.data
    data = "_".join(folders)

    if "bike" in data:
        args.num_nodes = 250
    elif "taxi" in data:
        args.num_nodes = 266

    device = torch.device(args.device)
    data_paths = [f"{folder}/" for folder in folders]
    dataloader = util.load_dataset(data_paths, args.batch_size, args.batch_size, args.batch_size)
    dataloader = get_sample_data(dataloader, args.sample_days)  # 获取只筛选后的训练集数据
    scaler = dataloader["scaler"]

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
    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        train_wmape = []

        t1 = time.time()
        for iter, (x, y) in enumerate(dataloader["train_loader"].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:, 0, :, :])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            train_wmape.append(metrics[3])

        t2 = time.time()
        log = "Epoch: {:03d}, Training Time: {:.4f} secs"
        print(log.format(i, (t2 - t1)))
        train_time.append(t2 - t1)

        valid_loss = []
        valid_mape = []
        valid_wmape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader["val_loader"].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:, 0, :, :])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
            valid_wmape.append(metrics[3])

        s2 = time.time()
        log = "Epoch: {:03d}, Inference Time: {:.4f} secs"
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_wmape = np.mean(train_wmape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_wmape = np.mean(valid_wmape)
        mvalid_rmse = np.mean(valid_rmse)

        his_loss.append(mvalid_loss)
        print("-----------------------")

        train_m = dict(
            train_loss=mtrain_loss,
            train_rmse=mtrain_rmse,
            train_mape=mtrain_mape,
            train_wmape=mtrain_wmape,
            valid_loss=mvalid_loss,
            valid_rmse=mvalid_rmse,
            valid_mape=mvalid_mape,
            valid_wmape=mvalid_wmape,
        )
        train_m = pd.Series(train_m)
        result.append(train_m)

        log = "Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Train WMAPE: {:.4f}, "
        print(log.format(i, mtrain_loss, mtrain_rmse, mtrain_mape, mtrain_wmape), flush=True)

        log = "Epoch: {:03d}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Valid WMAPE: {:.4f}"
        print(log.format(i, mvalid_loss, mvalid_rmse, mvalid_mape, mvalid_wmape), flush=True)

        if mvalid_loss < loss:
            print("### Update tasks appear ###")
            loss = mvalid_loss
            torch.save(engine.model.state_dict(), args.save + "best_model.pth")
            bestid = i
            epochs_since_best_mae = 0
            print("Updating! Valid Loss:{:.4f}".format(mvalid_loss), end=", ")
            print("epoch: ", i)

        # 测试逻辑
        outputs = []
        realy = torch.Tensor(dataloader["y_test"]).to(device)
        realy = realy.transpose(1, 3)[:, 0, :, :]

        for iter, (x, y) in enumerate(dataloader["test_loader"].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            with torch.no_grad():
                preds = engine.model(testx).transpose(1, 3)
            outputs.append(preds.squeeze())

        yhat = torch.cat(outputs, dim=0)
        yhat = yhat[: realy.size(0), ...]

        amae = []
        amape = []
        armse = []
        awmape = []
        future_amae = []
        future_amape = []
        future_armse = []
        future_awmape = []

        for j in range(args.output_len):
            pred = scaler.inverse_transform(yhat[:, :, j])
            real = realy[:, :, j]
            metrics = util.metric(pred, real)

            if j < 2:
                future_amae.append(metrics[0])
                future_amape.append(metrics[1])
                future_armse.append(metrics[2])
                future_awmape.append(metrics[3])

            amae.append(metrics[0])
            amape.append(metrics[1])
            armse.append(metrics[2])
            awmape.append(metrics[3])

        log = "On average over 12 horizons, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}"
        print(log.format(np.mean(amae), np.mean(armse), np.mean(amape), np.mean(awmape)))

        if len(future_amae) > 1:
            future_log = "Future Step 2 MAE: {:.4f}, Future Step 2 MAPE: {:.4f}, Future Step 2 RMSE: {:.4f}, Future Step 2 WMAPE: {:.4f}"
            print(future_log.format(future_amae[1], future_amape[1], future_armse[1], future_awmape[1]))

        # Store metrics for the current epoch in the test result list
        epoch_test_metrics = {
            'epoch': i,
            'test_mae': np.mean(amae),
            'test_rmse': np.mean(armse),
            'test_mape': np.mean(amape),
            'test_wmape': np.mean(awmape),
            'future_2_mae': future_amae[1] if len(future_amae) > 1 else None,
            'future_2_rmse': future_armse[1] if len(future_armse) > 1 else None,
            'future_2_mape': future_amape[1] if len(future_amape) > 1 else None,
            'future_2_wmape': future_awmape[1] if len(future_awmape) > 1 else None,
        }

        test_result.append(pd.Series(epoch_test_metrics))

        # Save train results to CSV after each epoch
        train_csv = pd.DataFrame(result)
        train_csv.round(8).to_csv(f"{args.save}/train.csv")

        # Save test results at the end of each epoch to CSV
        test_csv = pd.DataFrame(test_result)
        test_csv.round(8).to_csv(f"{args.save}/test.csv")

        if np.mean(amae) < test_log:
            test_log = np.mean(amae)
            loss = mvalid_loss
            torch.save(engine.model.state_dict(), args.save + "best_model.pth")
            epochs_since_best_mae = 0
            print("Test low! Updating! Test Loss:", np.mean(amae), end=", ")
            print("Test low! Updating! Valid Loss:", mvalid_loss, end=", ")
            bestid = i
            print("epoch: ", i)
        else:
            epochs_since_best_mae += 1
            print("No update")

        # Early stop
        if epochs_since_best_mae >= args.es_patience and i >= 200:
            break

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    print("Training ends")
    print("The epoch of the best result：", bestid)
    print("The valid loss of the best model", str(round(his_loss[bestid - 1], 4)))

    # Final Test
    engine.model.load_state_dict(torch.load(args.save + "best_model.pth"))
    outputs = []
    realy = torch.Tensor(dataloader["y_test"]).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader["test_loader"].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx).transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[: realy.size(0), ...]

    amae = []
    amape = []
    armse = []
    awmape = []

    for j in range(args.output_len):
        pred = scaler.inverse_transform(yhat[:, :, j])
        real = realy[:, :, j]
        metrics = util.metric(pred, real)

        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        awmape.append(metrics[3])

    log = "On average over 12 horizons, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}"
    print(log.format(np.mean(amae), np.mean(armse), np.mean(amape), np.mean(awmape)))

    future_amae = []
    future_amape = []
    future_armse = []
    future_awmape = []

    for j in range(2):
        pred = scaler.inverse_transform(yhat[:, :, j])
        real = realy[:, :, j]
        metrics = util.metric(pred, real)

        future_amae.append(metrics[0])
        future_amape.append(metrics[1])
        future_armse.append(metrics[2])
        future_awmape.append(metrics[3])

    if len(future_amae) > 1:
        future_log = "Future Step 2 MAE: {:.4f}, Future Step 2 MAPE: {:.4f}, Future Step 2 RMSE: {:.4f}, Future Step 2 WMAPE: {:.4f}"
        print(future_log.format(future_amae[1], future_amape[1], future_armse[1], future_awmape[1]))

    test_metrics = {
        'test_mae': np.mean(amae),
        'test_rmse': np.mean(armse),
        'test_mape': np.mean(amape),
        'test_wmape': np.mean(awmape),
        'future_2_mae': future_amae[1] if len(future_amae) > 1 else None,
        'future_2_rmse': future_armse[1] if len(future_armse) > 1 else None,
        'future_2_mape': future_amape[1] if len(future_amape) > 1 else None,
        'future_2_wmape': future_awmape[1] if len(future_awmape) > 1 else None,
    }
    test_metrics = pd.Series(test_metrics)

    # Save final test results to CSV
    test_result.append(test_metrics)
    test_csv = pd.DataFrame(test_result)
    test_csv.round(8).to_csv(f"{args.save}/test.csv")

if __name__ == "__main__":
    torch.cuda.empty_cache()  # 清除 GPU 缓存
    t1 = time.time()  # 记录开始时间
    main()  # 执行主函数
    t2 = time.time()  # 记录结束时间
    print("Total time spent: {:.4f}".format(t2 - t1))  # 打印总耗时
