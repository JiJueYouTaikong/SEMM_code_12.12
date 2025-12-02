import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn


class InputEmdLayer(nn.Module):
    def __init__(self, num_filters=64, embedding_dim=768, seq=1):
        super(InputEmdLayer, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=1)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm([num_filters, seq])
        self.linear = nn.Linear(in_features=seq, out_features=embedding_dim)

    def forward(self, speed):
        # 输入 speed 形状: [B, N, T]
        B, N, T = speed.shape

        # 调整形状以适应一维卷积输入 [B * N, 1, T]
        speed = speed.reshape(B * N, 1, T)

        # 一维卷积
        SE = self.conv1d(speed)  # [B * N, F, T]

        SE = self.relu(SE)  # [B * N, F, T]

        # 层归一化
        SE = self.layer_norm(SE)  # [B * N, F, T]

        # 线性层将T映射到D
        SE = self.linear(SE)  # [B * N, F, D]

        SE = SE.reshape(B, N, -1, SE.shape[-1])  # [B, N, F, D]

        # 取最后一个特征通道的状态
        input_emb = SE[:, :, -1, :]

        # 最终input_emb：[B, N, D]
        return input_emb


class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()

        self.time = time
        # temporal embeddings
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)

        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        day_emb = x[..., 1]
        time_day = self.time_day[
            (day_emb[:, -1, :] * self.time).type(torch.LongTensor)
        ]
        time_day = time_day.transpose(1, 2).unsqueeze(-1)

        week_emb = x[..., 2]
        time_week = self.time_week[
            (week_emb[:, -1, :]).type(torch.LongTensor)
        ]
        time_week = time_week.transpose(1, 2).unsqueeze(-1)

        # temporal embeddings
        tem_emb = time_day + time_week
        return tem_emb



class PFA(nn.Module):
    def __init__(self, device="cuda:0", gpt_layers=6, U=1):
        super(PFA, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained(
            "gpt2", output_attentions=True, output_hidden_states=True
        )
        self.gpt2.h = self.gpt2.h[:gpt_layers]
        self.U = U

        for layer_index, layer in enumerate(self.gpt2.h):
            for name, param in layer.named_parameters():
                if layer_index < gpt_layers - self.U:
                    if "ln" in name or "wpe" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    if "mlp" in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

    def forward(self, x):
        return self.gpt2(inputs_embeds=x).last_hidden_state


class PFA_DS(nn.Module):
    def __init__(self, device="cuda:3", deepseek_layers=10, U=1):
        super().__init__()
        self.deepseek = AutoModel.from_pretrained(
            "deepseek-ai/deepseek-moe-16b-base",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="eager",
            output_hidden_states=True
        )
        total_layers = len(self.deepseek.layers)
        if deepseek_layers > total_layers:
            raise ValueError(f"deepseek_layers ({deepseek_layers}) exceeds the total number of layers ({total_layers}) in the model.")
        self.deepseek.layers = self.deepseek.layers[:deepseek_layers]
        self.U = U
        # print("总层数",total_layers,"冻结层数",deepseek_layers,"不冻结层数",self.U)
        for layer_index, layer in enumerate(self.deepseek.layers):
            for name, param in layer.named_parameters():
                if layer_index < deepseek_layers - self.U:
                    if "ln" in name or "wpe" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    if "mlp" in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

    def forward(self, x):
        x = x.to(torch.bfloat16)
        outputs = self.deepseek(inputs_embeds=x)
        output = outputs.last_hidden_state
        torch.cuda.empty_cache()  # Release cache after forward pass
        return output



class ST_LLM(nn.Module):
    def __init__(
        self,
        device="cuda:3",
        input_dim=3,
        channels=64,
        num_nodes=170,
        input_len=12,
        output_len=12,
        U=1,
    ):
        super().__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.node_dim = channels
        self.input_len = input_len
        self.input_dim = input_dim
        self.output_len = output_len
        self.U = U
        # #print(f"U is of type {type(self.U)} with value {self.U}")

        # Determine time based on num_nodes
        if num_nodes == 170 or num_nodes == 307:
            time = 288
        elif num_nodes == 250 or num_nodes == 266:
            time = 48
        elif num_nodes == 110:
            time = 48
        else:
            raise ValueError(f"Invalid num_nodes value: {num_nodes}")

        # time = 48
        gpt_channel = 256
        to_gpt_channel = 768

        self.Temb = TemporalEmbedding(time, gpt_channel)

        # 输入embedding层
        self.Input_emb = InputEmdLayer(num_filters=64, embedding_dim=768)

        self.node_emb = nn.Parameter(torch.empty(self.num_nodes, gpt_channel))
        nn.init.xavier_uniform_(self.node_emb)

        self.start_conv = nn.Conv2d(
            self.input_dim * self.input_len, gpt_channel, kernel_size=(1, 1)
        )

        # embedding layer
        self.gpt = PFA(device=self.device, gpt_layers=6, U=self.U)


        self.feature_fusion = nn.Conv2d(
            gpt_channel * 3, to_gpt_channel, kernel_size=(1, 1)
        )

        # regression
        self.regression_layer = nn.Conv2d(
            gpt_channel * 3, self.output_len, kernel_size=(1, 1)
        )

        self.od_linear = nn.Linear(to_gpt_channel, num_nodes)

    # return the total parameters of model
    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, history_data):
        # [B,N] --> [B,N,T=1]
        history_data = history_data.unsqueeze(-1)


        # input_data = history_data  # [B,C,N,T]
        # batch_size, _, num_nodes, _ = input_data.shape
        # history_data = history_data.permute(0, 2, 3, 1)  # [B,N,T,C]
        # history_data = history_data[:,:,:,0]  # [B,N,T]

        # print("emd前的维度", history_data.shape)
        input_emb = self.Input_emb(history_data)  # [B,N,T] --> [B,N,D]
        #print("emd后的维度", input_emb.shape)


        #print("送入gpt前的输入的最终维度", input_emb.shape)
        data_st = self.gpt(input_emb)
        #print("gpt输出的初始维度：", data_st.shape)

        prediction = self.od_linear(data_st)
        # print("最终输出的预测维度", prediction.shape)
        return prediction



class ST_LLM_DS(nn.Module):
    def __init__(
        self,
        device="cuda:3",
        input_dim=3,
        channels=64,
        num_nodes=170,
        input_len=12,
        output_len=12,
        U=1,
    ):
        super().__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.node_dim = channels
        self.input_len = input_len
        self.input_dim = input_dim
        self.output_len = output_len
        self.U = U
        # #print(f"U is of type {type(self.U)} with value {self.U}")

        # Determine time based on num_nodes
        if num_nodes == 170 or num_nodes == 307:
            time = 288
        elif num_nodes == 250 or num_nodes == 266:
            time = 48
        elif num_nodes == 110:
            time = 48
        else:
            raise ValueError(f"Invalid num_nodes value: {num_nodes}")


        to_gpt_channel = 768
        deepseek_channel = 1024
        to_deepseek_channel = 2048


        # 输入embedding层
        self.Input_emb = InputEmdLayer(num_filters=64, embedding_dim=to_deepseek_channel)


        self.deepseek = PFA_DS(device=self.device, deepseek_layers=3, U=self.U)



        self.od_linear = nn.Linear(to_deepseek_channel, num_nodes)

    # return the total parameters of model
    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, history_data):
        # [B,N] --> [B,N,T=1]
        history_data = history_data.unsqueeze(-1)


        # input_data = history_data  # [B,C,N,T]
        # batch_size, _, num_nodes, _ = input_data.shape
        # history_data = history_data.permute(0, 2, 3, 1)  # [B,N,T,C]
        # history_data = history_data[:,:,:,0]  # [B,N,T]

        # print("emd前的维度", history_data.shape)
        input_emb = self.Input_emb(history_data)  # [B,N,T] --> [B,N,D]
        # print("emd后的维度", input_emb.shape)


        # print("送入ds前的输入的最终维度", input_emb.shape)
        data_st = self.deepseek(input_emb)
        data_st = data_st.to(torch.float32)
        # print("ds输出的初始维度：", data_st.shape)

        prediction = self.od_linear(data_st)
        # print("最终输出的预测维度", prediction.shape)
        return prediction






        # *** 原始ST-LLM ***
        # input_data = history_data  # [B,C,N,T]
        # batch_size, _, num_nodes, _ = input_data.shape
        # history_data = history_data.permute(0, 3, 2, 1) # [B,T,N,C]
        #
        # #print("进入emd模块的维度",history_data.shape)
        # tem_emb = self.Temb(history_data)
        # #print("emb后的维度",tem_emb.shape)
        # node_emb = []
        # node_emb.append(
        #     self.node_emb.unsqueeze(0)
        #     .expand(batch_size, -1, -1)
        #     .transpose(1, 2)
        #     .unsqueeze(-1)
        # )
        #
        # input_data = input_data.transpose(1, 2).contiguous()
        # input_data = (
        #     input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        # )
        # input_data = self.start_conv(input_data)
        #
        # data_st = torch.cat(
        #     [input_data] + [tem_emb] + node_emb, dim=1
        # )
        # #print("emb后cat的维度",data_st.shape)
        #
        # data_st = self.feature_fusion(data_st)
        # # data_st = F.leaky_relu(data_st)
        # #print("fusion后的维度",data_st.shape)
        #
        # data_st = data_st.permute(0, 2, 1, 3).squeeze(-1)
        # #print("送入gpt前的输入的最终维度",data_st.shape)
        # data_st = self.gpt(data_st)
        # #print("gpt输出的初始维度：",data_st.shape)
        # data_st = data_st.permute(0, 2, 1).unsqueeze(-1)
        #
        # prediction = self.regression_layer(data_st)
        # #print("最终输出的预测维度",prediction.shape)
        # return prediction
