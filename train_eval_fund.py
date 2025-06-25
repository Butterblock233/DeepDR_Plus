"""
模块名称：train_eval_fund.py

模块描述：实现基于眼底图像的糖尿病视网膜病变进展时间预测模型的训练和评估。
该模块使用深度生存分析方法，结合CNN特征提取和Weibull混合分布，通过眼底图像预测患者
发展为更严重DR级别的时间。

主要组件：
- DeepSurModel：深度生存分析模型，结合CNN和Weibull混合分布
- ProgressionData：眼底图像数据集处理类
- TrainerDR：眼底模型训练器，继承自基础Trainer类

使用方法：
直接运行此文件以训练眼底图像模型：python train_eval_fund.py
可通过环境变量配置训练参数，如batch_size、epochs、lr等
"""

import math
from model import ModelProgression
from torch import nn
import torch
import numpy as np
from functools import cached_property
from trainer import Trainer
from torch.utils.data import Dataset
import pandas as pd
import cv2
import albumentations as aug
import albumentations.pytorch as aug_torch


class DeepSurModel(nn.Module):
    """
    深度生存分析模型，用于预测糖尿病视网膜病变的进展时间。
    
    该模型结合了CNN特征提取和Weibull混合分布生存分析，通过眼底图像
    预测患者发展为更严重DR级别的时间。模型使用ResNet50作为骨干网络
    提取图像特征，然后通过混合Weibull分布建模生存时间。
    
    属性：
        K (int): 混合模型的组件数量
        b (Tensor): Weibull分布的尺度参数
        k (Tensor): Weibull分布的形状参数
        cnn (nn.Module): 用于特征提取的CNN网络
    """
    def __init__(self, K=512) -> None:
        """
        初始化深度生存分析模型。
        
        参数：
            K (int, 可选): 混合模型的组件数量，默认为512
        """
        super().__init__()
        self.K = K
        # 为混合模型采样参数
        rnd = np.random.RandomState(12345)  # 固定随机种子以确保可重复性
        # 生成Weibull分布的尺度参数b（正值）
        b = torch.FloatTensor(abs(rnd.normal(0, 10, (1, 1, self.K))+5.0))
        # 生成Weibull分布的形状参数k（正值）
        k = torch.FloatTensor(abs(rnd.normal(0, 10, (1, 1, self.K))+5.0))
        # 将参数注册为缓冲区，这样它们会被保存但不会被视为模型参数
        self.register_buffer('b', b)
        self.register_buffer('k', k)

        # 初始化CNN特征提取器，使用ResNet50作为骨干网络
        self.cnn = ModelProgression(backbone='resnet50', output_size=512)

    def _cdf_at(self, t):
        """
        计算Weibull分布在时间点t的累积分布函数(CDF)值。
        
        参数：
            t (Tensor): 形状为[nBatch, n, 1]的张量，表示时间点
            
        返回：
            Tensor: 形状为[nBatch, n, K]的张量，表示每个组件的CDF值
        """
        # 计算Weibull分布的CDF: F(t) = 1 - exp(-(t/b)^k)
        cdf = 1 - torch.exp(-(1/self.b * (t)) ** self.k)
        return cdf

    def _pdf_at(self, t):
        """
        计算Weibull分布在时间点t的概率密度函数(PDF)值。
        
        参数：
            t (Tensor): 形状为[nBatch, n, 1]的张量，表示时间点
            
        返回：
            Tensor: 形状为[nBatch, n, K]的张量，表示每个组件的PDF值
        """
        # 首先获取CDF值
        cdf = self._cdf_at(t)
        # 计算Weibull分布的PDF: f(t) = (k/b)*(t/b)^(k-1)*exp(-(t/b)^k)
        # 可以表示为: f(t) = (1-F(t)) * (k/b)*(t/b)^(k-1)
        pdf = (1-cdf) * self.k * (1/self.b)*(t/self.b)**(self.k-1)
        return pdf

    def calculate_cdf(self, w, t):
        """
        计算给定数据的累积概率分布函数(CDF)。
        
        使用混合Weibull分布模型计算CDF值，即在时间t之前发生事件的概率。
        
        参数：
            w (Tensor): 形状为[nBatch, K]的张量，表示混合模型的权重
            t (Tensor): 形状为[nBatch, n]的张量，表示目标时间点
            
        返回：
            Tensor: 形状为[nBatch, n]的张量，表示CDF值
        """
        t = t.unsqueeze(dim=2)  # 扩展维度以便进行广播，变为[nBatch, n, 1]
        w = nn.functional.softmax(w, dim=1)  # 使用softmax将权重归一化
        w = w.unsqueeze(dim=1)  # 扩展维度以便进行广播，变为[nBatch, 1, K]
        
        # 计算每个组件的CDF值
        cdf = self._cdf_at(t)  # [nBatch, n, K]
        # 加权求和得到混合分布的CDF
        cdf = cdf * w  # [nBatch, n, K]
        cdf = cdf.sum(dim=2)  # [nBatch, n]
        return cdf

    def calculate_pdf(self, w, t):
        """
        计算给定数据的概率密度函数(PDF)。
        
        使用混合Weibull分布模型计算PDF值，表示事件在时间t发生的概率密度。
        
        参数：
            w (Tensor): 形状为[nBatch, K]的张量，表示混合模型的权重
            t (Tensor): 形状为[nBatch, n]的张量，表示目标时间点
            
        返回：
            Tensor: 形状为[nBatch, n]的张量，表示PDF值
        """
        t = t.unsqueeze(dim=2)  # 扩展维度以便进行广播，变为[nBatch, n, 1]
        w = nn.functional.softmax(w, dim=1)  # 使用softmax将权重归一化
        w = w.unsqueeze(dim=1)  # 扩展维度以便进行广播，变为[nBatch, 1, K]
        
        # 计算每个组件的PDF值
        pdf = self._pdf_at(t)  # [nBatch, n, K]
        # 加权求和得到混合分布的PDF
        pdf = pdf * w  # [nBatch, n, K]
        pdf = pdf.sum(dim=2)  # [nBatch, n]
        return pdf

    def calculate_survial_time(self, w, t_max=10, resolution=20):
        """
        计算给定数据的生存时间估计值。
        
        通过计算PDF曲线在时间区间[0, t_max]上的最大值点来估计生存时间。
        
        参数：
            w (Tensor): 形状为[nBatch, K]的张量，表示混合模型的权重
            t_max (float, 可选): 最大时间范围，默认为10
            resolution (int, 可选): 时间分辨率，默认为20（每单位时间的采样点数）
            
        返回：
            Tensor: 形状为[nBatch]的张量，表示每个样本的生存时间估计值
        """
        # 生成时间点序列，从1/resolution到t_max，共resolution*t_max-1个点
        t = torch.linspace(
            1/resolution,
            t_max,
            math.ceil(resolution*t_max)-1,
            dtype=torch.float32,
            device=w.device).view(1, -1)
        
        # 计算每个时间点的PDF值
        pdf = self.calculate_pdf(w, t)
        
        # 找到PDF最大值对应的时间点作为生存时间估计
        est = t.view(-1)[torch.argmax(pdf, dim=1)]
        return est

    def forward(self, x, t=None):
        """
        模型的前向传播方法。
        
        参数：
            x (Tensor): 输入图像张量，形状为[nBatch, channels, height, width]
            t (Tensor, 可选): 目标时间点张量，形状为[nBatch, n]。如果为None，则只返回特征
            
        返回：
            如果t为None:
                Tensor: CNN提取的特征，形状为[nBatch, K]
            否则:
                Tuple[Tensor, Tensor]: 
                    - 第一个元素是CNN提取的特征，形状为[nBatch, K]
                    - 第二个元素是在时间t的CDF值，形状为[nBatch, n]
        """
        # 通过CNN提取特征
        x = self.cnn(x)
        if t is None:
            return x
        return x, self.calculate_cdf(x, t)


class ProgressionData(Dataset):
    """
    眼底图像数据集类，继承自PyTorch的Dataset类。
    
    负责加载和处理糖尿病视网膜病变进展预测任务所需的眼底图像数据。
    数据包括眼底图像文件路径、时间信息和事件标记。
    """

    def __init__(self, datasheet, transform):
        """
        初始化数据集。
        
        参数：
            datasheet (str): 包含数据信息的CSV文件路径
            transform (albumentations.Compose): 图像预处理变换组合
        """
        super().__init__()
        self.df = pd.read_csv(datasheet)  # 读取数据表
        self.transform = transform  # 图像预处理变换

    def __len__(self):
        """
        返回数据集中的样本数量。
        
        返回：
            int: 数据集中的样本数量
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        获取指定索引的数据样本。
        
        参数：
            idx (int): 样本索引
            
        返回：
            dict: 包含以下键的字典：
                - image (Tensor): 预处理后的眼底图像张量
                - t1 (float): 上次检查时间
                - t2 (float): 下次检查时间
                - e (int): 事件标记(1表示观察到事件，0表示删失)
                - gt (float): 真实进展时间(仅用于模拟数据)
        """
        # 获取图像文件路径并读取图像
        img_file = self.df.iloc[idx]['image']
        image = cv2.imread(img_file, cv2.IMREAD_COLOR)
        
        # 应用图像预处理变换
        image = self.transform(image=image)['image']
        
        return dict(
            image=image,
            t1=self.df.iloc[idx]['t1'],
            t2=self.df.iloc[idx]['t2'],
            e=self.df.iloc[idx]['e'],
            # 仅用于模拟数据的真实进展时间
            gt=self.df.iloc[idx]['gt'] if 'gt' in self.df.columns else 0,
        )


class TrainerDR(Trainer):
    """
    糖尿病视网膜病变进展预测模型的训练器类，继承自基础Trainer类。
    
    负责眼底图像模型的训练流程，包括数据加载、模型初始化、优化器配置、
    训练循环和评估等。使用深度生存分析方法进行训练。
    """

    @cached_property
    def model(self):
        """
        初始化并返回DeepSurModel模型实例。
        
        如果配置中指定了预训练模型路径，会加载预训练权重。
        
        返回：
            DeepSurModel: 初始化好的模型实例
        """
        model = DeepSurModel().to(self.device)
        if self.cfg.load_pretrain is not None:
            print('loading ', self.cfg.load_pretrain)
            print(model.cnn.backbone.load_state_dict(
                torch.load(self.cfg.load_pretrain, map_location=self.device) # 把预训练的模型加载到特征提取器模块中
            ))
        return model

    @cached_property
    def beta(self):
        """
        获取损失函数中的beta权重参数。
        
        该参数用于调整事件观察样本和删失样本在损失函数中的权重比例。
        
        返回：
            int: beta权重值，默认为1
        """
        return 1

    @cached_property
    def train_dataset(self):
        """
        创建并返回训练数据集。
        
        训练数据集使用多种数据增强技术，包括：
        - 图像大小调整和中心裁剪
        - 随机水平翻转
        - 图像压缩和质量变化
        - 中值模糊
        - 随机亮度和对比度调整
        - 随机Gamma校正
        - 高斯噪声
        - 随机旋转
        
        返回：
            ProgressionData: 训练数据集实例
        """
        transform = aug.Compose([
            aug.SmallestMaxSize(
                max_size=self.cfg.image_size, always_apply=True),
            aug.CenterCrop(self.cfg.image_size, self.cfg.image_size,
                           always_apply=True),
            aug.Flip(p=0.5),
            aug.ImageCompression(quality_lower=10, quality_upper=80, p=0.2),
            aug.MedianBlur(p=0.3),
            aug.RandomBrightnessContrast(p=0.5),
            aug.RandomGamma(p=0.2),
            aug.GaussNoise(p=0.2),
            aug.Rotate(border_mode=cv2.BORDER_CONSTANT,
                       value=0, p=0.7, limit=45),
            aug.ToFloat(always_apply=True),
            aug_torch.ToTensorV2(),
        ])
        return ProgressionData('data_fund/train.csv', transform)

    @cached_property
    def test_dataset(self):
        """
        创建并返回测试数据集。
        
        测试数据集使用基本预处理流程，包括：
        - 图像大小调整和中心裁剪
        - 归一化为浮点类型
        - 转换为PyTorch张量
        
        返回：
            ProgressionData: 测试数据集实例
        """
        transform = aug.Compose([
            aug.SmallestMaxSize(
                max_size=self.cfg.image_size, always_apply=True),
            aug.CenterCrop(self.cfg.image_size, self.cfg.image_size,
                           always_apply=True),
            aug.ToFloat(always_apply=True),
            aug_torch.ToTensorV2(),
        ])
        return ProgressionData('data_fund/test.csv', transform)

    @cached_property
    def optimizer(self):
        """
        创建并返回模型优化器。
        
        使用Adam优化算法，配置如下参数：
        - 学习率(lr): 从配置中获取
        - 权重衰减(weight_decay): 1e-5，用于L2正则化
        
        返回：
            torch.optim.Adam: 初始化好的优化器实例
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=1e-5)
        return optimizer

    def batch(self, epoch, i_batch, data) -> dict:
        """
        处理一个训练批次的数据，包括前向传播和损失计算。
        
        参数：
            epoch (int): 当前训练轮次
            i_batch (int): 批次索引
            data (dict): 包含批次数据的字典
            
        返回：
            dict: 包含以下键的字典：
                - loss (Tensor): 当前批次的平均损失值
                - pdf (Tensor): 计算得到的PDF值
                - cdf (Tensor): 计算得到的CDF值
                - t1 (Tensor): 上次检查时间
                - t2 (Tensor): 下次检查时间
                - survival_time (Tensor): 预测的生存时间
                - gt (Tensor): 真实进展时间(仅用于模拟数据)
        """
        # 准备输入数据并转移到指定设备
        imgs = data['image'].to(self.device)
        t1 = data['t1'].to(self.device)
        t2 = data['t2'].to(self.device)
        e = data['e'].to(self.device)

        # 前向传播，获取模型输出
        w, P = self.model(imgs, torch.stack([t1, t2], dim=1))
        P1 = P[:, 0]  # 上次检查时间的CDF值
        P2 = P[:, 1]  # 下次检查时间的CDF值
        
        # 计算损失函数
        # 第一部分：对于所有样本，最大化1-P1的概率（即事件尚未发生的概率）
        # 第二部分：对于观察到事件的样本(e=1)，最大化P2的概率（即事件已经发生的概率）
        loss = -torch.log(1-P1 + 0.000001) - torch.log(P2 + 0.000001) * self.beta * (e)
        # 添加L1正则化项（权重很小）
        loss += torch.abs(w).mean() * 0.00000001
        
        # 计算额外的评估指标
        time_to_cal = torch.linspace(0, 20, 240).to(self.cfg.device).view(1, -1)
        cdf = self.model.calculate_cdf(w, time_to_cal)  # 计算CDF曲线
        pdf = self.model.calculate_pdf(w, time_to_cal)  # 计算PDF曲线
        survival_time = self.model.calculate_survial_time(w)  # 计算预测生存时间
        
        return dict(
            loss=loss.mean(),
            pdf=pdf,
            cdf=cdf,
            t1=t1,
            t2=t2,
            survival_time=survival_time,
            gt=data['gt'],
        )

    def matrix(self, epoch, data) -> dict:
        """
        计算并返回训练指标。
        
        参数：
            epoch (int): 当前训练轮次
            data (dict): 包含训练数据的字典
            
        返回：
            dict: 包含以下键的字典：
                - loss (float): 当前批次的平均损失值
        """
        return dict(
            loss=float(data['loss'].mean())
        )

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

if __name__ == '__main__':
    trainer = TrainerDR()
    trainer.train()
    save_model(trainer.model, 'output_fund_model.pth')
    
