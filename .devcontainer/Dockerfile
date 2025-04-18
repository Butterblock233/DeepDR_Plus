# 使用 PyTorch 官方镜像作为基础镜像
FROM pytorch/pytorch:latest

# 替换 Ubuntu 官方源为阿里云镜像源
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.ustc.edu.cn/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu/|https://mirrors.ustc.edu.cn/ubuntu/|g' /etc/apt/sources.list

# 安装必要的工具和库
RUN apt-get update && apt install -y fish \ 
    && rm -rf /var/lib/apt/lists/*

# # 设置工作目录（可选）
# WORKDIR /workspace

# 设置默认命令（可选）
CMD ["fish"]