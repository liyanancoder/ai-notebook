## 一、Docker 介绍
Docker 是一个开源的容器化平台，它允许开发者将应用程序及其所有依赖（库、框架、环境变量等）打包成一个轻量级、可移植的"容器"。这个容器可以在任何安装了Docker的环境中运行，而不需要关心底层系统的差异。
## 二、Docker 使用
### 1、安装Docker
### 2、基于Docker部署redis-stack
官网：https://redis.io/docs/latest/operate/oss_and_stack/install/archive/install-stack/docker/
#### 2.1 拉取镜像
docker pull redis/redis-stack
#### 2.2 启动redis
先启动容器：docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest

再在浏览器输入：http://localhost:8001

全部同意


### 3、基于Docker部署FastGPT
源码位置：https://github.com/labring/FastGPT
#### 3.1 下载项目代码
#### 3.2 修改配置文件
参考文档：https://doc.fastgpt.cn/zh-CN/self-host/deploy/docker

config.json
docker-compose.yml

#### 3.3 拉取镜像
cd FastGPT/deploy/dev
docker compose up -d

#### 3.4 启动FastGPT 
在浏览器输入：http://localhost:3000

#### 3.5 配置模型及供应商