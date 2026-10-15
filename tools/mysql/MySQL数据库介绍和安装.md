## 一、MySQL简介
MySQL是一个关系型数据库管理系统，由瑞典MySQL AB公司开发，目前属于Oracle公司。它是最流行的开源数据库之一，广泛应用于各类Web应用和大型网站。

MySQL的特点:
- 开源免费：MySQL是开源的，可以免费使用和修改。
- 跨平台：支持多种操作系统，如Windows、Linux、Mac OS等。
- 可扩展性：支持大规模的数据和复杂的查询操作。
- 安全性：提供多种安全特性，如用户认证、加密连接等。

目前推荐大家安装的是Mysql8这个版本

## 二、MySQL安装与配置
官网下载地址：https://dev.mysql.com/downloads/mysql/

1、配置全局环境变量
终端切到zsh模式
cd ~/.zshrc
export PATH=“/usr/local/mysql/bin:$PATH”
source ~/.zshrc
mysql -uroot -p 查看是否能用

## 三、MySQL可视化工具安装
SQLyog 是一款专为 MySQL 设计的高效图形化数据库管理工具，由 Webyog 公司开发，在数据库开发和运维领域广泛应用。