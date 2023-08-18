# !/usr/bin/python3
# -*-coding:utf-8-*-
# Author: jiajia
# CreatDate: 2020/3/17 10:30
# !/home/xx/.virtualenvs/xx/bin/python
# encoding: utf-8
# gunicorn 配置文件
"""
运行方式
gunicorn -c app/gunicorn_conf.py service:main
"""
import multiprocessing
import os


# 监听端口
bind = '{}:{}'.format('0.0.0.0', 8014)
# 工作模式  使用gevent模式
#  某种情况下开启反而会更慢
worker_class = 'uvicorn.workers.UvicornWorker'
# worker_class = "gevent"
# 启动的进程数
# workers = multiprocessing.cpu_count() * 2 + 1
workers = 2
# 设置守护进程
daemon = True

# 设置日志记录水平
loglevel = 'debug'
# 设置错误信息日志路径
errorlog = "logs/error.log"
# 设置访问日志路径
accesslog = "logs/access.log"
capture_output = True
access_log_format = '%(t)s %(p)s %(h)s "%(r)s" %(s)s %(L)s %(b)s %(f)s" "%(a)s"'    #设置gunicorn访问日志格式，错误日志无法设置
# 代码更新时将重启工作，默认为False。此设置用于开发，每当应用程序发生更改时，都会导致工作重新启动。
reload = True
debug = True
# certfile='/etc/nginx/cert/1_wujie.rpaii.com_bundle.crt'
# keyfile='/etc/nginx/cert/2_wujie.rpaii.com.key'
# 设置超时时间
timeout = 600
