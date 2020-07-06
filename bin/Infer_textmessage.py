#!/usr/bin/env python 
# -*- coding:utf-8 -*-

#   LYS
#   2020/5/16 19:12
#   Infer_textmessage.py

from IPython.core.pylabtools import figsize
# IPython.core.pylabtools.figsize(sizex, sizey)

import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats  # 统计推断包
import pymc3 as pm
import theano.tensor as tt

figsize(12.5, 3.5)  # set figure size

count_data = np.loadtxt("./data/txtdata.csv")
n_count_data = len(count_data)
plt.bar(np.arange(n_count_data), count_data, color="#348ABD")  # 画柱状图
plt.xlabel("Time (days)")
plt.ylabel("count of text-msgs received")
plt.title("Did the user's texting habits change over time?")
plt.xlim(0, n_count_data)  # 设置参数范围,图像所显示x轴的长度
plt.show()

with pm.Model() as model:
    alpha = 1.0 / count_data.mean()  # Recall count_data is the variable that holds our txt counts
    lambda_1 = pm.Exponential("lambda_1", alpha)
    lambda_2 = pm.Exponential("lambda_2", alpha)

    tau = pm.DiscreteUniform("tau", lower=0, upper=n_count_data)
    # 产生参数λ1、λ2、τ、α。其中λ1、λ2由机器随机产生，在训练过程中找到更优的τ值。

    with model:
        idx = np.arange(n_count_data)  # Index
        lambda_ = pm.math.switch(tau >= idx, lambda_1, lambda_2)
        # The switch() function assigns lambda_1 or lambda_2 as the value of lambda_, depending on what side of tau we are on.
        # The values of lambda_ up until tau are lambda_1 and the values afterwards are lambda_2.
        # tau, lambda_1, lambda_2 are random, lambda_ will be random. we are not fixing any variables yet!

    # Poisson(lambda_)
    with model:
        observation = pm.Poisson("obs", lambda_, observed=count_data)

    # MCMC
    with model:
        step = pm.Metropolis()
        trace = pm.sample(10000, tune=5000, step=step)
        #对后验分布进行10000次采样，指定采样器pm.Metropolis()，turn要调整的迭代次数（如果适用）（默认为无）预烧期

        lambda_1_samples = trace['lambda_1']
        lambda_2_samples = trace['lambda_2']
        tau_samples = trace['tau']
        #采样值存储在trace对象中，trace对象是一个字典

###########
figsize(12.5, 10)
# histogram of the samples:

ax = plt.subplot(311)
# subplot(numRows, numCols, plotNum)
# 图表的整个绘图区域被分成 numRows 行和 numCols 列plotNum 参数指定创建的 Axes 对象所在的区域numRows,numCols和plotNum这三个数都小于10的话,可以把它们缩写为一个整数,
# 例如 subplot(323) 和 subplot(3,2,3) 是相同的.

ax.set_autoscaley_on(False)
plt.hist(lambda_1_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="posterior of $\lambda_1$", color="#A60628", density=True)
#density=True归一化；bins=30 30个箱子，即柱子；histtype : {‘bar’, ‘barstacked’, ‘step’, ‘stepfilled’}, optional(选择展示的类型,默认为bar)

plt.legend(loc="upper left")
plt.title(r"""Posterior distributions of the variables $\lambda_1,\;\lambda_2,\;\tau$""")
plt.xlim([15, 30])
plt.xlabel("$\lambda_1$ value")

ax = plt.subplot(312)
ax.set_autoscaley_on(False)
plt.hist(lambda_2_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="posterior of $\lambda_2$", color="#7A68A6", density=True)
plt.legend(loc="upper left")
plt.xlim([15, 30])
plt.xlabel("$\lambda_2$ value")

plt.subplot(313)
w = 1.0 / tau_samples.shape[0] * np.ones_like(tau_samples)  #归一化
plt.hist(tau_samples, bins=n_count_data, alpha=1, label=r"posterior of $\tau$", color="#467821", weights=w, rwidth=2.)
plt.xticks(np.arange(n_count_data))

plt.legend(loc="upper left")
plt.ylim([0, .75])
plt.xlim([35, len(count_data) - 20])
plt.xlabel(r"$\tau$ (in days)")
plt.ylabel("probability")
plt.show()

#########
figsize(12.5, 5)
# tau_samples, lambda_1_samples, lambda_2_samples contain
# N samples from the corresponding posterior distribution
N = tau_samples.shape[0]
expected_texts_per_day = np.zeros(n_count_data)
for day in range(0, n_count_data):
    # ix is a bool index of all tau samples corresponding to
    # the switchpoint occurring prior to value of 'day'
    ix = day < tau_samples
    # Each posterior sample corresponds to a value for tau.
    # for each day, that value of tau indicates whether we're "before"
    # (in the lambda1 "regime") or
    #  "after" (in the lambda2 "regime") the switchpoint.
    # by taking the posterior sample of lambda1/2 accordingly, we can average
    # over all samples to get an expected value for lambda on that day.
    # As explained, the "message count" random variable is Poisson distributed,
    # and therefore lambda (the poisson parameter) is the expected value of
    # "message count".
    expected_texts_per_day[day] = (lambda_1_samples[ix].sum()
                                   + lambda_2_samples[~ix].sum()) / N

plt.plot(range(n_count_data), expected_texts_per_day, lw=4, color="#E24A33",
         label="expected number of text-messages received")
plt.xlim(0, n_count_data)
plt.xlabel("Day")
plt.ylabel("Expected # text-messages")
plt.title("Expected number of text-messages received")
plt.ylim(0, 60)
plt.bar(np.arange(len(count_data)), count_data, color="#348ABD", alpha=0.65,
        label="observed texts per day")

plt.legend(loc="upper left")
plt.show()
