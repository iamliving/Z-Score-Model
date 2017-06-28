# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import scipy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

fm = 100  # down_perc=i_8*1.00/fm,fm=10 or 100,一般为10，用来控制净利润下降程度
n_var = 5  # 第一轮MDA的变量个数

def z_score_zb(data_tech):
    data_tech['x_1'] = (data_tech['current_asset'] - data_tech['current_debt']) / data_tech['asset'] * 100
    data_tech['x_2'] = (data_tech['wfplr'] + data_tech['yygj']) / data_tech['asset'] * 100
    data_tech['x_3'] = (data_tech['profit'] + data_tech['fiancial_expense']) / data_tech['asset'] * 100
    data_tech['x_4_1'] = data_tech['close_price'] * data_tech['traded_stocks'] / data_tech['debt'] * 100
    data_tech['x_4_2'] = data_tech['equity'] / data_tech['total_stocks'] * data_tech['restricted_stocks'] / data_tech[
        'debt'] * 100
    data_tech['x_4'] = data_tech['x_4_1'] + data_tech['x_4_2']
    # x_4 wind的计算方法有误，以自算为准
    data_tech['x_5'] = data_tech['income'] / data_tech['asset']
    # compute net profit change %
    data_tech['net_profit_perc'] = data_tech['net_profit_2015'] / data_tech['net_profit_2014'] - 1
    return

# group生成,net_profit decreased > a
# 分组,(净利润下降比例机器所在数据，所有样本数)
def group (down_perc, data_x, num_all) :
    group_tech = Series(np.zeros(num_all))
    for i_1 in range(0, num_all):
        if data_x['net_profit_perc'][i_1] > - down_perc:
            group_tech[i_1] = 1
    return group_tech

# 在数据data_x中寻找下降组,(数据，样本数),输出下降组的code、asset、group
def bad_comp(data_x, num_all):
    bad = DataFrame(np.zeros((1, 3)))
    j_1 = 0
    for i_2 in range(0, num_all):
        if data_x['group'][i_2] == 0:
            bad.ix[i_2] = [data_x['code'][i_2], data_x['asset'][i_2], data_x['group'][i_2]]
            j_1 = j_1 + 1
    return bad, j_1

# 在数据data_x中寻找正常组,(数据，样本数),输出正常组的code、asset、group
def good_comp(data_x, num_all):
    good = DataFrame(np.zeros((1, 3)))
    j_2 = 0
    for i_3 in range(0, num_all):
        if data_x['group'][i_3] == 1:
            # print data_tech['code'][i],data_tech['asset'][i],data_tech['group'][i]
            good.ix[i_3] = [data_x['code'][i_3], data_x['asset'][i_3], data_x['group'][i_3]]
            j_2 = j_2 + 1
    return good, j_2

# 删除空白的row 0, and 重新index
def reindex_good(good, amount_good):
    if good.ix[0, 0] == 0:
        good = good.drop([0])
    good[3] = range(0, amount_good)
    good.set_index(good[3], inplace=True)
    return good

def reindex_bad(bad, amount_bad):
    if bad.ix[0, 0] == 0:
        bad = bad.drop([0])
    # 重新index
    bad[3] = range(0, amount_bad)
    bad.set_index(bad[3], inplace=True)
    return bad

# 寻找资产值相差最少的
def min_ass_gap(amount_bad, amount_good, bad, good):
    best = DataFrame(np.zeros((1, 3)))
    for i_4 in range(0, amount_bad):
        best.ix[i_4] = good.ix[0]
        for k in range(0, amount_good):
            if abs(bad[1][i_4] - best[1][i_4]) > abs(bad[1][i_4] - good[1][k]):
                best.ix[i_4] = good.ix[k]
    return best

# 是否异均值，T-test
# 是否异均值，T-test
def test(amount_bad, data_l, n, n_var):
    keep = Series()
    not_keep = Series()
    for i_5 in range(n, n + n_var):
        m = amount_bad - 1
        j = m * 2 + 1
        x_0 = data_l.ix[0:m, i_5]
        x_1 = data_l.ix[(m + 1):j, i_5]
        # print var(x_0)
        # print var(x_1)
        f = scipy.stats.bartlett(x_0, x_1).pvalue
        # scipy.stats.levene(x_1_0,x_1_1)
        # print f
        if f > 0.025:
            # paired=False
            t = scipy.stats.ttest_ind(x_0, x_1, axis=0, equal_var=True).pvalue
        else:
            t = scipy.stats.ttest_ind(x_0, x_1, axis=0, equal_var=False).pvalue
        if t < 0.025:
            # print i-3
            # print t
            keep = keep.append(Series(i_5 - n))
        if t > 0.025:
            # print i-3
            # print t
            not_keep = not_keep.append(Series(i_5 - n))
    return keep, not_keep

# 贡献率之标准差
def v_std(n, data_l, n_var):
    d_std = Series()
    for i_6 in range(n, n + n_var):
        d_std = d_std.append(Series(np.std(data_l.ix[:, i_6])))
    return d_std

def my_lda(x, y):
    sklearn_lda = LDA(n_components=2)  # 建立LDA对象，叫nd.array，不怎么懂
    sklearn_lda.fit_transform(x, y)  # 减掉均值后的得分
    # lda_coef = DataFrame(sklearn_lda.coef_).transpose()
    # print 'first lda coef : \n', sklearn_lda.coef_
    # 若系数有-，跳出循环
    # jump(n_var,lda_coef)
    # for i_7 in range(0, n_var):
        # if lda_coef[0][i_7] < 0:
            # return
    # print 'all first lda_coef > 0'
    return sklearn_lda.coef_

def my_lda_after(x_after, x_all_after, y, amount_bad, num_all,x_all_group):
    sklearn_lda_after = LDA(n_components=2)  # 建立LDA对象，叫nd.array，不怎么懂
    lda_score_after = sklearn_lda_after.fit_transform(x_after, y)  # 减掉均值后的得分
    # lda_coef_after=DataFrame(sklearn_lda_after.coef_).transpose()
    # 若系数有-，跳出循环
    # jump(keep_count, lda_coef_after)
    # print 'final lda coef : \n', sklearn_lda_after.coef_
    # for i_7 in range(0, keep_count):
        # if lda_coef_after[0][i_7] < 0:
            # return
    # print 'all after lda_coef > 0 \n',
    # 预测
    lda_predict_after = sklearn_lda_after.predict(x_after)  # 预测的类别
    # lda_predit_after=Series(lda_predict_after)
    sum_0_after = amount_bad - sum(Series(lda_predict_after)[0:amount_bad])
    sum_1_after = sum(Series(lda_predict_after)[amount_bad:(amount_bad * 2)])
    h_perc_after = (sum_0_after + sum_1_after) / amount_bad / 2  # 准确率
    # print 'percents correct in training sample of final variables :',h_perc_after

    lda_predit_all = sklearn_lda_after.predict(x_all_after)  # 全部预测的类别

    # 全部预测正确率
    j_6 = 0
    for i_14 in range(0, num_all):
        if lda_predit_all[i_14] == x_all_group['group'][i_14]:
            j_6 += 1
    h_perc_all = j_6 * 1.00 / num_all  # 准确率
    # print 'percents correct in all of original variables :',h_perc_all

    # 得分，排序
    lda_score_sorted_0_after = np.sort(lda_score_after[0:amount_bad].transpose())
    lda_score_sorted_1_after = np.sort(lda_score_after[amount_bad:(2 * amount_bad)].transpose())

    return h_perc_after, h_perc_all, sklearn_lda_after.coef_, lda_score_sorted_0_after, lda_score_sorted_1_after

def circle(data_x, x_pure):

    down_perc = i_8 * 1.00 / fm
    # group生成,net_profit decreased > a
    # 分组,(净利润下降比例机器所在数据，所有样本数)
    data_x['group'] = group(down_perc, data_x, num_all)
    # 在数据data_x中寻找下降组,(数据，样本数)
    bad, amount_bad = bad_comp(data_x, num_all)
    # 在数据data_x中寻找正常组,(数据，样本数)
    good, amount_good = good_comp(data_x, num_all)
    # 删除空白的row 0, and 重新index
    good = reindex_good(good, amount_good)
    bad = reindex_bad(bad, amount_bad)
    # 寻找资产值相差最少的
    best = min_ass_gap(amount_bad, amount_good, bad, good)
    # 0组和对照组放一起
    mda_data = pd.concat([bad, best], ignore_index=True)
    # 删去无用列，mda_data 为训练样本,只要code和group
    mda_data = mda_data.drop([3, 1], axis=1)
    # 生成新表
    # x原为data_x中a,b,c,d,e列，生成x为all样本的数据
    x = pd.concat([x_pure, data_x['group']], axis=1)
    x_all_group = x
    x_all = x.drop(['code', 'group'], axis=1)

    # mda_data加财务数据，生成 mda_data_t
    mda_data_t = DataFrame(np.zeros((1, n_var + 2)),
                           columns=['code', 'group', 'x_1','x_2','x_3','x_4','x_5'])
    # i_9 = 0
    # j_3 = 0
    for i_9 in range(0, amount_bad * 2):
        for j_3 in range(0, num_all):
            if mda_data[0][i_9] == x['code'][j_3]:
                mda_data_t.ix[i_9] = x.ix[j_3]
    mda_data = pd.concat([mda_data, mda_data_t], axis=1)  # 为后面测试准备数据

    # 测试两组数据的code是否相等,输出相等的数字
    j_4 = 0
    for i_10 in range(0, amount_bad * 2):
        if mda_data[0][i_10] == mda_data['code'][i_10]:
            j_4 += 1
    # print j
    if j_4 == 2 * amount_bad:
        print down_perc, 'all code is equal'

    # keep 为9个变量本次循环中的序号，0-8
    keep, not_keep = test(amount_bad, mda_data, n, n_var)

    # 贡献率之标准差
    d_std = v_std(n, mda_data, n_var)

    # LDA 数据准备
    lda_x = mda_data.drop([0, 2, 'code', 'group'], axis=1)
    lda_y = mda_data[2]

    lda_coef = my_lda(lda_x, lda_y)
    # if lda_coef is None:
        # print 'first lda_coef is None'
        # continue
    # 贡献率
    devote = np.array(d_std) * lda_coef
    devote_keep = DataFrame(devote).ix[:, keep]
    for i_7 in tuple(keep):
        if devote_keep.ix[0, i_7] < 0:
            print 'devote for keep in 1st LDA ：\n', devote_keep
            return
    print 'devote for keep in 1st LDA > 0'

    # 利用贡献率删减变量
    for j_5 in tuple(not_keep):
        if devote_keep.ix[0].min() < devote[0][j_5]:
            keep = keep.append(Series(j_5))
    print 'final variables : \n', keep

    devote_keep_plus = DataFrame(devote).ix[:, keep]
    for i_5 in tuple(keep):
        if devote_keep_plus.ix[0, i_5] < 0:
            print 'devote for keep+ in 1st LDA ：\n', devote_keep_plus
            return
    print 'devote for keep+ in 1st LDA > 0'

    # 删减变量后
    x_after = lda_x.ix[:, keep]
    x_all_after = x_all.ix[:, keep]
    h_perc_after, h_perc_all, lda_coef_after, lda_score_sorted_0_after, lda_score_sorted_1_after = my_lda_after(x_after,
                                                                                                                x_all_after,
                                                                                                                lda_y,
                                                                                                                amount_bad,
                                                                                                                num_all,
                                                                                                                x_all_group)
    # if lda_coef_after is None:
        # continue
    # h_perc_after,lda_coef_after,lda_score_sorted_0_after,lda_score_sorted_1_after=my_lda(x_after,lda_y,amount_bad,keep.count())

    devote_after = np.array(d_std)[[keep]] * lda_coef_after
    for i_15 in range(0,keep.count()):
        if devote_after[0][i_15] < 0:
            print 'devote for keep+ in 2nd LDA ：\n', devote_after
            return
    print 'devote for keep+ in 2nd LDA > 0'

    # 区分度，各在good和best中的位置
    for i_11 in range(0, amount_bad):
        if lda_score_sorted_0_after.transpose()[i_11] > lda_score_sorted_1_after.transpose()[1]:
            s = (i_11 * 1.00) / num_all
            # s = (i_11 * 1.00) / amount_bad
            # print 'start point :',s
            break

    for i_12 in range(0, amount_bad):
        if lda_score_sorted_1_after.transpose()[i_12] > lda_score_sorted_0_after.transpose()[-1]:
            e = (i_12+amount_bad) * 1.00 / num_all
            # e = i_12 * 1.00 / amount_bad
            # print 'end   point :',e
            break
    gap = e - s
    hit_perc =DataFrame([h_perc_after, h_perc_all, s, e, gap, np.array(keep), lda_coef_after, down_perc,amount_bad]).transpose()
    return hit_perc


# 导入数据
data_tech_other = pd.read_csv('g:\data\z\kj.csv')  # 其他财务指标
data_tech = pd.read_csv('g:\data\z\kj-zscore.csv')  # 原z指标
# 计算Z-Score中指标,data_tech中含有全部所需指标，在其中增加z指标列和分组列
z_score_zb(data_tech)

# 合并
data_tech_t = pd.concat([data_tech_other, data_tech.ix[:, 17:20], data_tech.ix[:, 22:25], data_tech.ix[:, 1]], axis=1) # ,data_profit.ix[:,5:7]
data_x = data_tech_t.dropna(axis=0)  # 除掉缺失数据
# 最终数据 data_x, 含code的所有数据, esp: asset, profit_down_perc
num_all = data_x['code'].count()  # 样本数

# reindex
data_x[30] = range(0, num_all)
data_x.set_index(data_x[30], inplace=True)

x_pure =pd.concat([data_x.ix[:,0],data_x.ix[:,23:28]], axis=1)
# data_x 中含code的循环数据
data_x = pd.concat([x_pure, data_x.ix[:, 28:30]], axis=1)

# 场景循环
n=4 # 第一列财务指标的列号

hit_perc = DataFrame()

for i_8 in range(1, 11):
    hit_perc = hit_perc.append(circle(data_x, x_pure))
print 'h_perc_after, h_perc_all, s, e, gap, np.array(keep), lda_coef_after, down_perc,amount_bad \n', hit_perc

hit_percs = hit_perc
# hit_percs[8] = range(1, 11)
# hit_percs.set_index(hit_percs[8], inplace=True)
# hit_percs = hit_percs.drop(8, axis=1)
hit_percs.columns = ['hit_perc', 'hit_perc_all', 'start_point', 'end_point', 'gap', 'variables', 'coef', 'down_perc','amount_bad']
# hit_percs['gap']=hit_percs['end_point']-hit_percs['start_point']

print hit_percs  # , '\n', keep, hit_percs.plot()

# hit_percs['s_all']=hit_percs['start_point'] * amount_bad / num_all
# hit_percs['e_all']=(hit_percs['end_point'] * amount_bad + amount_bad)/ num_all

hit_percs_sort=hit_percs.sort_values(by='hit_perc_all',ascending=False)
hit_percs_sort.to_csv('g:\data\z\hit_percs_100.csv', index=False)
