import numpy as np
import math
from matplotlib import pyplot as plt
import random
from functools import reduce
from collections import Counter
import seaborn as sns
import pandas as pd


# Parameter Define
Item_NUM = 100
Agent_NUM = 2
K = 10

# Functions
def Initial(item_num, agent_num, binary=False, bivalue=False, aa=0.2, bb=0.8):
    # Return agent list, item list, empty dictionary for allocation, valuation dict

    AList = []
    IList = []
    _= []
    vList = []
    # agent list
    for i in range(agent_num):
        AList.append(i)
        # AList.append('{}'.format(i+1) + "A")
        _.append([])
    # item list
    for j in range(item_num):
        IList.append(j)
    # item valuation
    for k in range(item_num):
        v = []
        for l in range(agent_num):
            if binary:
                v.append(np.random.randint(2))
            elif bivalue:
                vv = np.random.randint(2)
                if vv == 0:
                    vap = aa
                else:
                    vap = bb
                v.append(vap)
            else:
                v.append(round(random.random(),3))
        vList.append(v)

    # empty allocation dict
    allocations = dict(zip(AList,_))
    # valuation dict
    valuations = dict(zip(IList,vList))

    return AList, IList, allocations, valuations

def NashWelfare(allocations,valuations):
    nash_elmt = []
    _ = []
    for key in allocations.keys():
        for i in allocations[key]:

            _.append(valuations[i][key])
        utility = sum(_)
        _ = []

        nash_elmt.append(utility)
    nw = reduce(lambda x,y: x*y, nash_elmt)

    return nw


# def NashWelfare(allocations, valuations):
#     nash_elmt = []
#     _ = []
#     for key in allocations.keys():
#         for i in allocations[key]:
#             _.append(valuations[i][key])
#         utility = sum(_)
#         _ = []
#         if utility != 0 and utility != 0.0:
#             nash_elmt.append(utility)
#     nw = reduce(lambda x, y: x * y, nash_elmt)
#
#     return nw


def allocate(item_idx, allocations, valuations):

    if max(valuations[item_idx]) == 0 or max(valuations[item_idx]) == 0.0:
        return allocations

    poor_agent_eval = []
    poor_agent_idx = []
    nashwef = []
    # collect the agents has 0 item
    for key in allocations.keys():
        if len(allocations[key]) == 0:
            poor_agent_idx.append(key)
            poor_agent_eval.append(valuations[item_idx][key])

    # allocate the item to the highest bid
    if len(poor_agent_eval) != 0:
        Maxeval = max(poor_agent_eval)
        if Maxeval == 0 or Maxeval == 0.0:
            for key2 in allocations.keys():
                allocations[key2].append(item_idx)
                nashwef.append(NashWelfare(allocations, valuations))
                allocations[key2].pop()
            # print(nashwef)
            winner = nashwef.index(max(nashwef))
            allocations[winner].append(item_idx)
            return allocations
        else:
            winner_idx = poor_agent_eval.index(Maxeval)
            winner = poor_agent_idx[winner_idx]
            allocations[winner].append(item_idx)
            return allocations

    # allocate the item to the high nashwef
    else:
        for key3 in allocations.keys():
            # if valuations[item_idx][key] != 0 and valuations[item_idx][key] != 0.0:
            allocations[key3].append(item_idx)
            nashwef.append(NashWelfare(allocations,valuations))
            allocations[key3].pop()

        winner = nashwef.index(max(nashwef))
        allocations[winner].append(item_idx)
        return allocations




def max_k(lst, k=1):
  return sorted(lst, reverse=True)[:k]


def EFcheck(allocations, valuations, k = 1):
    counter = 0
    triger =  len(allocations) * (len(allocations)-1)

    for key in allocations.keys():
        # compute the valuation of it own bundle
        _ = []
        for i in allocations[key]:
            _.append(valuations[i][key])
        Vi_Ai = sum(_)

        # compute the valuation of other agents' bundles
        for key2 in allocations.keys():
            if key2 == key:

                continue
            else:
                _j = []
                for i in allocations[key2]:
                    _j.append(valuations[i][key])
                Vi_Aj = sum(_j)

                # compute Vi(Aj \setminus g)
                k_item = max_k(_j,k)
                for kitem in k_item:
                    Vi_Aj-=kitem
                Vi_Aj_mius_g = Vi_Aj

                # compare the valuation
                if Vi_Ai >= Vi_Aj_mius_g:
                    counter+=1
                    # print(counter)
                    if counter == triger:
                        # print('This allocation is EF-{}'.format(k))
                        return 1

                    else:
                        continue
                else:
                    # print('This allocation is NOT EF-{}'.format(k))

                    return 0


def PropCheck(allocations, valuations, k=1):
    counter = 0
    triger = len(allocations)

    # compute Vi_Ai
    for key in allocations.keys():
        # compute the valuation of it own bundle
        _ = []
        # print('key:',key)
        for i in allocations[key]:
            # print(i)
            _.append(valuations[i][key])
        Vi_Ai = sum(_)
        # print('V_{}'.format(key), Vi_Ai)
        # compute Vi_G
        all_index = []
        all_index_mins_Ai = []
        for key2 in allocations.keys():
            # compute the valuation of it own bundle
            for j in allocations[key2]:
                all_index.append(valuations[j][key])
                if key2 != key:
                    all_index_mins_Ai.append(valuations[j][key])
            Vi_G = sum(all_index)
        # print('V_{}_G'.format(key), Vi_G)

        # compute Vi(G \setminus C)
        k_item = max_k(all_index_mins_Ai, k)
        # print(k_item)
        for kitem in k_item:
            Vi_G -= kitem
        Vi_G_mius_C = Vi_G
        # print('V_{}_G_mins'.format(key),Vi_G_mius_C)

        # print(Vi_Ai)
        # print(Vi_G_mius_C)
        if round(Vi_Ai, 3) >= round((1 / len(allocations) ) * Vi_G_mius_C,3):
            counter += 1
            if counter >= triger:
                # print(counter)
                # print('This is a Prop-{} allocation'.format(k))
                return 1
        else:
            # print('This is not a Prop-{} allocation'.format(k))
            return 0


def main(item_num, agent_num, efk):
    Agent_list, Item_list, Allocations, Valuations = Initial(item_num,agent_num,bivalue=True)

    for i in range(item_num):
        Allocations = allocate(i,Allocations,Valuations)

    # r = PropCheck(Allocations,Valuations, k=efk)
    r = EFcheck(Allocations,Valuations, k=efk)

    return r, Allocations, Valuations


# result = {}
# for j in range(100):
#     _ = {}
#     # different
#     for i in range(1, 21):
#         r, a, v = main(i, Agent_NUM, K)
#         # if r != 1:
#         #     print(a)
#         #     print(v)
#         _[i] = r
#         # print(_)
#         # print(result)
#     C_ = Counter(_)
#     Cr = Counter(result)
#     result = dict(Cr + C_)
#
#
# tempDF = pd.DataFrame(result, index = [K])
# print(tempDF)
# print(tempDF.append(tempDF))
#
# print(result)
# plt.bar(result.keys(),result.values())
# plt.title('Agent number:{}'.format(Agent_NUM))
# plt.xlabel('item number')
# plt.ylabel('Possibility of EF{}'.format(K))
# plt.show()





# heatmap
hmDF = pd.DataFrame()
# 100 times repeat experiments
for ki in range(0, 11):
    print('Prop-{}'.format(ki))
    result = {}
    for j in range(100):
        _ = {}
        # different
        for i in range(1, 21):
            r, a, v = main(i, Agent_NUM, ki)
            if r != 1:
                print(a)
                print(v)
            _[i] = r
            # print(_)
            # print(result)
        C_ = Counter(_)
        Cr = Counter(result)
        result = dict(Cr + C_)

    tempDF = pd.DataFrame(result, index=[ki])
    # print(tempDF)

    hmDF= hmDF.append(tempDF,ignore_index=True)

print(hmDF)

plt.figure(dpi=120)
fig, ax = plt.subplots(figsize=(15, 9))
sns.heatmap(data=hmDF,
            cmap=plt.get_cmap('Blues'),linewidths=0.25,
            linecolor='black', ax=ax, annot=True,
            fmt='.1f', robust=True)

ax.set_title('Possibility of EF-N for Agent number:{}'.format(Agent_NUM))
ax.set_xlabel('item number')
ax.set_ylabel('EF-N')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)


plt.show()



# Step check
# Agent_list, Item_list, Allocations, Valuations = Initial(item_num=2, agent_num=3)
#
# Valuations[0] = [0.63, 0.58, 0.02] #2
# Valuations[1] = [0.17, 0.60, 0.80] #1
#
#
# # Valuations[3] = [1,1,0] #
# # Valuations[4] = [1,1,0] #
# # Valuations[5] = [1,0,0] #
#
#
# print(Valuations)
# for i in range(2):
#     Allocations = allocate(i,Allocations,Valuations)
#
# print(Allocations)
#
# r = PropCheck(Allocations,Valuations, k=2)
#
# print(r)
# print(0.60+0.58)
#
