#결정 트리 -> 정보 이득이 최대가 되는 특성으로 데이터를 분류 
#분할 조건 -> 지니 불순도, 엔트로피, 분류오차
#엔트로피 -> 한 노드의 모든 샘플이 같은 클래스이면 엔트로피는 0, 클래스 분포가 균등하면 엔트로피는 최대값
#지니 불순도 -> 클래스가 완벽하게 섞여 있을 때, 최대
#분류 오차 -> 두 클래스가 같은 비율일 때 최대(0.5), 한 클래스의 비중이 ↑이면 값이 작아짐


#불순도 지표(분할 조건) 지표 비교
import matplotlib.pyplot as plt
import numpy as np

def gini(p):
    return (p)*(1- (p)) + (1 - p)*(1 - (1 - p))

def entropy(p):
    return -p*np.log2(p) - (1 - p)*np.log2((1 - p))

def error(p):
    return 1 - np.max([p, 1 - p])

x  = np.arange(0.0, 1.0, 0.01)

ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e * 0.5 if e else None for e in ent]
err = [error(i) for  i in x]

fig = plt.figure()
ax = plt.subplot(111)
for i, lab, ls, c in zip([ent, sc_ent, gini(x), err], ['Entropy', 'Entropy (scaled)', 'Gini Impurity', 'Misclassfication Error'],
                                                    ['-','-','--','-'], ['black','lightgray','red','green','cyan']):
    line = ax.plot(x, i, label = lab, linestyle = ls, lw = 2, color = c)

ax.legend(loc = 'upper left', bbox_to_anchor = (0.5,1.15), ncol = 5, fancybox = True, shadow = False)
ax.axhline(y = 0.5, linewidth = 1, color = 'k', linestyle = '--')
ax.axhline(y = 1.0, linewidth = 1, color = 'k', linestyle = '--')

plt.ylim([0.0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')
plt.show()