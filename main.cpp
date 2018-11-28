import pandas as pd
from io import StringIO
from sklearn import linear_model
import matplotlib.pyplot as plt
# 房屋面积与价格历史数据
csv_data = 'square_feet,price\n150,6450\n200,7450\n250,8450\n300,9450\n350,11450\n400,15450\n600,18450\n'
# 读入dataframe
df = pd.read_csv(StringIO(csv_data))
print(df)
#values.reshape(-1,1)把数组转换成一列
x = df['square_feet'].values.reshape(-1, 1)     
y = df['price']
# 建立线性回归模型
regr = linear_model.LinearRegression()
# 拟合
regr.fit(x, y)
# 得到直线的斜率、截距
a, b = regr.coef_, regr.intercept_
# 给出待预测面积
area = 238.5
# 方式1：根据直线方程计算的价格
print('price=', a * area + b)
# 方式2：根据predict方法预测的价格
print('price predicted =', regr.predict(area))
# 画图
# 1.真实的点
plt.scatter(x, y, color='blue',label='real price')
# 2.拟合的直线
plt.plot(x, regr.predict(x), color='red', linewidth=4, label='predicted price')
plt.xlabel('area')                  # x坐标标题
plt.ylabel('price')                 # y坐标标题
plt.legend(loc='lower right')       # 图例显示在右下方
plt.show()                          # 显示图形

from numpy import *    
import pandas as pd    
import matplotlib.pyplot as plt
#sigmoid函数    
def sigmoid(inX):    
    return 1.0/(1+exp(-inX))    
#梯度上升算法    
def gradAscent(dataMat,labelMat):    
    m,n=shape(dataMat)
    print(m,n)
    alpha=0.1    
    maxCycles=500    
    weights=array(ones((n,1)))    
    for k in range(maxCycles):     
        a=dot(dataMat,weights)    
        h=sigmoid(a)    
        error=(labelMat-h)    
        weights=weights+alpha*dot(dataMat.transpose(),error)    
    return weights    
#随机梯度上升    
def randomgradAscent(dataMat,label,numIter=50):    
    m,n=shape(dataMat)    
    weights=ones(n)    
    for j in range(numIter):    
        dataIndex=range(m)    
        for i in range(m):    
            alpha=40/(1.0+j+i)+0.2    
            randIndex_Index=int(random.uniform(0,len(dataIndex)))    
            randIndex=dataIndex[randIndex_Index]    
            h=sigmoid(sum(dot(dataMat[randIndex],weights)))    
            error=(label[randIndex]-h)    
            weights=weights+alpha*error[0,0]*(dataMat[randIndex].transpose())    
            del(dataIndex[randIndex_Index])    
    return weights    
#画图    
def plotBestFit(weights):    
    m=shape(dataMat)[0]    
    xcord1=[]    
    ycord1=[]    
    xcord2=[]    
    ycord2=[]    
    for i in range(m):    
        if labelMat[i]==1:    
            xcord1.append(dataMat[i,1])    
            ycord1.append(dataMat[i,2])    
        else:    
            xcord2.append(dataMat[i,1])    
            ycord2.append(dataMat[i,2])    
    plt.figure(1)    
    ax=plt.subplot(111)    
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')    
    ax.scatter(xcord2,ycord2,s=30,c='green')    
    x=arange(0.2,0.8,0.1)    
    y=array((-weights[0]-weights[1]*x)/weights[2])    
    print(shape(x))   
    print(shape(y))
    plt.sca(ax)    
    #plt.plot(x,y)      #ramdomgradAscent    
    plt.plot(x,y[0])   #gradAscent    
    plt.xlabel('density')    
    plt.ylabel('ratio_sugar')    
    plt.title('gradAscent logistic regression')    
    #plt.title('ramdom gradAscent logistic regression')    
    plt.show()    
#读入csv文件数据
df=pd.read_csv('/root/Data/watermelon_3a.csv')
m,n=shape(df)
df['idx']=ones((m,1))    
dataMat=array(df[['idx','density','ratio_sugar']].values[:,:])
labelMat=mat(df['label'].values[:]).transpose()
weights=gradAscent(dataMat,labelMat)    
#weights=randomgradAscent(dataMat,labelMat)    
plotBestFit(weights)

      import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    # 岭回归预测销量
    data = pd.read_csv('/root/Data/Advertising.csv')
    x = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']
    ##0.8表示80%的数据用来训练，如果是整数100，则表示100个数据用来训练，随机数种子random_state为1
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)
    #岭回归
    r = Ridge()
    ridge = r.fit(x_train, y_train)
    print("Training set score:{}".format(ridge.score(x,y)))
    print("ridge.coef_: {}".format(ridge.coef_))
    print("ridge.intercept_: {}".format(ridge.intercept_))
    order = y_test.argsort(axis=0) #
    y_test = y_test.values[order]
    x_test = x_test.values[order, :]
    y_predict = r.predict(x_test)
    mse = np.average((y_predict - np.array(y_test)) ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    print( 'MSE = ', mse,)
    print( 'RMSE = ', rmse)
    plt.figure(facecolor='w')
    t = np.arange(len(x_test))
    plt.plot(t,y_test,'r-', linewidth=2, label=u'real data')
    plt.plot(t,y_predict,'b-', linewidth=2, label=u'predicted data')
    plt.legend(loc='upper right')
    plt.title(u'predict sales by ridge regression', fontsize=18)
    plt.grid(b=True)
    plt.show()
