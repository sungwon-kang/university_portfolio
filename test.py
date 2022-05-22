
# 모델 모듈 불러오기
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

# 데이터 셋, 분할 및 교차 검증 모듈
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import numpy as np
# 랜덤 포레스트 설명
# 랜덤 포레스트 트리의 깊이나 노드의 수에 따른 성능 차이, 나의 생각 : 실행시간 등
# 랜덤 포레스트에 최상의 경우

#digit 데이터 셋 로드
digit=datasets.load_digits()

# SVM 모델 하이퍼 매개변수 리스트 정의
lst_kernel = ['rbf','linear','poly','sigmoid']
lst_margin = [0.001,0.1,10,1000]

# Random forest 모델 하이퍼 매개 변수 리스트 정
lst_estimators =[10,100,1000,10000]
lst_max_depth =[1,10,100,1000]


#훈련 집합과 테스트 집합을 6:4 비율로 나누어서 분할한다.
np.random.seed(0)
x_train,x_test,y_train,y_test=train_test_split(digit.data,digit.target,train_size=0.6)

#%%
# 모델 학습을 수행하는 함수
def fitting(model, fst, sec):
    
    # model 조건 분기문
    if model=='svm':    
        s = svm.SVC(gamma=0.001,kernel=fst,C=sec)
    elif model=='rf':
        s = RandomForestClassifier(n_estimators=fst, max_depth=sec)
    
    # 모델 학습
    s.fit(x_train,y_train)
    res=s.predict(x_test)
    
    return s,res


# 혼동 행렬을 생성하는 함수
def get_confusion_mat(res):
    conf = np.zeros((10,10))            #10x10 0으로 채운 행렬 생성
    for i in range(len(res)):       	#예측한 값이 들어간 res의 길이만큼 반복
        conf[res[i]][y_test[i]]+=1 	    #res[i]측정한 값, y_test[i]실제 값 위치에 +1
        
    #print(conf)		                # 혼동 행렬 출력
    return conf

# 정확률을 계산하는 함수
def sum_diagonal_mat(conf):
    no_correct =0
    for i in range(10):
        no_correct+=conf[i][i]          # 혼동행렬의 대각선 부분을 모두 더한다. 
    
    return no_correct

# 
def RUN(model, fst_list, sec_list):
    for i in fst_list:                  # 첫번째 하이퍼 매개변수 리스트 요소만큼 반복
        for j in sec_list:              # 두번째 하이퍼 매개변수 리스트 요소만큼 반복
            s,res = fitting(model,i,j)      # 모델 학습 후 모델 객체s와 예측결과 리스트 res 반환받음
            conf = get_confusion_mat(res)   # 리스트 res를 전달하고, 혼동행렬을 반환받음
            accuracy = sum_diagonal_mat(conf)/len(res)*100              # 혼동행렬conf를 전달하고 정확률 계산
            accuracies=cross_val_score(s,digit.data,digit.target,cv=5)  # 교차검증함수 호출 (5회 실행)
            
            # 출력
            print('[1] [lst1]:',i,' [lst2]:',j)
            print('[2] [Accuracy]:',round(accuracy,5),'[5-corss Accuracies]:',accuracies.round(5))
            print('[3] [Accuracies_mean]:',accuracies.mean()*100,'[Accuracies_std]:',accuracies.std())  
            print('\n')

    
#%%
def main():
    # 주석을 풀고, 원하는 모델의 성능을 측정
    #SVM 모델
    RUN('svm',lst_kernel,lst_margin)
    
    #Random Forest 모델
    #RUN('rf',lst_estimators,lst_max_depth)
        
main();

