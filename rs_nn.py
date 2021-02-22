import os
import pandas as pypd
import numpy as pynp
import matplotlib.pyplot as pymap
import matplotlib.patches as pypt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import time
import six
import sys
import itertools as it

sys.modules['sklearn.externals.six'] = six
import mlrose
import mlrose_hiive as mlrh

winedf = pypd.read_csv("winequality-red.csv", delimiter=';')
winedf.notna
winedf.notnull
### There are no null values in the data set.
#Train and Test split
wine_test_sample = 0.2
winesample_rs = 10
WineX = winedf.iloc[:, :11]
Winey = winedf.iloc[:, 11]
#print(Winey)
WineX_train, WineX_test, Winey_train, Winey_test = train_test_split(WineX, Winey, random_state=winesample_rs, test_size=wine_test_sample)

WineSC=StandardScaler()
WineSC.fit(WineX_train)
WineX_trainSC=WineSC.transform(WineX_train)
WineX_testSC=WineSC.transform(WineX_test)
WineHE= OneHotEncoder()
Winey_trainHE = WineHE.fit_transform(Winey_train.values.reshape(-1, 1)).todense()
Winey_testHE = WineHE.transform(Winey_test.values.reshape(-1, 1)).todense()

##--------------
# #restart_iteration = [100,200,300,400,500,600,700,800,900,1000]
# restart_iteration = [100,200,300,400,500]
# WineNN = mlrose.NeuralNetwork(hidden_nodes=[10, 10], max_iters=1000, max_attempts=100,algorithm='gradient_descent',random_state=0, \
#                                       bias=True, is_classifier=True, restarts=6, curve=True,learning_rate=0.01, early_stopping=True, clip_max=5)
# WineNN.fit(WineX_trainSC, Winey_trainHE)
# #neural_fit.append(WineNN.fitness_curve)
# #Wine_t_loss.append(WineNN.loss)
# nn_train_accuracy = accuracy_score(Winey_trainHE, WineNN.predict(WineX_trainSC))
# #neural_acc_train.append(nn_train_accuracy)
# nn_test_accuracy = accuracy_score(Winey_testHE, WineNN.predict(WineX_testSC))
# #neural_acc_test.append(nn_test_accuracy)
# #print("RHC Average Fit: " , neuralr_ftavg)
# #print("RHC Maximum Fit: ", neuralr_ftavg)
# #print("RHC Average Time: ", prr_time)
# #print("RHC Maximum Time: ", prr_time)
# print("RHC Avg Loss : ", WineNN.loss)
# #print("RHC Maximum Loss: ", pynp.max(Winer_t_loss))
# print("RHC Max test accuracy: ", nn_test_accuracy)
# print("RHC Max train accuracy: ", nn_train_accuracy)


restart_iteration = [100,200,300,400,500]
#nn_fit=time.time()
#neural_fit=[]
#neural_pred = []
neuralr_acc_train = []
neuralr_acc_test = []
Wine_tr_err = []
Wine_t_err = []
Winer_t_loss = []
prr_time = []
t1 = time.time()
varlist = []
neuralr_ftavg = []
for r_i in restart_iteration:
    print("restart iteration : ",r_i)
    nn_test_accuracy = []
    neural_fit = []
    fts = []
    crvs = []
    nq_t = time.time()
    Wine_t_loss = []
    nn_train_accuracy = []
    neural_acc_train = []
    neural_acc_test = []
    for i, j in enumerate(range(6)):
        WineNN = mlrose.NeuralNetwork(hidden_nodes=[10, 10], max_iters=1000, max_attempts=100,algorithm='random_hill_climb',random_state=j + i, \
                                      bias=True, is_classifier=True, restarts=6, curve=True,learning_rate=0.1, early_stopping=True, clip_max=5)
        #nn_fit=time.time()
        WineNN.fit(WineX_trainSC, Winey_trainHE)
        neural_fit.append(WineNN.fitness_curve)
        Wine_t_loss.append(WineNN.loss)
        nn_train_accuracy = accuracy_score(Winey_trainHE, WineNN.predict(WineX_trainSC))
        neural_acc_train.append(nn_train_accuracy)
        nn_test_accuracy = accuracy_score(Winey_testHE, WineNN.predict(WineX_testSC))
        neural_acc_test.append(nn_test_accuracy)
        #pr_time.append(time.time() - nn_fit)
        print("test data : ",WineNN.loss, " : ", nn_train_accuracy, " : ", nn_test_accuracy)
    prr_time.append((time.time() - nq_t) / 6)
    #neural_ftavg.append(pynp.mean(neural_fit))
    neuralr_acc_test.append(pynp.mean(neural_acc_test))
    neuralr_acc_train.append(pynp.mean(neural_acc_train))
    Winer_t_loss.append(pynp.mean(Wine_t_loss))
    neuralr_ftavg.append(int(len(list(it.chain(*neural_fit))) / 6))
    varlist.append(r_i)

t_rhc = t1-time.time()
print("rhc time : ", t_rhc)
print("RHC Average Fit: " , pynp.mean(neuralr_ftavg))
print("RHC Maximum Fit: ",  pynp.max(neuralr_ftavg))
print("RHC Average Time: ", pynp.mean(prr_time))
print("RHC Maximum Time: ", pynp.max(prr_time))
print("RHC Avg Loss : ", pynp.max(Winer_t_loss))
print("RHC Maximum Loss: ", pynp.max(Winer_t_loss))
print("RHC Max test accuracy: ", pynp.max(neuralr_acc_test))
print("RHC Max train accuracy: ", pynp.max(neuralr_acc_train))

# pymap.figure()
# pymap.plot(neural_fit)
# pymap.title("Iteration vs Fitness Curve")
# pymap.xlabel('Number of Iterations')
# pymap.ylabel('FitnessCurve')
# pymap.savefig("rhc_fit.png")
# pymap.show()


pymap.figure()
pymap.plot(restart_iteration, neuralr_acc_train, '-o',color = 'green', label='Train',dashes=[4, 3])
pymap.plot(restart_iteration, neuralr_acc_test, '-o',color='blue', label='Test',dashes=[4, 3])
pymap.title("Iteration vs Accuracy")
pymap.xlabel('Number of Iterations')
pymap.ylabel('Accuracy for RHC')
pymap.legend(loc="best")
pymap.savefig("rhc_val2.png")
pymap.show()

pymap.figure()
pymap.title("Iteration vs Loss")
pymap.plot(restart_iteration, Winer_t_loss, '-o',color = 'green', label='Train',dashes=[4, 3])
pymap.xlabel('Number of RHC')
pymap.ylabel('Loss for RHC')
pymap.legend(loc="best")
pymap.savefig("rhc_val3.png")
pymap.show()

# pymap.figure()
# pymap.title("Iteration vs Fitness Curve")
# pymap.plot(restart_iteration, neural_fit, '-o',color = 'green', label='Train',dashes=[4, 3])
# pymap.xlabel('Number of RHC')
# pymap.ylabel('Loss for RHC')
# pymap.show()

#----------- Gradient Descent

restart_iteration = [100,200,300,400,500]
#neural_fit=[]
#neural_pred = []
neuralg_acc_train = []
neuralg_acc_test = []
Wine_tg_err = []
Wine_t_err = []
Wineg_t_loss = []
prg_time = []
t1 = time.time()
varlist = []
neuralg_ftavg = []
for r_i in restart_iteration:
    print("restart iteration : ",r_i)
    nn_test_accuracy = []
    neural_fit = []
    fts = []
    crvs = []
    nq_t = time.time()
    Wine_t_loss = []
    nn_train_accuracy = []
    neural_acc_train = []
    neural_acc_test = []
    for i, j in enumerate(range(6)):
        WineNN = mlrose.NeuralNetwork(hidden_nodes=[10, 5], max_iters=1000, max_attempts=100,algorithm='gradient_descent',random_state=j+i, \
                                      bias=True, is_classifier=True, restarts=6, curve=True,learning_rate=0.2, early_stopping=True, clip_max=5)
        #nn_fit=time.time()
        WineNN.fit(WineX_trainSC, Winey_trainHE)
        neural_fit.append(WineNN.fitness_curve)
        Wine_t_loss.append(WineNN.loss)
        nn_train_accuracy = accuracy_score(Winey_trainHE, WineNN.predict(WineX_trainSC))
        neural_acc_train.append(nn_train_accuracy)
        nn_test_accuracy = accuracy_score(Winey_testHE, WineNN.predict(WineX_testSC))
        neural_acc_test.append(nn_test_accuracy)
        #pr_time.append(time.time() - nn_fit)
    prg_time.append((time.time() - nq_t) / 5)
    #neural_ftavg.append(pynp.mean(neural_fit))
    neuralg_acc_test.append(pynp.mean(neural_acc_test))
    neuralg_acc_train.append(pynp.mean(neural_acc_train))
    Wineg_t_loss.append(pynp.mean(Wine_t_loss))
    neuralg_ftavg.append(int(len(list(it.chain(*neural_fit))) / 6))
    varlist.append(r_i)

t_rhc = t1-time.time()
print("GD time : ", t_rhc)
print("GD Average Fit: " , pynp.mean(neuralg_ftavg))
print("GD Maximum Fit: ",  pynp.max(neuralg_ftavg))
print("GD Average Time: ", pynp.mean(prg_time))
print("GD Maximum Time: ", pynp.max(prg_time))
print("GD Avg Loss : ", pynp.max(Wineg_t_loss))
print("GD Maximum Loss: ", pynp.max(Wineg_t_loss))
print("GD Max test accuracy: ", pynp.max(neuralg_acc_test))
print("GD Max train accuracy: ", pynp.max(neuralg_acc_train))

# pymap.figure()
# pymap.plot(neural_fit)
# pymap.title("Iteration vs Fitness Curve")
# pymap.xlabel('Number of Iterations')
# pymap.ylabel('FitnessCurve')
# pymap.savefig("rhc_fit.png")
# pymap.show()


pymap.figure()
pymap.plot(restart_iteration, neuralg_acc_train, '-o',color = 'green', label='Train',dashes=[4, 3])
pymap.plot(restart_iteration, neuralg_acc_test, '-o',color='blue', label='Test',dashes=[4, 3])
pymap.title("Iteration vs Accuracy")
pymap.xlabel('Number of Iterations')
pymap.ylabel('Accuracy for GD')
pymap.legend(loc="best")
pymap.savefig("gd_val2.png")
pymap.show()

pymap.figure()
pymap.title("Iteration vs Loss")
pymap.plot(restart_iteration, Wineg_t_loss, '-o',color = 'green', label='Train',dashes=[4, 3])
pymap.xlabel('Number of GD')
pymap.ylabel('Loss for GD')
pymap.legend(loc="best")
pymap.savefig("gd_val3.png")
pymap.show()

## ---- SA---
#neural_fit=[]
#neural_pred = []
neurals_acc_train = []
neurals_acc_test = []
Wine_tr_err = []
Wine_t_err = []
Wines_t_loss = []
prs_time = []
t1 = time.time()
varlist = []
neurals_ftavg = []
temparatures = [0.1,0.2,0.4,0.6,0.8]
constt = [0.001,0.002,0.004,0.006]
for c in constt:
    for r_i in temparatures:
        print("restart iteration : ",r_i)
        nn_test_accuracy = []
        neural_fit = []
        fts = []
        crvs = []
        nq_t = time.time()
        Wine_t_loss = []
        nn_train_accuracy = []
        neural_acc_train = []
        neural_acc_test = []
        for i, j in enumerate(range(6)):
            WineNN = mlrose.NeuralNetwork(hidden_nodes=[10, 10], max_iters=1000, max_attempts=100,schedule=mlrose.ExpDecay(init_temp = r_i, exp_const = c),algorithm='simulated_annealing',random_state=j + i, \
                                          bias=True, is_classifier=True, curve=True,learning_rate=0.2, early_stopping=True)
            #nn_fit=time.time()
            WineNN.fit(WineX_trainSC, Winey_trainHE)
            neural_fit.append(WineNN.fitness_curve)
            Wine_t_loss.append(WineNN.loss)
            nn_train_accuracy = accuracy_score(Winey_trainHE, WineNN.predict(WineX_trainSC))
            neural_acc_train.append(nn_train_accuracy)
            nn_test_accuracy = accuracy_score(Winey_testHE, WineNN.predict(WineX_testSC))
            neural_acc_test.append(nn_test_accuracy)
            #pr_time.append(time.time() - nn_fit)
        prs_time.append((time.time() - nq_t) / 6)
        #neural_ftavg.append(pynp.mean(neural_fit))
        neurals_acc_test.append(pynp.mean(neural_acc_test))
        neurals_acc_train.append(pynp.mean(neural_acc_train))
        Wines_t_loss.append(pynp.mean(Wine_t_loss))
        neurals_ftavg.append(int(len(list(it.chain(*neural_fit))) / 6))
        varlist.append(r_i)
        # WineNN = mlrose.NeuralNetwork(hidden_nodes=[10, 10], max_iters=1000, max_attempts=100,schedule=mlrose.ExpDecay(init_temp = r_i, exp_const = c),algorithm='simulated_annealing', \
        #                               bias=True, is_classifier=True, curve=True,learning_rate=0.01, early_stopping=True, clip_max=5)
        # #nn_fit=time.time()
        # WineNN.fit(WineX_trainSC, Winey_trainHE)
        # neural_fit.append(WineNN.fitness_curve)
        # Wine_t_loss.append(WineNN.loss)
        # nn_train_accuracy = accuracy_score(Winey_trainHE, WineNN.predict(WineX_trainSC))
        # neural_acc_train.append(nn_train_accuracy)
        # nn_test_accuracy = accuracy_score(Winey_testHE, WineNN.predict(WineX_testSC))
        # neural_acc_test.append(nn_test_accuracy)
        # #pr_time.append(time.time() - nn_fit)
        # prs_time.append((time.time() - nq_t))
        # #neural_ftavg.append(pynp.mean(neural_fit))
        # neurals_acc_test.append(neural_acc_test)
        # neurals_acc_train.append(neural_acc_train)
        # Wines_t_loss.append(Wine_t_loss)
        # neurals_ftavg.append(int(len(list(it.chain(*neural_fit)))))
        # varlist.append(r_i)

t_rhc = t1-time.time()
print("SA time : ", t_rhc)
print("SA Average Fit: " , pynp.mean(neurals_ftavg))
print("SA Maximum Fit: ",  pynp.max(neurals_ftavg))
print("SA Average Time: ", pynp.mean(prs_time))
print("SA Maximum Time: ", pynp.max(prs_time))
print("SA Avg Loss : ", pynp.max(Wines_t_loss))
print("SA Maximum Loss: ", pynp.max(Wines_t_loss))
print("SA Max test accuracy: ", pynp.max(neurals_acc_test))
print("SA Max train accuracy: ", pynp.max(neurals_acc_train))

# pymap.figure()
# pymap.plot(neural_fit)
# pymap.title("Iteration vs Fitness Curve")
# pymap.xlabel('Number of Iterations')
# pymap.ylabel('FitnessCurve')
# pymap.savefig("rhc_fit.png")
# pymap.show()


pymap.figure()
pymap.plot(varlist, neurals_acc_train, '-o',color = 'green', label='Train',dashes=[4, 3])
pymap.plot(varlist, neurals_acc_test, '-o',color='blue', label='Test',dashes=[4, 3])
pymap.title("Iteration vs Accuracy")
pymap.xlabel('Number of Iterations')
pymap.ylabel('Accuracy for SA')
pymap.legend(loc="best")
pymap.savefig("sa_val2.png")
pymap.show()

pymap.figure()
pymap.title("Iteration vs Loss")
pymap.plot(varlist, Wines_t_loss, '-o',color = 'green', label='Train',dashes=[4, 3])
pymap.xlabel('Number of SA')
pymap.ylabel('Loss for SA')
pymap.legend(loc="best")
pymap.savefig("sa_val3.png")
pymap.show()


## --GA --

#neural_fit=[]
#neural_pred = []
neuralga_acc_train = []
neuralga_acc_test = []
Wine_tr_err = []
Wine_t_err = []
Winega_t_loss = []
prga_time = []
t1 = time.time()
varlist = []
neuralga_ftavg = []
sampleprob = [0.1, 0.3, 0.5]
sizep = [25, 50, 75]
for sp in sizep:
    for r_i in sampleprob:
        print("test test", r_i)
        nn_test_accuracy = []
        neural_fit = []
        fts = []
        crvs = []
        nq_t = time.time()
        Wine_t_loss = []
        nn_train_accuracy = []
        neural_acc_train = []
        neural_acc_test = []
        for i, j in enumerate(range(5)):
            WineNN = mlrose.NeuralNetwork(hidden_nodes=[10, 10], max_iters=1000, max_attempts=100,
                                          algorithm='genetic_alg',pop_size=sp,mutation_prob=r_i, random_state=j + i,bias=True, is_classifier=True, curve=True, learning_rate=0.1,early_stopping=True, clip_max=5)
            #nn_fit=time.time()
            WineNN.fit(WineX_trainSC, Winey_trainHE)
            neural_fit.append(WineNN.fitness_curve)
            Wine_t_loss.append(WineNN.loss)
            nn_train_accuracy = accuracy_score(Winey_trainHE, WineNN.predict(WineX_trainSC))
            neural_acc_train.append(nn_train_accuracy)
            nn_test_accuracy = accuracy_score(Winey_testHE, WineNN.predict(WineX_testSC))
            neural_acc_test.append(nn_test_accuracy)
            #pr_time.append(time.time() - nn_fit)
        prga_time.append((time.time() - nq_t) / 5)
        #neural_ftavg.append(pynp.mean(neural_fit))
        neuralga_acc_test.append(pynp.mean(neural_acc_test))
        neuralga_acc_train.append(pynp.mean(neural_acc_train))
        Winega_t_loss.append(pynp.mean(Wine_t_loss))
        neuralga_ftavg.append(int(len(list(it.chain(*neural_fit))) / 5))
        varlist.append(r_i)

t_rhc = t1-time.time()
print("GA time : ", t_rhc)
print("GA Average Fit: " , pynp.mean(neuralga_ftavg))
print("GA Maximum Fit: ",  pynp.max(neuralga_ftavg))
print("GA Average Time: ", pynp.mean(prga_time))
print("GA Maximum Time: ", pynp.max(prga_time))
print("GA Avg Loss : ", pynp.max(Winega_t_loss))
print("GA Maximum Loss: ", pynp.max(Winega_t_loss))
print("GA Max test accuracy: ", pynp.max(neuralga_acc_test))
print("GA Max train accuracy: ", pynp.max(neuralga_acc_train))

# pymap.figure()
# pymap.plot(neural_fit)
# pymap.title("Iteration vs Fitness Curve")
# pymap.xlabel('Number of Iterations')
# pymap.ylabel('FitnessCurve')
# pymap.savefig("rhc_fit.png")
# pymap.show()


pymap.figure()
pymap.plot(varlist, neuralga_acc_train, '-o',color = 'green', label='Train',dashes=[4, 3])
pymap.plot(varlist, neuralga_acc_test, '-o',color='blue', label='Test',dashes=[4, 3])
pymap.title("Iteration vs Accuracy")
pymap.xlabel('Number of Iterations')
pymap.ylabel('Accuracy for GA')
pymap.legend(loc="best")
pymap.savefig("ga_val2.png")
pymap.show()

pymap.figure()
pymap.title("Iteration vs Loss")
pymap.plot(varlist, Winega_t_loss, '-o',color = 'green', label='Train',dashes=[4, 3])
pymap.xlabel('Number of GA')
pymap.ylabel('Loss for GA')
pymap.legend(loc="best")
pymap.savefig("ga_val3.png")
pymap.show()

# pymap.figure()
# pymap.title("Iteration vs Fitness Curve")
# pymap.plot(restart_iteration, neural_fit, '-o',color = 'green', label='Train',dashes=[4, 3])
# pymap.xlabel('Number of RHC')
# pymap.ylabel('Loss for RHC')
# pymap.show()


# ###loss curve
# pymap.figure()
# pymap.title("Iteration vs Loss")
# pymap.plot(varlist, Wineg_t_loss, '-o',color = 'blue', label='GD',dashes=[4, 3])
# pymap.plot(varlist, Winer_t_loss, '-o',color = 'red', label='RHC',dashes=[4, 3])
# pymap.plot(varlist, Wines_t_loss, '-o',color = 'green', label='SA',dashes=[4, 3])
# pymap.plot(varlist, Winega_t_loss, '-o',color = 'yellow', label='GA',dashes=[4, 3])
# pymap.xlabel('Number of Iterations')
# pymap.ylabel('Loss for GD,RHC,SA,GA')
# pymap.savefig("loss_val.png")
# pymap.show()
#
#
# ## Accuracy Test score
#
# pymap.figure()
# pymap.plot(varlist, neuralg_acc_train, '-o',color = 'blue', label='GD',dashes=[4, 3])
# pymap.plot(varlist, neuralr_acc_train, '-o',color = 'red', label='RHC',dashes=[4, 3])
# pymap.plot(varlist, neurals_acc_train, '-o',color = 'green', label='SA',dashes=[4, 3])
# pymap.plot(varlist, neuralga_acc_train, '-o',color = 'yellow', label='GA',dashes=[4, 3])
# pymap.legend(loc="upper left")
# pymap.title("Iteration vs Accuracy")
# pymap.xlabel('Number of Iterations')
# pymap.ylabel('Test Accuracy for GD,RHC,SA,GA')
# pymap.savefig("test_val.png")
# pymap.show()
#
# ##Accuracy Train score
# pymap.figure()
# pymap.plot(varlist, neuralg_acc_test, '-o',color = 'blue', label='GD',dashes=[4, 3])
# pymap.plot(varlist, neuralr_acc_test, '-o',color = 'red', label='RHC',dashes=[4, 3])
# pymap.plot(varlist, neurals_acc_test, '-o',color = 'green', label='SA',dashes=[4, 3])
# pymap.plot(varlist, neuralga_acc_test, '-o',color = 'yellow', label='GA',dashes=[4, 3])
# pymap.legend(loc="upper left")
# pymap.title("Iteration vs Accuracy")
# pymap.xlabel('Number of Iterations')
# pymap.ylabel('Test Accuracy for GD,RHC,SA,GA')
# pymap.savefig("test_val.png")
# pymap.show()
#
#
#


#-------------- SA ------------------

# neural_fit=[]
# neural_pred = []
# neural_acc_train = []
# neural_acc_test = []
# Wine_tr_err = []
# Wine_t_err = []
#
# rhc_nn = mlrose.NeuralNetwork(hidden_nodes = [10,10], max_iters = 1000, max_attempts = 100,algorithm = 'simulated_annealing',\
#                                   bias = True, is_classifier = True,restarts = 10,curve=True, \
#                                   learning_rate = 0.1, early_stopping = True, clip_max = 5)
# rhc_nn = rhc_nn.fit(WineX_trainSC, Winey_trainHE)
# rhc_nn_train_p = [rhc_nn.score(WineX_trainSC, Winey_trainHE)]
# rhc_nn_test_p = [rhc_nn.score(WineX_testSC, Winey_testHE)]
# pymap.figure()
# #print("test :", list(range(len(rhc_nn.fitness_curve))))
# #print("test1 : ",rhc_nn.fitness_curve)
# pymap.plot(list(range(len(rhc_nn.fitness_curve))), rhc_nn.fitness_curve, 'b', list(range(len(rhc_nn_test_p))), rhc_nn_test_p, 'r', alpha=0.5)
# pymap.plot(list(range(len(rhc_nn.fitness_curve))), rhc_nn.fitness_curve, 'g', list(range(len(rhc_nn_train_p))), rhc_nn_train_p, 'r', alpha=0.5)
# pymap.xlabel('Number of Iterations')
# pymap.ylabel('Error in SA')
# # pymap.savefig("rhc_val.png")
# pymap.show()
#
# for r_i in restart_iteration:
#     WineNN = mlrose.NeuralNetwork(hidden_nodes=[10, 10], max_iters=r_i, max_attempts=100,
#                                   algorithm='simulated_annealing', \
#                                   bias=True, is_classifier=True, restarts=10, curve=True, \
#                                   learning_rate=0.1, early_stopping=True, clip_max=5)
#     nn_fit=time.time()
#     WineNN.fit(WineX_trainSC, Winey_trainHE)
#     nn_fit_time = time.time()
#     nn_time = nn_fit_time - nn_fit
#     #print('NN Fitting Time = ', nn_time)
#     neural_fit.append(nn_time)
#     nn_pred=time.time()
#     nn_t_predict = WineNN.predict(WineX_testSC)
#     nn_pred_time = time.time()
#     nn_p_time = nn_pred_time - nn_pred
#     #print ('Test fitting time for NN model = ', nn_p_time)
#     neural_pred.append(nn_p_time)
#     nn_pred=time.time()
#     nn_tr_predict = WineNN.predict(WineX_trainSC)
#     nn_pred_time = time.time()
#     nn_pt_time = nn_pred_time - nn_pred
#     #print ('Test fitting time for NN model = ', nn_pt_time)
#
#     nn_train_accuracy = accuracy_score(Winey_trainHE, nn_tr_predict)
#     neural_acc_train.append(nn_train_accuracy)
#     nn_test_accuracy = accuracy_score(Winey_testHE, nn_t_predict)
#     neural_acc_test.append(nn_test_accuracy)
#
#
# pymap.figure()
# pymap.plot(restart_iteration, neural_acc_train, '-o',color = 'green', label='Train')
# pymap.plot(restart_iteration, neural_acc_test, '-o',color='blue', label='Test')
# pymap.xlabel('Simmulated Annealing')
# pymap.ylabel('Accuracy for SA')
# pymap.show()
#
#
# ##-------------GA -------------
#
# neural_fit=[]
# neural_pred = []
# neural_acc_train = []
# neural_acc_test = []
# Wine_tr_err = []
# Wine_t_err = []
#
#
#
# rhc_nn = mlrose.NeuralNetwork(hidden_nodes = [10,10], max_iters = 1000, max_attempts = 100,algorithm = 'genetic_alg',\
#                                   bias = True, is_classifier = True,restarts = 10,curve=True, \
#                                   learning_rate = 0.1, early_stopping = True, clip_max = 5)
# rhc_nn = rhc_nn.fit(WineX_trainSC, Winey_trainHE)
# rhc_nn_train_p = [rhc_nn.score(WineX_trainSC, Winey_trainHE)]
# rhc_nn_test_p = [rhc_nn.score(WineX_testSC, Winey_testHE)]
# pymap.figure()
# #print("test :", list(range(len(rhc_nn.fitness_curve))))
# #print("test1 : ",rhc_nn.fitness_curve)
# pymap.plot(list(range(len(rhc_nn.fitness_curve))), rhc_nn.fitness_curve, 'b', list(range(len(rhc_nn_test_p))), rhc_nn_test_p, 'r', alpha=0.5)
# pymap.plot(list(range(len(rhc_nn.fitness_curve))), rhc_nn.fitness_curve, 'g', list(range(len(rhc_nn_train_p))), rhc_nn_train_p, 'r', alpha=0.5)
# pymap.xlabel('Number of Iterations')
# pymap.ylabel('Error in SA')
# # pymap.savefig("rhc_val.png")
# pymap.show()
#
# for r_i in restart_iteration:
#     WineNN = mlrose.NeuralNetwork(hidden_nodes=[10, 10], max_iters=r_i, max_attempts=100,
#                                   algorithm='genetic_alg', \
#                                   bias=True, is_classifier=True, restarts=10, curve=True, \
#                                   learning_rate=0.1, early_stopping=True, clip_max=5)
#     nn_fit=time.time()
#     WineNN.fit(WineX_trainSC, Winey_trainHE)
#     nn_fit_time = time.time()
#     nn_time = nn_fit_time - nn_fit
#     #print('NN Fitting Time = ', nn_time)
#     neural_fit.append(nn_time)
#     nn_pred=time.time()
#     nn_t_predict = WineNN.predict(WineX_testSC)
#     nn_pred_time = time.time()
#     nn_p_time = nn_pred_time - nn_pred
#     #print ('Test fitting time for NN model = ', nn_p_time)
#     neural_pred.append(nn_p_time)
#     nn_pred=time.time()
#     nn_tr_predict = WineNN.predict(WineX_trainSC)
#     nn_pred_time = time.time()
#     nn_pt_time = nn_pred_time - nn_pred
#     #print ('Test fitting time for NN model = ', nn_pt_time)
#
#     nn_train_accuracy = accuracy_score(Winey_trainHE, nn_tr_predict)
#     neural_acc_train.append(nn_train_accuracy)
#     nn_test_accuracy = accuracy_score(Winey_testHE, nn_t_predict)
#     neural_acc_test.append(nn_test_accuracy)
#
#
# pymap.figure()
# pymap.plot(restart_iteration, neural_acc_train, '-o',color = 'green', label='Train')
# pymap.plot(restart_iteration, neural_acc_test, '-o',color='blue', label='Test')
# pymap.xlabel('Genetic Algorithm')
# pymap.ylabel('Accuracy for GA')
# pymap.show()