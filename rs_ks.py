import numpy as pynp
import time
import itertools
#from functools import reduce
import matplotlib.pyplot as pymap
import mlrose_hiive as mlrh


neural_fit=[]
neural_ftavg= []
neural_time = []
neural_crv = []
varlist = []
restart_iteration = [1,50,100,150,200,250,300]
t1=time.time()
for r_i in restart_iteration:

    sts = []
    fts = []
    crvs = []
    nq_t = time.time()
    #for i, j in enumerate([pynp.random.randint(7,size=7) for hm in range(10)]):
    for i,j in enumerate([pynp.random.randint(5, size=5) for hm in range(10)]):
        #test = mlrh.DiscreteOpt(length=7, fitness_fn=mlrh.Knapsack([8, 6,12, 10, 5,11,9],[1, 2, 3, 4,5,6,7],0.8), maximize=True, max_val=7)
        test = mlrh.DiscreteOpt(length=5,fitness_fn=mlrh.Knapsack([8, 6,12, 10, 5], [1, 2, 3, 4, 5], 0.7),maximize=True, max_val=5)
        #st, ft, crv = mlrh.random_hill_climb(test, max_attempts=20,max_iters=50, restarts=r_i,init_state=j, curve=True,random_state=1000 + i)
        st, ft, crv = mlrh.random_hill_climb(test, max_attempts=50, max_iters=1000, restarts=r_i, init_state=j,curve=True, random_state=1000 + i)

        sts.append(st)
        fts.append(ft)
        print("test: ", ft)
        crvs.append(crv)

    neural_time.append((time.time() - nq_t)/10)
    neural_ftavg.append(pynp.mean(fts))
    neural_crv.append(int(len(list(itertools.chain(*crvs))) / 10))
    varlist.append(r_i)

print(neural_ftavg)
t_rhc = t1-time.time()
print("rhc time : ", t_rhc)
print("RHC Average Fit: " , pynp.mean(neural_ftavg))
print("RHC Maximum Fit: ",  pynp.max(neural_ftavg))
print("RHC Average Time: ", pynp.mean(neural_time))
print("RHC Maximum Time: ", pynp.max(neural_time))
pymap.figure()
pymap.plot(varlist, neural_time, 'o-',label='Time',color='green',dashes=[4, 3])
pymap.plot(varlist, neural_ftavg, 'o-',label='Fitness',color='red',dashes=[4, 3])
pymap.plot(varlist, neural_crv, 'o-',label='Fitness Curve',color='blue',dashes=[4, 3])
pymap.tick_params(axis='x', rotation=90)

pymap.title('Knapsack Problems - RHC')
pymap.xlabel('Restarts')
pymap.ylabel("Time - fitness - Curve ")
pymap.legend(loc="best")
pymap.grid()
pymap.savefig("nq_rhc.png")
pymap.show()

pymap.figure()
#pymap.plot(varlist, neural_time, 'o-',label='Time',color='green',dashes=[4, 3])
pymap.plot(varlist, neural_ftavg, 'o-',label='Fitness',color='red',dashes=[4, 3])
#pymap.plot(varlist, neural_crv, 'o-',label='Fitness Curve',color='blue',dashes=[4, 3])
pymap.tick_params(axis='x', rotation=90)

pymap.title('Knapsack Problems - RHC')
pymap.xlabel('Restarts')
pymap.ylabel("Fitness")
pymap.legend(loc="best")
pymap.grid()
pymap.savefig("nq_rhc1.png")
pymap.show()

pymap.figure()
#pymap.plot(varlist, neural_time, 'o-',label='Time',color='green',dashes=[4, 3])
#pymap.plot(varlist, neural_ftavg, 'o-',label='Fitness',color='red',dashes=[4, 3])
pymap.plot(varlist, neural_crv, 'o-',label='Fitness Curve',color='blue',dashes=[4, 3])
pymap.tick_params(axis='x', rotation=90)

pymap.title('Knapsack Problems - RHC')
pymap.xlabel('Restarts')
pymap.ylabel("Fitness Curve")
pymap.legend(loc="best")
pymap.grid()
pymap.savefig("nq_rhc2.png")
pymap.show()

pymap.figure()
pymap.plot(varlist, neural_time, 'o-',label='Time',color='green',dashes=[4, 3])
#pymap.plot(varlist, neural_ftavg, 'o-',label='Fitness',color='red',dashes=[4, 3])
#pymap.plot(varlist, neural_crv, 'o-',label='Fitness Curve',color='blue',dashes=[4, 3])
pymap.tick_params(axis='x', rotation=90)

pymap.title('Knapsack Problems - RHC')
pymap.xlabel('Restarts')
pymap.ylabel("Time")
pymap.legend(loc="best")
pymap.grid()
pymap.savefig("nq_rhc3.png")
pymap.show()

##-----SA-----

neural_fit=[]
neural_ftavg= []
neural_time = []
neural_crv = []
varlist = []
temparatures = [0.1,0.2,0.4,0.6,0.8,1.0]
constt = [0.001,0.002,0.004,0.006,0.008,0.01]
t1 = time.time()
for c in constt:
    for r_i in temparatures:
        sts = []
        fts = []
        crvs = []
        nq_t = time.time()
        for i, j in enumerate([pynp.random.randint(5, size=5) for hm in range(7)]):
            test = mlrh.DiscreteOpt(length=5,fitness_fn=mlrh.Knapsack([8, 6, 12, 10, 5], [1, 2, 3, 4, 5], 0.7),maximize=True, max_val=5)
            st, ft, crv = mlrh.simulated_annealing(test, max_attempts=20,max_iters=50,init_state=j, curve=True,random_state=1000 + i,schedule=mlrh.ExpDecay(init_temp=r_i,exp_const=c))
            sts.append(st)
            fts.append(ft)
            crvs.append(crv)
        neural_time.append((time.time() - nq_t)/10)
        neural_ftavg.append(pynp.mean(fts))
        neural_crv.append(int(len(list(itertools.chain(*crvs))) / 10))
        varlist.append(f'c_t={c}_{r_i}')

t_sa = t1-time.time()
print("SA time : ", t_sa)
print("SA Average Fit: " , pynp.mean(neural_ftavg))
print("SA Maximum Fit: ",  pynp.max(neural_ftavg))
print("SA Average Time: ", pynp.mean(neural_time))
print("SA Maximum Time: ", pynp.max(neural_time))
pymap.figure()
pymap.plot(varlist, neural_time, 'o-',label='Time',color='green',dashes=[4, 3])
pymap.plot(varlist, neural_ftavg, 'o-',label='Fitness',color='red',dashes=[4, 3])
pymap.plot(varlist, neural_crv, 'o-',label='Fitness Curve',color='blue',dashes=[4, 3])
pymap.tick_params(axis='x', rotation=90)

pymap.title('Knapsack Problems - SA')
pymap.xlabel('Constant and Temparatures')
pymap.ylabel("Time - fitness - Curve")
pymap.legend(loc="best")
pymap.grid()
pymap.savefig("nq_sa.png")
pymap.show()

pymap.figure()
#pymap.plot(varlist, neural_time, 'o-',label='Time',color='green',dashes=[4, 3])
pymap.plot(varlist, neural_ftavg, 'o-',label='Fitness',color='red',dashes=[4, 3])
#pymap.plot(varlist, neural_crv, 'o-',label='Fitness Curve',color='blue',dashes=[4, 3])
pymap.tick_params(axis='x', rotation=90)

pymap.title('Knapsack Problems - SA')
pymap.xlabel('Constant and Temparatures')
pymap.ylabel("Fitness")
pymap.legend(loc="best")
pymap.grid()
pymap.savefig("nq_sa1.png")
pymap.show()

pymap.figure()
#pymap.plot(varlist, neural_time, 'o-',label='Time',color='green',dashes=[4, 3])
#pymap.plot(varlist, neural_ftavg, 'o-',label='Fitness',color='red',dashes=[4, 3])
pymap.plot(varlist, neural_crv, 'o-',label='Fitness Curve',color='blue',dashes=[4, 3])
pymap.tick_params(axis='x', rotation=90)

pymap.title('Knapsack Problems - SA')
pymap.xlabel('Constant and Temparatures')
pymap.ylabel("Fitness Curve ")
pymap.legend(loc="best")
pymap.grid()
pymap.savefig("nq_sa2.png")
pymap.show()

pymap.figure()
pymap.plot(varlist, neural_time, 'o-',label='Time',color='green',dashes=[4, 3])
#pymap.plot(varlist, neural_ftavg, 'o-',label='Fitness',color='red',dashes=[4, 3])
#pymap.plot(varlist, neural_crv, 'o-',label='Fitness Curve',color='blue',dashes=[4, 3])
pymap.tick_params(axis='x', rotation=90)

pymap.title('Knapsack Problems - SA')
pymap.xlabel('Constant and Temparatures')
pymap.ylabel("Time")
pymap.legend(loc="best")
pymap.grid()
pymap.savefig("nq_sa3.png")
pymap.show()

#----- GA -----

neural_fit=[]
neural_ftavg= []
neural_time = []
neural_crv = []
varlist = []
sampleprob = [0.1, 0.3, 0.5, 0.7, 0.8]
sizep = [25, 50, 75, 100, 125,150,175, 200]
t1 = time.time()
for sp in sizep:
    for r_i in sampleprob:
        sts = []
        fts = []
        crvs = []
        nq_t = time.time()
        for i, j in enumerate([pynp.random.randint(5, size=5) for hm in range(10)]):
            test = mlrh.DiscreteOpt(length=5,fitness_fn=mlrh.Knapsack([8, 6, 12, 10, 5], [1, 2, 3, 4, 5], 0.7),maximize=True, max_val=5)
            st, ft, crv = mlrh.genetic_alg(test, max_attempts=20,max_iters=50, mutation_prob=r_i,curve=True,random_state=1000 + i,pop_size=sp)
            sts.append(st)
            fts.append(ft)
            crvs.append(crv)

        neural_time.append((time.time() - nq_t)/10)
        neural_ftavg.append(pynp.mean(fts))
        neural_crv.append(int(len(list(itertools.chain(*crvs))) / 10) * sp)
        varlist.append(f's_p={r_i}_{sp}')

t_ga = t1-time.time()
print("GA time : ", t_ga)
print("GA Average Fit: " , pynp.mean(neural_ftavg))
print("GA Maximum Fit: ",  pynp.max(neural_ftavg))
print("GA Average Time: ", pynp.mean(neural_time))
print("GA Maximum Time: ", pynp.max(neural_time))
pymap.figure()
pymap.plot(varlist, neural_time, 'o-',label='Time',color='green',dashes=[4, 3])
pymap.plot(varlist, neural_ftavg, 'o-',label='Fitness',color='red',dashes=[4, 3])
pymap.plot(varlist, neural_crv, 'o-',label='Fitness Curve',color='blue',dashes=[4, 3])
pymap.tick_params(axis='x', rotation=90)

pymap.title('Knapsack Problems - GA')
pymap.xlabel('Sample Size and Sample Probability')
pymap.ylabel("Time - fitness - Curve")
pymap.legend(loc="best")
pymap.grid()
pymap.savefig("nq_ga.png")
pymap.show()

pymap.figure()
#pymap.plot(varlist, neural_time, 'o-',label='Time',color='green',dashes=[4, 3])
pymap.plot(varlist, neural_ftavg, 'o-',label='Fitness',color='red',dashes=[4, 3])
#pymap.plot(varlist, neural_crv, 'o-',label='Fitness Curve',color='blue',dashes=[4, 3])
pymap.tick_params(axis='x', rotation=90)

pymap.title('Knapsack Problems - GA')
pymap.xlabel('Sample Size and Sample Probability')
pymap.ylabel("Fitness")
pymap.legend(loc="best")
pymap.grid()
pymap.savefig("nq_ga1.png")
pymap.show()

pymap.figure()
#pymap.plot(varlist, neural_time, 'o-',label='Time',color='green',dashes=[4, 3])
#pymap.plot(varlist, neural_ftavg, 'o-',label='Fitness',color='red',dashes=[4, 3])
pymap.plot(varlist, neural_crv, 'o-',label='Fitness Curve',color='blue',dashes=[4, 3])
pymap.tick_params(axis='x', rotation=90)

pymap.title('Knapsack Problems - GA')
pymap.xlabel('Sample Size and Sample Probability')
pymap.ylabel("Fitness Curve")
pymap.legend(loc="best")
pymap.grid()
pymap.savefig("nq_ga2.png")
pymap.show()

pymap.figure()
pymap.plot(varlist, neural_time, 'o-',label='Time',color='green',dashes=[4, 3])
#pymap.plot(varlist, neural_ftavg, 'o-',label='Fitness',color='red',dashes=[4, 3])
#pymap.plot(varlist, neural_crv, 'o-',label='Fitness Curve',color='blue',dashes=[4, 3])
pymap.tick_params(axis='x', rotation=90)

pymap.title('Knapsack Problems - GA')
pymap.xlabel('Sample Size and Sample Probability')
pymap.ylabel("Time")
pymap.legend(loc="best")
pymap.grid()
pymap.savefig("nq_ga3.png")
pymap.show()


##----- MIMIC -----

neural_fit=[]
neural_ftavg= []
neural_time = []
neural_crv = []
varlist = []
sampleprob = [0.01,0.03,0.05,0.07,0.09]
sizep = [25,50,75,100,125,150]
t1=time.time()
for sp in sizep:
    for r_i in sampleprob:
        sts = []
        fts = []
        crvs = []
        nq_t = time.time()
        for i, j in enumerate([pynp.random.randint(5, size=5) for hm in range(10)]):
            test = mlrh.DiscreteOpt(length=5,fitness_fn=mlrh.Knapsack([8, 6, 12, 10, 5], [1, 2, 3, 4, 5], 0.7),maximize=True, max_val=5)
            st, ft, crv = mlrh.mimic(test, max_attempts=20,max_iters=50, curve=True,random_state=1000 + i,pop_size=sp,keep_pct=r_i)
            sts.append(st)
            fts.append(ft)
            crvs.append(crv)

        neural_time.append((time.time() - nq_t)/10)
        neural_ftavg.append(pynp.mean(fts))
        neural_crv.append(int(len(list(itertools.chain(*crvs))) / 10) * sp)
        varlist.append(f'S_p%={sp}_{r_i}')

t_mimic = t1-time.time()
print("mimic time : ", t_mimic)
print("MIMIC Average Fit: " , pynp.mean(neural_ftavg))
print("MIMIC Maximum Fit: ",  pynp.max(neural_ftavg))
print("MIMIC Average Time: ", pynp.mean(neural_time))
print("MIMIC Maximum Time: ", pynp.max(neural_time))

pymap.figure()
pymap.plot(varlist, neural_time, 'o-',label='Time',color='green',dashes=[4, 3])
pymap.plot(varlist, neural_ftavg, 'o-',label='Fitness',color='red',dashes=[4, 3])
pymap.plot(varlist, neural_crv, 'o-',label='Fitness Curve',color='blue',dashes=[4, 3])
pymap.tick_params(axis='x', rotation=90)

pymap.title('Knapsack Problems - MIMIC')
pymap.xlabel('Percentage and Size of Sample')
pymap.ylabel("Time - fitness - Curve ")
pymap.legend(loc="best")
pymap.grid()
pymap.savefig("nq_mimic.png")
pymap.show()

pymap.figure()
#pymap.plot(varlist, neural_time, 'o-',label='Time',color='green',dashes=[4, 3])
pymap.plot(varlist, neural_ftavg, 'o-',label='Fitness',color='red',dashes=[4, 3])
#pymap.plot(varlist, neural_crv, 'o-',label='Fitness Curve',color='blue',dashes=[4, 3])
pymap.tick_params(axis='x', rotation=90)

pymap.title('Knapsack Problems - MIMIC - Fitness')
pymap.xlabel('Percentage and Size of Sample')
pymap.ylabel("Fitness of Sample")
pymap.legend(loc="best")
pymap.grid()
pymap.savefig("nq_mimic1.png")
pymap.show()

pymap.figure()
#pymap.plot(varlist, neural_time, 'o-',label='Time',color='green',dashes=[4, 3])
#pymap.plot(varlist, neural_ftavg, 'o-',label='Fitness',color='red',dashes=[4, 3])
pymap.plot(varlist, neural_crv, 'o-',label='Fitness Curve',color='blue',dashes=[4, 3])
pymap.tick_params(axis='x', rotation=90)

pymap.title('Knapsack Problems - MIMIC - Fitness Curve')
pymap.xlabel('Percentage and Size of Sample')
pymap.ylabel("Fitness Curve")
pymap.legend(loc="best")
pymap.grid()
pymap.savefig("nq_mimic2.png")
pymap.show()

pymap.figure()
pymap.plot(varlist, neural_time, 'o-',label='Time',color='green',dashes=[4, 3])
#pymap.plot(varlist, neural_ftavg, 'o-',label='Fitness',color='red',dashes=[4, 3])
#pymap.plot(varlist, neural_crv, 'o-',label='Fitness Curve',color='blue',dashes=[4, 3])
pymap.tick_params(axis='x', rotation=90)

pymap.title('Knapsack Problems - MIMIC - Time')
pymap.xlabel('Percentage and Size of Sample')
pymap.ylabel("Time")
pymap.legend(loc="best")
pymap.grid()
pymap.savefig("nq_mimic3.png")
pymap.show()