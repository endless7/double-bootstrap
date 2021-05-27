# -*- coding: utf-8 -*-

import numpy as np
import scipy
import sys, copy
import multiprocessing
from scipy import stats
from joblib import Parallel, delayed
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

class DataSet:
    def __init__(self, a_array, aprime_array, nulls = False):
        N, B = aprime_array.shape
        self.N, self.B = N, B
        self.a_array = a_array #a
        self.aprime_array = aprime_array #x (residual bootstrap)
        self.nulls = nulls # true nulls (synthetic only, we do not know which is true null in real data)
        self.a_array_abs = np.abs(self.a_array) 
        self.a_array_abs_sorted = sorted(self.a_array_abs)
        self.aprime_array_abs = np.abs(self.aprime_array) # N x B
        self.aprime_array_abs_sorted = np.array([sorted(self.aprime_array_abs[i,:]) for i in range(N)]) # N x B
        self.pvalues = np.array([self.obtain_pvalue(i, self.a_array_abs[i]) for i in range(N)])
        print(f"pvalues (sorted) = {sorted(self.pvalues)}")
        tmp = [(self.pvalues[i],i) for i in range(N)]  
        self.pvalues_sorted_indices = [i for (p,i) in sorted(tmp)] #zipped
        self.pvalues_sorted = [p for (p,i) in sorted(tmp)]
    def obtain_pvalue(self, i, v):
        pos = np.searchsorted(self.aprime_array_abs_sorted[i,], v)
        B = self.B
        if pos == B:
            return 0.5/(B+1)
        elif pos == self.aprime_array_abs_sorted[i,pos]:
            return 1 - (pos+1)/(B+1)
        else: # (pos-1, pos)
            return 1 - (pos+0.5)/(B+1)

class FDRMethods():
    def __init__(self, delta = 0.05):
        self.delta = delta
        self.thresholds = {}
        self.num_discoveries = {}
        self.FDRs = {}
        print(f"delta = {delta}")
    def FDR_exists(self):
        return len(self.FDRs.keys())>0
    def my_print(self):
        for method in self.thresholds.keys():
            if method in self.FDRs:
                print(f"{method}: threshold = {self.thresholds[method]}, num_discoveries = {self.num_discoveries[method]}, FDR = {self.FDRs[method]}")
            else:
                print(f"{method}: threshold = {self.thresholds[method]}, num_discoveries = {self.num_discoveries[method]}")
    def getResults(self):
        return (self.thresholds, self.num_discoveries, self.FDRs)
    def SingleTesting(self,  dataset): #fixed delta
        name = "single"
        N, B = dataset.N, dataset.B
        delta = self.delta
        for i in range(N):
            threshold = delta 
            if dataset.pvalues_sorted[i] > threshold:
                break 
        num_disc = i
        self.thresholds[name] = 0
        if num_disc > 0:
          self.thresholds[name] = dataset.pvalues_sorted[num_disc-1]
        self.num_discoveries[name] = num_disc
        if dataset.nulls != False:
            if num_disc == 0:
                FDR = 0
            else:
                FDR = sum([dataset.pvalues[j] <= threshold and dataset.nulls[j] for j in range(N)]) / num_disc
            self.FDRs[name] = FDR
        return num_disc
    def BH(self, dataset, use_BY = False):
        if use_BY:
            name = "BY"
        else:
            name = "BH"
        N, B = dataset.N, dataset.B
        delta = self.delta
        if use_BY:
            correction = np.log(N) + 0.5
        else:
            correction = 1
        num_disc, threshold = 0, 0
        for i in range(N):
            threshold_tmp = delta * (i+1) / (N * correction)
            if dataset.pvalues_sorted[i] <= threshold_tmp:
                threshold = dataset.pvalues_sorted[i]
                num_disc = i+1
        print(f"{name} threshold: {threshold}, {num_disc} hypotheses found")
        self.num_discoveries[name] = num_disc
        self.thresholds[name] = threshold
        if dataset.nulls != False:
            if num_disc == 0:
                FDR = 0
            else:
                FDR = sum([dataset.pvalues[j] <= threshold and dataset.nulls[j] for j in range(N)]) / num_disc
            self.FDRs[name] = FDR   
        return num_disc
    def BKY(self, dataset): #BKY2006
        name = "BKY"
        N, B = dataset.N, dataset.B
        delta = self.delta
        beta1, beta2 = delta/(1+delta), delta/(1+delta)
        # Run BH with beta1
        num_disc_fst = 0
        for i in range(N):
            threshold = beta1 * (i+1) / N
            if dataset.pvalues_sorted[i] <= threshold:
                num_disc_fst, threshold = i+1, delta * i / N
        hatN0 = max(N - num_disc_fst, 1) #estimated number of null hypotheses
        print(f"hatN0 = {hatN0}")
        # run BH again with beta2 and hatN0
        num_disc, threshold = 0, 0
        for i in range(N):
            threshold_tmp = beta2 * (i+1) / hatN0
            if dataset.pvalues_sorted[i] <= threshold_tmp:
                num_disc, threshold = (i+1), dataset.pvalues_sorted[i]
        print(f"{name} threshold: {threshold}, {num_disc} hypotheses found")
        self.thresholds[name] = threshold
        self.num_discoveries[name] = num_disc
        if dataset.nulls != False:
            if num_disc == 0:
                FDR = 0
            else:
                FDR = sum([dataset.pvalues[j] <= threshold and dataset.nulls[j] for j in range(N)]) / num_disc
            self.FDRs[name] = FDR   
        return num_disc
    def Storey(self, dataset, t0 = 0.5, auto=False, hatpi0=False):
        if auto:
            name = "Storey-Auto"
        elif type(hatpi0) == int: #Storey with Boostrap-estimated t0
            name = "Storey-Boot"
        else:
            name = "Storey"
        print(f"hatpi0 = {hatpi0}, name={name}")
        N, B = dataset.N, dataset.B
        delta = self.delta
        if self.SingleTesting(dataset) == 0: #to reduce computation, we immediately quit when delta=0.05 find reject no hypotheses
            self.thresholds[name] = 0
            self.num_discoveries[name] = 0
            if dataset.nulls != False:
                self.FDRs[name] = 0   
            return 0
        def hatG(t): #empirical CDF
            return np.searchsorted(dataset.pvalues_sorted, t, side="right")/N
        def my_clip(x):return min(1, max(x, 0))
        # null ratio
        if not hatpi0:
          hatpi0 = my_clip( (1-hatG(t0))/(1.0 - t0) )
        print(f"hatpi0 = {hatpi0}")
        for i in range(N-1, -1, -1):
            p = dataset.pvalues_sorted[i]
            if hatpi0*p/((i+1.)/N) <= delta:
                break
        else:
            p = 0
        threshold = p 
        self.thresholds[name] = threshold
        num_disc = sum([dataset.pvalues[j] <= threshold for j in range(N)])
        self.num_discoveries[name] = num_disc
        if dataset.nulls != False:
            if num_disc == 0:
                FDR = 0
            else:
                FDR = sum([dataset.pvalues[j] <= threshold and dataset.nulls[j] for j in range(N)]) / num_disc
            self.FDRs[name] = FDR   
        print(f"{name} threshold: {threshold}, {num_disc} hypotheses found")
        return num_disc
    def AutoStorey(self, dataset, R2=500): #Storey with bootstrap-estimated t0
        N, B = dataset.N, dataset.B
        delta = self.delta
        t0_cand = [0.05 * i for i in range(1,20)] # 0.05-0.95
        def hatG_arg(pvalues_arg, t): #empirical CDF
            return np.searchsorted(pvalues_arg, t, side="right")/N
        def my_clip(x):return min(1, max(x, 0))
        hatpis = [my_clip( (1. - hatG_arg(dataset.pvalues_sorted, t0))/(1.0 - t0) ) for t0 in t0_cand] 
        hatpi_min = min(hatpis) 
        errors = np.zeros(len(t0_cand))
        for r2 in range(R2):
            pvalues_boost_sorted = np.sort(np.random.choice(dataset.pvalues, N))
            hatpis_boost = [my_clip( (1. - hatG_arg(pvalues_boost_sorted,t0))/(1.0 - t0) ) for t0 in t0_cand]
            for i,t0 in enumerate(t0_cand):
                 errors[i] += (hatpis_boost[i] - hatpi_min)**2
        t0_mse = t0_cand[ np.argmin(errors) ]
        print(f"Auto t0={t0_mse}")
        return self.Storey(dataset, t0 = t0_mse, auto = True)
    def YBBoot(self, dataset, beta = False, rec_num = 15, R2 = 500): #YB1999 Sec 4.1 
        name = "YB-Boot"
        N, B = dataset.N, dataset.B
        delta = self.delta
        if self.SingleTesting(dataset) == 0: 
            self.thresholds[name] = 0
            self.num_discoveries[name] = 0
            if dataset.nulls != False:
                self.FDRs[name] = 0   
            return 0
        if beta == False:
            beta = delta/2.
            deltap = delta/2.
        else:
            deltap = delta
        print(f"beta,delta' = {beta},{deltap}")
        bs = np.random.randint(0, B-1, R2)
        aprime_array_subset = dataset.aprime_array[:,bs]
        pvalue_aprime_subset = np.zeros((N,R2))
        for i in range(N):
            for r2 in range(R2):
                pvalue_aprime_subset[i,r2] = dataset.obtain_pvalue(i, np.abs(aprime_array_subset[i,r2]))
        pvalue_aprime_subset_sorted = np.sort(pvalue_aprime_subset, axis=0)
        def obtainUpperLimit(pvalue_aprime_subset_sorted, p): 
            upper_pos = int( (1-beta)*R2 ) - 1
            Gs = np.zeros(R2)
            for r2 in range(R2):
                Gs[r2] = np.searchsorted(pvalue_aprime_subset_sorted[:,r2], p)
            return np.sort(Gs)[upper_pos]
        def estFDR_YB(x, b): #YB1999 (10)
            r_beta = obtainUpperLimit(pvalue_aprime_subset_sorted, x)
            r = np.searchsorted(dataset.pvalues_sorted, x)
            pvalues_null = [dataset.obtain_pvalue(i, np.abs(dataset.aprime_array_abs[i,b])) for i in range(N)]
            R = np.searchsorted(sorted(pvalues_null), x)
            if R == 0:
                return 0
            elif r > r_beta:
                return R/(R + r - r_beta)
            else:
                return  1
        x_l, x_u = 0.01*deltap, deltap*10
        for rec in range(rec_num): #binary search
            x_cur = (x_l*x_u)**0.5 
            FDRs = np.zeros(R2)
            for r2 in range(R2):
                b = np.random.randint(0, B-1)
                FDRs[r2] = estFDR_YB(x_cur, b)
            mean_FDR = np.mean(FDRs)
            if mean_FDR <= deltap:
                x_l = x_cur
            else:
                x_u = x_cur
        threshold = (x_l*x_u)**0.5 
        num_disc = sum([dataset.pvalues[j] < threshold for j in range(N)])
        if num_disc == 0:
          self.thresholds[name] = 0
        else:
          self.thresholds[name] = dataset.pvalues_sorted[num_disc-1]
        self.num_discoveries[name] = num_disc
        if dataset.nulls != False:
            if num_disc == 0:
                FDR = 0
            else:
                FDR = sum([dataset.pvalues[j] <= threshold and dataset.nulls[j] for j in range(N)]) / num_disc
            self.FDRs[name] = FDR   
        print(f"YB-Boot threshold: {threshold}, {num_disc} hypotheses found")
        return num_disc
    def ProposedMethod(self, dataset, R1 = 10, R2 = 1000, Bayes = False, Double=True):
        if not Bayes:
            name = "Proposed"
        else:
            name = "Proposed-Bayes"
        if not Double:
            name = name + "-Agg"
        N, B = dataset.N, dataset.B
        delta = self.delta
        if self.SingleTesting(dataset) == 0:
            self.thresholds[name] = 0
            self.num_discoveries[name] = 0
            if dataset.nulls != False:
                self.FDRs[name] = 0   
            return 0
        def hatG(pvalues_sorted_arg, t):
            return np.searchsorted(pvalues_sorted_arg, t, side="right")/N
        def estFDR(zipped_pvalues_sorted, t_coef): 
            p_lb, p_ub = delta/(N*t_coef), 1.0
            pvalues_sorted = np.array([p for (p,n) in zipped_pvalues_sorted])
            for i_ind in range(N-1, -1, -1):
                p = pvalues_sorted[i_ind]
                if p*t_coef/((i_ind+1.)/N) <= delta:
                    break
            else:
                p = 0
            p_decision = p 
            # obtain FDR
            rej_true, rej_false = 0, 0
            for (p,n) in zipped_pvalues_sorted:
                if p <= p_decision:
                    if n == True:
                        rej_false += 1
                    else:
                        rej_true += 1
            if rej_true + rej_false == 0:
                FDR = 0
            else:
                if Double:
                  FDR = 2 * rej_false / (rej_true + rej_false) # doubling (DDB)
                else:
                  FDR = rej_false / (rej_true + rej_false) # DDBA
            return FDR
        def obtain_estimated_mu(a_array_arg, aprime_array_arg):
            a_array_tmp_true_list = []
            N0_list = []
            for r1 in range(R1):
                b = np.random.randint(0, B-1)
                null_a_samples = aprime_array_arg[:,b]
                a_array_tmp_true = np.array(a_array_arg)
                a_array_tmp_true += null_a_samples
                a_array_nullest = np.zeros(len(a_array_tmp_true)) #whether i-th hyp is null or not
                num_null = 0
                for i in range(len(a_array_tmp_true)): #two-sided case nullfy |a_i| <= |x_i|
                    if np.abs(a_array_arg[i]) <= np.abs(null_a_samples[i]):
                        a_array_tmp_true[i] = 0
                        num_null +=1 
                a_array_tmp_true_abs_sorted = sorted([(np.abs(a_array_tmp_true[i]), i) for i in range(len(a_array_tmp_true))])
                N0_list.append(num_null)
                a_array_tmp_true_list.append( a_array_tmp_true )
            return N0_list, a_array_tmp_true_list
        def obtain_optimal_threshold(a_array_arg, aprime_array_arg, bs, rec_num = 15): #rec_num = # of binary search
            t_coef_l, t_coef_u = 0.1, 100 
            for rec in range(rec_num):
                t_coef = (t_coef_l*t_coef_u)**0.5 #geom mean
                FDRs = np.zeros(R2)
                for r in range(R2): #R2 = W in paper
                    # sampling null 
                    b = bs[r]
                    null_a_samples = aprime_array_arg[:,b]
                    null_a_samples_abs = np.abs( aprime_array_arg[:,b] )
                    a_array_tmp = copy.deepcopy(a_array_arg)
                    a_array_tmp += null_a_samples
                    a_array_tmp_abs = np.abs(a_array_tmp)
                    a_array_tmp_abs_sorted = sorted( a_array_tmp_abs )
                    zipped_pvalues = [(dataset.obtain_pvalue(i, a_array_tmp_abs[i]), a_array_arg[i] == 0) for i in range(N)]
                    zipped_pvalues_sorted = sorted(zipped_pvalues)
                    FDR = estFDR(zipped_pvalues_sorted, t_coef)
                    FDRs[r] = FDR
                mean_FDR = np.mean(FDRs)
                if mean_FDR > delta: #binary search: halved region
                    t_coef_l = t_coef
                else:
                    t_coef_u = t_coef
            return t_coef
        def obtain_bayes_optimal_threshold(a_array_arg, aprime_array_arg, rec_num = 15): #rec_num = # of binary search. Note this Bayesian version is not used in the paper
            t_coef_l, t_coef_u = 0.1, 100 
            N0_list, a_array_tmp_true_list = obtain_estimated_mu(a_array_arg, aprime_array_arg)
            for rec in range(rec_num):
                t_coef = (t_coef_l*t_coef_u)**0.5 #geom mean
                FDRs = np.zeros(R1*R2)
                for r1 in range(R1): #R1 = V in paper
                    a_array_tmp_true = a_array_tmp_true_list[r1]
                    zipped_pvalues_sorted_list = []
                    for r2 in range(R2): #R2 = W in paper
                        # sampling null
                        b = np.random.randint(0, B-1)
                        null_a_samples = aprime_array_arg[:,b]
                        null_a_samples_abs = np.abs( aprime_array_arg[:,b] )
                        a_array_tmp = copy.deepcopy(a_array_tmp_true)
                        a_array_tmp += null_a_samples
                        a_array_tmp_abs = np.abs(a_array_tmp)
                        a_array_tmp_abs_sorted = sorted( a_array_tmp_abs )
                        zipped_pvalues = [(dataset.obtain_pvalue(i, a_array_tmp_abs[i]), a_array_tmp_true[i] == 0) for i in range(N)]
                        zipped_pvalues_sorted = sorted(zipped_pvalues)
                        zipped_pvalues_sorted_list.append(zipped_pvalues_sorted)
                    FDRs_batch  = Parallel(n_jobs=-1)( [delayed(estFDR)(zipped_pvalues_sorted_list[r2], t_coef) for r2 in range(R2)] )
                    for r2 in range(R2):
                        FDRs[r1*R2 + r2] = FDRs_batch[r2]
                mean_FDR = np.mean(FDRs)
                if mean_FDR > delta: #halving region
                    t_coef_l = t_coef
                else:
                    t_coef_u = t_coef
            return t_coef
            #####
        # 区分 贝叶斯
        ####
        if not Bayes:
            N0_list, a_array_tmp_true_list = obtain_estimated_mu(dataset.a_array, dataset.aprime_array)

            print(f"starting parallel computation: # of Cores = {multiprocessing.cpu_count()}");sys.stdout.flush()

            bs_list = [np.random.randint(0, B-1, size=R2) for r1 in range(R1)]
            t_coefs = Parallel(n_jobs=-1)( [delayed(obtain_optimal_threshold)(a_array_tmp_true_list[r1], dataset.aprime_array, bs_list[r1]) for r1 in range(R1)] )
            t_coef_final = max(t_coefs) #worst-case of t_coefs
            print(f"t_coef_final = {t_coef_final}")
        else: # Bayes
            t_coef_final = obtain_bayes_optimal_threshold(dataset.a_array, dataset.aprime_array)

        pvalues_sorted = sorted([dataset.obtain_pvalue(i, dataset.a_array_abs[i]) for i in range(N)])
        p_lb, p_ub = delta/(N*t_coef_final), 1.0
        for i_ind in range(N-1, -1, -1):
            p = pvalues_sorted[i_ind]
            #if hatpi0*p/max(0.000001, hatG(p)) <= delta:
            if p*t_coef_final/((i_ind+1.)/N) <= delta:
                break
        else:
            p = 0
        threshold = p 
        num_disc = sum([dataset.pvalues[j] <= threshold for j in range(N)])
        if num_disc == 0:
          self.thresholds[name] = 0
        else:
          self.thresholds[name] = dataset.pvalues_sorted[num_disc-1]
        self.num_discoveries[name] = num_disc
        print(f"{name} threshold: {threshold}, {num_disc} hypotheses found")
        if dataset.nulls != False:
            if num_disc == 0:
                FDR = 0
            else:
                FDR = sum([dataset.pvalues[j] <= threshold and dataset.nulls[j] for j in range(N)]) / num_disc
            self.FDRs[name] = FDR   
        return num_disc

# sampling from student-t dist
def MyTDist(N=50, T=100, corrcoef = 0.5, subset = False):
    mean = np.zeros(N)
    cov = np.eye(N)
    if corrcoef >= 0:
      for i in range(N):
          for j in range(N):
              if i!=j:
                  if not subset:
                    cov[i,j]=corrcoef
                  elif (i in subset) and (j in subset):
                    cov[i,j]=corrcoef
    else: #negative corr
      for i in range(N):
          for j in range(N):
              if i!=j:
                  cov[i,j] = corrcoef/(N-1)
    X = np.random.multivariate_normal(mean, cov, T).T
    theta = np.mean(X, axis=1)
    std = np.std(X, axis=1)
    Tstat = (theta / std) * np.sqrt(T) 
    return Tstat

#synthetic dataset factory class
class DataSetMaker:
    def __init__(self):
        pass
    def createDependentHypotheses(self, N=50, B=10000, null_ratio = 1.0, corrcoef = 0.5, mu_base = 0.2, random_subset=False):
        null_num = int(N * null_ratio)
        # 0 - null_num-1までがnull
        nulls = [i<null_num for i in range(N)]
        mu = np.zeros(N)
        for i in range(null_num, N):
            mu[i] = mu_base * 2 * np.random.uniform(0, 1) # alternative hyps
        print(f"corrcoef={corrcoef}")
        if random_subset:
          K = np.random.randint(1, N) 
          S = set(np.random.choice(N, K, replace = False))
          a_array = MyTDist(N=N, corrcoef=corrcoef, subset = S)
        else:
          a_array = MyTDist(N=N, corrcoef=corrcoef, subset = False) 
        a_array += mu
        print(f"a_array = {a_array}")
        aprime_array = np.zeros((N,B))
        for b in range(B):
            aprime_array[:,b] = MyTDist(N=N, corrcoef=corrcoef)
        return DataSet(a_array, aprime_array, nulls)

def main(scenario = 3, run_num = 10, delta = 0.05, mu_base = 0.5, R1=20, R2=500, df_true = False, df_bootstrap = False):
    np.set_printoptions(precision=4)
    print(f"mu_base = {mu_base}")

    dm = DataSetMaker()
    results_list = [] # (self.thresholds, self.num_discoveries, self.FDRs)
    for run in range(run_num):
        print(f"scenario = {scenario}")
        
        if scenario == 0:
            N = len( list(df_true['V_X']) )
            print(f"N={N}")
            if N > 100:
              vx_sub = np.random.choice(list(df_true['V_X']), N//2, replace = False)
              print(f"vx_sub = {vx_sub}")
              df_true_sub = df_true[df_true['V_X'].isin(vx_sub)]
              a_array = np.array(df_true_sub['alpha'])
              print(f"a_array = {a_array}")
              aprime_array = np.array(df_bootstrap[ vx_sub ]).T
            else:
              a_array = np.array(df_true['alpha'])
              aprime_array = np.array(df_bootstrap[ list(df_true['V_X']) ]).T
            dataset = DataSet(a_array, aprime_array)
        elif scenario == 1: #
            dataset = dm.createDependentHypotheses(N=50, null_ratio = 0.5, corrcoef=0.0, mu_base = mu_base) 
        elif scenario == 2: #
            dataset = dm.createDependentHypotheses(N=50, null_ratio = 0.5, corrcoef=0.5, mu_base = mu_base)
        elif scenario == 3: #
            dataset = dm.createDependentHypotheses(N=50, null_ratio = 0.5, corrcoef=0.9, mu_base = mu_base)
        elif scenario == 4: #
            dataset = dm.createDependentHypotheses(N=50, null_ratio = 1.0, corrcoef=0.0, mu_base = mu_base) 
        elif scenario == 5: #
            dataset = dm.createDependentHypotheses(N=50, null_ratio = 1.0, corrcoef=0.5, mu_base = mu_base)
        elif scenario == 6: #
            dataset = dm.createDependentHypotheses(N=50, null_ratio = 1.0, corrcoef=0.9, mu_base = mu_base)
        elif scenario == 7: #
            dataset = dm.createDependentHypotheses(N=50, null_ratio = 0.25, corrcoef=0.0, mu_base = mu_base)
        elif scenario == 8: #
            dataset = dm.createDependentHypotheses(N=50, null_ratio = 0.25, corrcoef=0.5, mu_base = mu_base)
        elif scenario == 9: #
            dataset = dm.createDependentHypotheses(N=50, null_ratio = 0.25, corrcoef=0.9, mu_base = mu_base)
        elif scenario == 10: #not used
            dataset = dm.createDependentHypotheses(N=50, null_ratio = 1.0, corrcoef=-0.5, mu_base = mu_base) 
        else:
            print("scenario not found")
            sys.exit()

        methods = FDRMethods(delta)
        Single_i = methods.SingleTesting(dataset)
        BH_i = methods.BH(dataset, use_BY=False)
        BY_i = methods.BH(dataset, use_BY=True)
        BKY_i = methods.BKY(dataset)
        Storey_i = methods.Storey(dataset, 0.5)
        AutoStorey_i = methods.AutoStorey(dataset, R2=R2)
        YBBoot_i = methods.YBBoot(dataset, R2=R2) 
        # 双重检验  core
        Resampling_i = methods.ProposedMethod(dataset, R1=R1, R2=R2, Bayes=False)
        Resampling_i = methods.ProposedMethod(dataset, R1=R1, R2=R2, Bayes=False, Double = False)
        results_list.append( methods.getResults() )
        print(f"run {run+1} results:")
        methods.my_print()
        if run_num < 10 or ((run+1) % (run_num // 10)) == 0:
          print(f"scenario{scenario}-results at run={run+1}")
          if methods.FDR_exists():
            print(f"method,mean_threshold,mean_numdisc,stddev_numdisc,mean_FDR,stddev_FDR")
          else:
            print(f"method,mean_threshold,mean_numdisc,stddev_numdisc")
          for method in results_list[0][0].keys():
            thresholds, num_discoveries, fdrs = [], [], []
            for results in results_list:
              thresholds.append(results[0][method])
              num_discoveries.append(results[1][method])
              if methods.FDR_exists():
                fdrs.append(results[2][method])
            if methods.FDR_exists():
              print(f"{method},{np.mean(thresholds)},{np.mean(num_discoveries)},{np.std(num_discoveries)},{np.mean(fdrs)},{np.std(fdrs)}")
            else:
              print(f"{method},{np.mean(thresholds)},{np.mean(num_discoveries)},{np.std(num_discoveries)}")
    print(f"scenario = {scenario}, simulation end")

def plot_realdata (df_true, df_bootstrap):
    tvalues = []
    columns = df_bootstrap.columns
    factors = list(columns)[1:] 
    for factor in factors:
        tvalues = tvalues + list(df_bootstrap[factor])
    print(f"factors = {factors}")
    s2 = np.random.standard_t(240-4, 10000*len(factors))

    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(1,1,1)
    bins = np.histogram(np.hstack((tvalues,s2)), bins=50)[1]
    ax.hist(tvalues, bins=bins, label="x (null resampling)", color="blue", alpha=0.3)
    ax.hist(s2, bins=bins, label="true of t-dist", color="green", alpha=0.3)
    print(f"len(tvalues) = {len(tvalues)}")
    print(f"len(s2) = {len(s2)}")
    plt.legend(loc='upper left')
    fig.savefig('tvalue_null.pdf', dpi=fig.dpi, bbox_inches='tight')
    plt.clf()

    np.quantile(tvalues, 0.95)
    print(stats.describe(s2)) # student-t
    print(stats.describe(tvalues))

    tvalues_sml = np.random.choice(tvalues, 5000)
    N = len(tvalues_sml)
    tvalues_x = sorted(np.abs(tvalues_sml))
    tvalues_y = [i/N for i in range(N)]
    fig = plt.figure(figsize=(4,3))
    plt.plot(tvalues_x, tvalues_y, label="x (null resampling)")

    alphas = np.abs(df_true['alpha'])
    N = len(alphas)
    alphas_x = sorted(alphas)
    alphas_y = [i/N for i in range(N)]
    plt.plot(alphas_x, alphas_y, label="a")

    plt.xlabel("tvalue")
    plt.ylabel("tvalue dist")
    plt.legend(loc='lower right')
    fig.savefig('tvalue_cdf.pdf', dpi=fig.dpi, bbox_inches='tight')
    plt.show()


import optparse

parser = optparse.OptionParser()
parser.add_option('-s', action="store", dest="s", default = 0, type="int")
parser.add_option('-r', action="store", dest="r", default = 10, type="int")
parser.add_option('-v', action="store", dest="v", default = 20, type="int")
parser.add_option('-w', action="store", dest="w", default = 500, type="int")
parser.add_option('-m', action="store", dest="m", default = 1.0, type="float")
parser.add_option('-d', action="store", dest="d", default = 0.05, type="float")
parser.add_option('-a', action="store", dest="a_file", type="string")
parser.add_option('-x', action="store", dest="x_file", type="string")
parser.add_option('-c', action="store", default = -1, dest="seed", type="int")
options, remainder = parser.parse_args()
scenario = options.s
run_num = options.r
mu_base = options.m
delta = options.d
V = options.v
W = options.w
a_file = options.a_file
x_file = options.x_file
seed = options.seed
if seed != -1:
    np.random.seed(seed)
print(f"synth scenario = {scenario} mu={mu_base} V={V}, W={W}")


path = "./"
df_true = pd.read_csv(path + "china_tvalues.csv", sep=",")
df_bootstrap = pd.read_csv(path + "china_tvalues_null.csv", sep=",")
#df_true = pd.read_csv(a_file)
#df_bootstrap = pd.read_csv(x_file)
plot_realdata(df_true, df_bootstrap)

main(scenario = scenario, run_num = run_num, delta = delta, mu_base = mu_base, R1=V, R2=W, df_true=df_true, df_bootstrap=df_bootstrap)