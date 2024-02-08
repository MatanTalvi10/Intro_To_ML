#################################
# Your name: Matan Talvi
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals
import math

class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        X = np.random.uniform(0,1,m)
        X.sort()
        Y = np.zeros((m,), dtype=int)
        for i in range(m):
            if(X[i]<=0.2 or (X[i]<=0.6 and X[i]>=0.4) or X[i]>=0.8):
                y = np.random.choice([0,1],size = 1,p=[0.2, 0.8])
            else:
                y = np.random.choice([0,1],size = 1,p=[0.9, 0.1])
            Y[i] = y
        res = np.column_stack((X, Y))
        return res

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        M = [m for m in range(m_first, m_last + 1, step)]
        true_error_avg = []
        empirical_error_avg = []
        for m in M:
            true_error_m = np.zeros(T,dtype=float) 
            empirical_error_m = np.zeros(T,dtype=float ) 
            for i in range(T):
                samples = self.sample_from_D(m)
                intervals_l, error_cnt = intervals.find_best_interval(samples[:,0], samples[:,1], k)
                true_error_m[i] = self.calc_true_error(intervals_l)
                empirical_error_m[i] = error_cnt/m
            true_error_avg.append(true_error_m.mean())
            empirical_error_avg.append(empirical_error_m.mean())
        res = np.column_stack((empirical_error_avg, true_error_avg))
        plt.plot(M,empirical_error_avg,label = "Emprical errors")
        plt.plot(M,true_error_avg,label = "True errors")
        plt.legend()
        plt.title("Q1.b")
        plt.show()
        return res




    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        true_error = []
        empirical_error = []
        K = [k for k in range(k_first, k_last + 1, step)]
        samples = self.sample_from_D(m)
        for k in K:
            intervals_l, error_cnt = intervals.find_best_interval(samples[:,0], samples[:,1], k)
            true_error.append(self.calc_true_error(intervals_l))
            empirical_error.append(error_cnt/m)
        min_index = np.argmin(empirical_error)
        plt.plot(K,empirical_error,label = "Emprical errors")
        plt.plot(K,true_error,label = "True errors")
        plt.legend()
        plt.title("Q1.c")
        plt.show()
        return min_index*step + k_first
    

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        true_error = []
        empirical_error = []
        penalties = []
        sum_empirical_error_penalty = []
        K = [k for k in range(k_first, k_last + 1, step)]
        samples = self.sample_from_D(m)
        for k in K:
            intervals_l, error_cnt = intervals.find_best_interval(samples[:,0], samples[:,1], k)
            true_error.append(self.calc_true_error(intervals_l))
            ee = error_cnt/m
            empirical_error.append(ee)
            pen = self.penalty(k,m)
            penalties.append(pen)
            sum_empirical_error_penalty.append(ee+pen)
        min_index = np.argmin(sum_empirical_error_penalty)
        plt.plot(K,empirical_error,label = "Emprical errors")
        plt.plot(K,true_error,label = "True errors")
        plt.plot(K,penalties,label = "penalties")
        plt.plot(K,sum_empirical_error_penalty,label = "Empirical errors + Penalties")
        plt.legend()
        plt.title("Q1.d")
        plt.show()
        return min_index*step + k_first

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        samples = self.sample_from_D(m)
        np.random.shuffle(samples)
        s1 = samples[:int(m * 0.8)]
        s2 = samples[int(m * 0.8):]
        s1 = s1[s1[:, 0].argsort()]
        best_h_list = [intervals.find_best_interval(s1[:, 0], s1[:, 1], k)[0] for k in range(1, 11)]
        empirical_errors_s2 = np.array([self.calculate_empirical_error(s2, h) for h in best_h_list])
        tmp = np.argmin(empirical_errors_s2)
        print(best_h_list[tmp])
        print(tmp+1)
        return tmp + 1

    #################################
    # Place for additional methods

    def calc_intersection(self,intervals_list,L,R):
        """Returns the intersection of the intervals in the given list with [L, R]"""
        total = 0
        for interval in intervals_list:
            a,b = interval[0],interval[1]
            Li = max(L,a)
            Ri = min(R,b)
            if(Li < Ri):
                total += (Ri-Li)
        return total


    def zero_one_loss(self, intervals_list, x, y):
        """" Calculates if delta(h(x),y)) is 0 or 1"""
        h_x = 0
        for interval in intervals_list:
            if interval[0] <= x <= interval[1]:
                h_x = 1
                break        
        if (h_x and y == 1) or (not h_x and y == 0):
            return 0
        return 1

    def calculate_empirical_error(self, sample, intervals_list):
        """Calculates the emprical error given h as the intervals list and samples in the form
           of a two dimensional np.array that contains samples drawn from the distribution P"""
        return sum([self.zero_one_loss(intervals_list, x, y) for x, y in sample]) / len(sample)    

if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500)
