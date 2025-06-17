import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.special import gammaln
from scipy.stats import (t, norm)

class MLE:
    def __init__(self, 
                 data, 
                 distribution):
        """ 
        This class is built for computing the Maximum Likelihood parameters of the give distribution
        The class takes as inputs:
        - data: the variable that needs to be analyzed
        - the type of distribution (norm for normal distribution or t for t-student)
        """
        self.data = data
        self.distribution = distribution
        #from t_Expectation_Maximization
        self.means = {}
        self.stds = {}
        self.likelihoods = {}
        #from get_parameters
        self.mle_mean = None
        self.mle_std = None
        self.mle_df = None
        self.get_parameters()

    def get_parameters(self):
        data = self.data
        unc_mean = np.mean(data)
        unc_std = np.std(data)
        if self.distribution == "t":
            self.mle_mean, self.mle_std, self.mle_df = self.t_Expectation_Maximization(unc_mean, unc_std)
            return self.mle_mean, self.mle_std, self.mle_df
        if self.distribution == "norm":
            self.mle_mean = unc_mean
            self.mle_std = unc_std
            return self.mle_mean, self.mle_std
        
    def t_Expectation_Maximization(self, 
                                   unc_mean:float, 
                                   unc_std:float, 
                                   max_iter: int = 1000, 
                                   error:float = 1e-4):
        """
        This function computes the EM algo for t-distribution
        the inputs are:
        - unc_mean: this is the starting value of the iteration procedure for the mean
        - unc_std: this is the starting value of the iteration procedure for the standard deviation
        - max_iter: set to default equal to 1000
        - error: set to default equal to 0.0001
        """
        data = self.data
        grid_df = self.get_degrees_freedom(ndegrees_of_freedom = 10)
        for df in grid_df:
            log_likelihood = -np.inf
            i = 0
            list_likelihoods = []
            while i < max_iter:
                delta = (data - unc_mean)**2/unc_std**2
                weigths = (df + 1)/(df + delta)
                mu_new = np.sum(weigths * data)/np.sum(weigths)
                unc_var_new = np.mean(weigths * (data - mu_new)**2)
                log_likelihood_i = self.t_log_likelihood(loc = mu_new, 
                                                         scale = np.sqrt(unc_var_new), 
                                                         df = df
                                                        )
                if np.abs(log_likelihood_i - log_likelihood) < error:
                    break
                #appending the means, stds and log-likelihood
                self.means[df] = mu_new
                self.stds[df] = np.sqrt(unc_var_new)
                list_likelihoods.append(log_likelihood_i)
                #updating the measures
                unc_mean = mu_new
                unc_std = np.sqrt(unc_var_new)
                log_likelihood = log_likelihood_i
                i += 1
            self.likelihoods[df] = list_likelihoods
        best_df = max(self.likelihoods, key = self.likelihoods.get)
        return self.means[best_df], self.stds[best_df], best_df

    def t_log_likelihood(self, 
                        loc: float, 
                        scale: float, 
                        df:int):
        data = self.data
        delta = (data -loc)/scale
        log_pdf = (gammaln((df+1)/2)
                - gammaln(df/2)-1/2 * np.log(df * np.pi) 
                - np.log(scale) 
                - (df + 1)/2 * np.log(1 + delta ** 2/df)
                )   
        return np.sum(log_pdf)

    def get_degrees_freedom(self, 
                            ndegrees_of_freedom:int, 
                            starting_point: int = 2, 
                            ending_point: int = 101, 
                            focus:int = 10):
        grid = np.arange(starting_point, ending_point)
        weigths = []

        for index, i in enumerate(grid):
            if index - focus <= 0:
                weigths.append(1)
            else:
                weigths.append(np.exp(-(index - focus)))

        weigths /= np.sum(weigths)
        np.random.seed(42)
        selected_df = np.random.choice(grid, ndegrees_of_freedom, replace = False,p = weigths)
        return selected_df
    
'-------------------------------------------------------------------------------------------------------------------------'

class ExplanatoryPlots:
    def __init__(self, 
                 data: np.array, 
                 distribution: str, 
                 type: str,
                 plot: bool = True):
        
        """ 
        This class is built for explanatory data analysis
        The class takes as inputs:
        - data: the variable that needs to be analyzed
        - the type of distribution (norm for normal distribution or t for t-student)
        - type: this is the type of plot the is needed (Halff-plot or QQ-plot)
        - plot: default = True
        """
        self.data = data
        self.distribution = distribution
        self.type = type
        self.plot = plot
        #from class MLE
        self.mle = MLE(data = self.data, 
                       distribution = self.distribution)
        #from QQPlot_Data
        self.order_statistics = None
        self.mle_mean = None
        self.mle_std = None
        self.mle_df = None
        self.parametric_quantiles = None
        self.Plot_Data()
        
    def Plot_Data(self):
        #preliminary computations
        data = self.data
        #mean and std computation
        self.order_statistics = np.sort(data)
        #getting the type of distribution
        dist = self.get_distribution()
        #getting the expected quantiles
        expected_quantiles = self.get_expected_quantiles()
        if self.distribution == "t":
            self.mle_mean, self.mle_std, self.mle_df = self.mle.get_parameters()
            self.parametric_quantiles = dist.ppf(expected_quantiles, 
                                            df = self.mle_df, 
                                            loc = self.mle_mean, 
                                            scale = self.mle_std)
        if self.distribution == "norm":
            self.mle_mean, self.mle_std = self.mle.get_parameters()
            self.parametric_quantiles = dist.ppf(expected_quantiles, 
                                            loc = self.mle_mean, 
                                            scale = self.mle_std)
    def QQPlot_Plot(self):
        from sklearn.linear_model import LinearRegression
        parametric_quantiles = self.parametric_quantiles
        order_statistics = self.order_statistics
        lr = LinearRegression()
        results = lr.fit(parametric_quantiles.reshape(-1, 1), order_statistics.reshape(-1, 1))
        beta, intercept = results.coef_, results.intercept_
        if self.plot:
            plt.scatter(parametric_quantiles,
                        order_statistics, 
                        s=30,               
                        facecolors='none',  
                        edgecolors='blue', 
                        marker='o')
            
            plt.plot(parametric_quantiles,
                     intercept + beta * parametric_quantiles.reshape(-1, 1), 
                     color = "black", 
                     linestyle = "--")

            plt.xlabel("Parametric Quantile")
            plt.ylabel("Sample Data")
            plt.title("QQ-plot")
            plt.grid()
            plt.show()
        if self.type == "Half-plot":
            raise TypeError("You have selected QQ-Plot while using Expected quantile from Half Normal Plot")
    
    def HalfNormal_Plot(self, nshow: int = 10):
        parametric_quantiles = self.parametric_quantiles
        unconditional_mean = self.mle_mean
        data = self.data
        deviations = np.sort(np.abs(data - unconditional_mean))
        index = np.argsort(np.abs(data - unconditional_mean))
        if self.plot:
            plt.figure(figsize = (10, 6))
            plt.scatter(parametric_quantiles, 
                        deviations, 
                        facecolors = "none", 
                        edgecolors = "blue")

            for i in range(1, nshow):
                plt.text(parametric_quantiles[-i] + 1e-3, 
                        deviations[-i],
                        f"{index[-i]}", 
                        fontfamily = "Times New Roman", 
                        fontsize = 12, 
                        ha = "left", 
                        va = "center")
            plt.grid()
            plt.title("Half-Plot")
            plt.show()
        if self.type == "QQ-plot":
            raise TypeError("You have selected Half Normal Plot while using Expected quantile from QQ-Plot")
        
    def get_distribution(self):
        from scipy.stats import (t, norm)
        distribution = self.distribution
        if distribution == "t":
            return t
        if distribution == "norm":
            return norm
        else:
            raise TypeError("For now just the Normal distribution and the t-student distribution are supported")
        
    def get_expected_quantiles(self):
        n = len(self.data)
        if self.type == "QQ-plot":
            return (np.linspace(1, n, n) - 0.5)/n
        elif self.type == "Half-plot":
            return (n + np.linspace(1, n, n))/(2*n + 1)
        else:
            raise TypeError("The only type of plot accepted are QQ-plot and Half-plot")