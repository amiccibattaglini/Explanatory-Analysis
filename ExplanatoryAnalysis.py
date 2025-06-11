import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import (t, norm)


class ExplanatoryPlots:
    def __init__(self, 
                 data: np.array, 
                 distribution: str, 
                 type: str,
                 plot: bool = True,
                 df: int = None):
        
        """ 
        This class is built for explanatory data analysis
        The class takes as inputs:
        - data: the variable that needs to be analyzed
        - the type of distribution (norm for normal distribution or t for t-student)
        - type: this is the type of plot the is needed (Halff-plot or QQ-plot)
        - plot: default = True
        - df: the degrees of freedom for t-student
        """
        self.data = data
        self.distribution = distribution
        self.type = type
        self.plot = plot
        self.df = df
        #from QQPlot_Data
        self.order_statistics = None
        self.unconditional_mean = None
        self.unconditional_std = None
        self.parametric_quantiles = None
        self.Plot_Data()
    
    def Plot_Data(self):
        #preliminary computations
        data = self.data
        #mean and std computation
        self.unconditional_mean = np.mean(data)
        self.unconditional_std = np.std(data)
        self.order_statistics = np.sort(data)
        #getting the type of distribution
        dist = self.get_distribution()
        #getting the expected quantiles
        expected_quantiles = self.get_expected_quantiles()
        if self.distribution == "t":
            if self.df == None:
                raise TypeError("For t-student Distribution degrees  of freedom are needed")
            self.parametric_quantiles = dist.ppf(expected_quantiles, 
                                            df = self.df, 
                                            loc = self.unconditional_mean, 
                                            scale = self.unconditional_std)
        if self.distribution == "norm":
            self.parametric_quantiles = dist.ppf(expected_quantiles, 
                                            loc = self.unconditional_mean, 
                                            scale = self.unconditional_std)
    def QQPlot_Plot(self):
        parametric_quantiles = self.parametric_quantiles
        order_statistics = self.order_statistics
        lr = LinearRegression()
        results = lr.fit(parametric_quantiles.reshape(-1, 1), order_statistics.reshape(-1, 1))
        beta, intercept = results.coef_, results.intercept_
        if self.plot:
            plt.figure(figsize = (10, 6))
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
        unconditional_mean = self.unconditional_mean
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