import seaborn as sns

class CausalFunction():
    #initialize the class
    def __init__(self, ATE, SE, teststyle, samples, estimates, data, method, model, propensity_score_model, n_covars):
        self.ATE = ATE
        self.SE = SE
        self.teststyle = teststyle
        self.samples = samples
        self.estimates = estimates
        self.data = data
        self.method = method
        self.model = model
        self.propensity_score_model = propensity_score_model
        self.n_covars = n_covars

    def CI(self, alpha=0.05):
        """
        Calculate the confidence interval for the average treatment effect, depending on whether one uses the Naive SE, Permutation test, or Bootstrap SE.
    
        Parameters:
        alpha (float): The significance level. Default is 0.05.

        Returns:
        tuple: A tuple containing the lower and upper bounds of the confidence interval.
        """
        #calculate the confidence interval using the appropriate method
        if self.teststyle == 'Naive' or self.teststyle == 'Permutation':
            import scipy.stats as s
            z = s.norm.ppf(1 - alpha / 2)
            return (self.ATE - z*self.SE, self.ATE + z*self.SE)
        else:
            import numpy as np
            return (np.percentile(self.estimates, 100*alpha/2), np.percentile(self.estimates, 100*(1-alpha/2)))
    
    def test(self):
        """
        Perform a hypothesis test for the average treatment effect. Depending on if nonparametric is active, performs permutation test

        Returns:
        dict: A dictionary containing the test statistic and p-value
        """
        if self.teststyle == 'Naive':
            import scipy.stats as s
            import numpy as np
            t = self.ATE / self.SE
            t = np.float64(t)
            p = 2 * (1 - s.norm.cdf(np.abs(t)))
        
        elif self.teststyle == 'Permutation' or self.teststyle == 'Bootstrap':
            import numpy as np
            # tstar = list()
            # for ATE in self.estimates:
            #     tstar.append(ATE / np.std(self.estimates))
            #     p =  np.sum(np.abs(tstar) >= np.abs(t)) / len(tstar)

            # permutation test
            ATEs = list()

            #calculate ATE for each sample
            for i in range(1000):
                #take resamples (permuted) T from samples
                old_T = self.data[:, -2]
                new_T = np.random.permutation(old_T)
                #take X and y from the data
                X = self.data[:, :-2]
                y = self.data[:, -1]
                if self.method == 'RA':
                    pass
                elif self.propensity_score_model == 'logit':
                    from statsmodels.discrete.discrete_model import Logit
                    from statsmodels.tools import add_constant
                    propensity = Logit(new_T, X).fit().predict().reshape(-1,1)
                    #correct for values of 0
                    for j in range(len(propensity)):
                        if propensity[j] == 0:
                            propensity[j] = 1e-5
                        elif propensity[j] == 1:
                            propensity[j] = 0.99999
                elif self.propensity_score_model == 'probit':
                    from statsmodels.discrete.discrete_model import Probit
                    from statsmodels.tools import add_constant
                    propensity = Probit(new_T, X).fit().predict().reshape(-1,1)
                    #correct for values of 0
                    for j in range(len(propensity)):
                        if propensity[j] == 0:
                            propensity[j] = 1e-5
                        elif propensity[j] == 1:
                            propensity[j] = 0.99999
                elif self.propensity_score_model == 'SVC':
                    from sklearn.svm import SVC
                    propensity = SVC(probability=True).fit(X, new_T).predict_proba(X)[:,1].reshape(-1,1)
                    # correct for values of 0
                    for j in range(len(propensity)):
                        if propensity[j] == 0:
                            propensity[j] = 1e-5
                        elif propensity[j] == 1:
                            propensity[j] = 0.99999
                elif self.propensity_score_model == 'GAM':
                    from pygam import GAM, s
                    propensity = GAM(s(0, spline_order=3, n_splines=6)).fit(X, new_T).predict(X).reshape(-1,1)
                    #correct for values of 0
                    for j in range(len(propensity)):
                        if propensity[j] == 0:
                            propensity[j] = 1e-5
                        elif propensity[j] == 1:
                            propensity[j] = 0.99999
                elif self.propensity_score_model == 'forest':
                    from sklearn.ensemble import RandomForestRegressor
                    propensity = RandomForestRegressor().fit(X, new_T).predict(X).reshape(-1,1)
                    #correct for values of 0
                    for j in range(len(propensity)):
                        if propensity[j] == 0:
                            propensity[j] = 1e-5
                        elif propensity[j] == 1:
                            propensity[j] = 0.99999

                import numpy as np


                #stack the covariates and treatment into exog
                exog = np.append(np.concatenate((X,new_T.reshape(-1,1)), axis = 1),np.array(range(len(X))).reshape(-1, 1), axis = 1)

                #split into treated (T=1) and untreated (T=0) groups
                exog0 = exog[exog[:,-2] == 0]
                y0 = y[new_T.flatten() == 0].reshape(-1,1)
                exog1 = exog[exog[:,-2] == 1]
                y1 = y[new_T.flatten() == 1].reshape(-1,1)

                #run a regression model if needed
                if self.model == 'linear':
                    #import necessary libraries
                    from statsmodels.regression.linear_model import OLS
                    from statsmodels.tools import add_constant
                    #fit a GLS model and predict values
                    OLS1 = OLS(y1, add_constant(exog1[:,:self.n_covars])).fit()
                    OLS0 = OLS(y0, add_constant(exog0[:,:self.n_covars])).fit()
                    ey1 = OLS1.predict(add_constant(X)).reshape(-1,1)
                    ey0 = OLS0.predict(add_constant(X)).reshape(-1,1)
                elif self.model == 'GAM':
                    #import necessary libraries
                    from pygam import GAM, s

                    #fit a GAM model and predict values 
                    GAM1 = GAM(s(0, spline_order=3, n_splines=6), distribution='poisson', link='log').fit(exog1[:,:self.n_covars], y1)
                    GAM0 = GAM(s(0, spline_order=3, n_splines=6), distribution='poisson', link='log').fit(exog0[:,:self.n_covars], y0)
                    #changed exog1 to exog0 and y1 to y0 in the line above
                    ey1 = GAM1.predict(exog[:,:self.n_covars]).reshape(-1,1)
                    ey0 = GAM0.predict(exog[:,:self.n_covars]).reshape(-1,1)

                elif self.model == 'forest':
                    #import necessary libraries
                    from sklearn.ensemble import RandomForestRegressor

                    #fit a decision tree model and predict values
                    forest1 = RandomForestRegressor().fit(exog1[:,:self.n_covars], y1.ravel())
                    forest0 = RandomForestRegressor().fit(exog0[:,:self.n_covars], y0.ravel())
                    ey1 = forest1.predict(exog[:,:self.n_covars]).reshape(-1,1)
                    ey0 = forest0.predict(exog[:,:self.n_covars]).reshape(-1,1)

                if self.method == 'IPW':
                    y0 = np.concatenate((y.reshape(-1,1),np.array(range(len(X))).reshape(-1, 1)), axis = 1)[T.flatten() == 0]
                    y1 = np.concatenate((y.reshape(-1,1),np.array(range(len(X))).reshape(-1, 1)), axis = 1)[T.flatten() == 1]
                    ATE = IPW(new_T, y1, y0, propensity)
                elif self.method == 'RA':
                    ATE = RA(ey1, ey0)
                elif self.method == 'AIPW':
                    ATE = AIPW(new_T, y, propensity, ey1, ey0)
                    
                #print iteration
                ATEs.append(ATE)
                print('iteration', i+1, 'run')

            #calculate p-value
            #sns.histplot(ATEs)
            p = np.sum(np.abs(ATEs) >= np.abs(self.ATE)) / len(ATEs)

        results = {
            't_statistic': t,
            'p_value': p
        }
    
        return results
    
def Individual_IPW(a, y1, y0, ps):
    import numpy as np
    ey1 = np.zeros(len(ps))
    ey0 = ey1
    for i in range(len(ps)):
        if a[i] == 1:
            ey1[i] = y1[y1[:, 1] == i, 0]
            ey0[i] = 0
        else:
            ey1[i] = 0
            ey0[i] = y0[y0[:, 1] == i, 0]
    return (a * ey1) / ps - (1 - a) * ey0 / (1 - ps)

def IPW(a, y1, y0, ps):
    import numpy as np
    return np.mean(Individual_IPW(a, y1, y0, ps))


def RA(ey1, ey0):
    import numpy as np
    return np.mean(ey1-ey0)


def Individual_AIPW(a, y, ps, ey1, ey0):
    return (a*y)/ps - (1-a)*y/(1-ps) - ((a-ps)/(ps * (1-ps)))*((1-ps)*ey1+ps*ey0)


def AIPW(a, y, ps, ey1, ey0):
    import numpy as np
    return np.mean(
            (a*y)/ps - (1-a)*y/(1-ps) - ((a-ps)/(ps * (1-ps)))*((1-ps)*ey1+ps*ey0)
    )

def causal_bootstrap(data, n_samples=10000, propensity_score_model = 'GAM', model = 'linear', method = 'AIPW', n_covars = 20, graph = False):
    #import resample function
    from sklearn.utils import resample
    import numpy as np

    #resample the data, result is a 4d numpy array
    samples = []
    samples.append([resample(data, replace=True) for i in range(n_samples)])
    samples = np.array(samples)  
    
    results = np.zeros(n_samples)
    #iterate through the samples
    for i in range(n_samples):
        X = samples[0, i, :, :-2]
        T = samples[0, i, :, -2].reshape(-1,1)
        y = samples[0, i, :, -1].reshape(-1,1)

        #estimate propensity scores using a model if needed
        if method == 'RA':
            print("no propensity score required for RA")
        elif propensity_score_model == 'logit':
            from statsmodels.discrete.discrete_model import Logit
            from statsmodels.tools import add_constant
            propensity = Logit(T, add_constant(X)).fit().predict().reshape(-1,1)
            # correct for values of 0
            for j in range(len(propensity)):
                if propensity[j] == 0:
                    propensity[j] = 1e-5
                elif propensity[j] == 1:
                    propensity[j] = 0.99999
        elif propensity_score_model == 'probit':
            from statsmodels.discrete.discrete_model import Probit
            from statsmodels.tools import add_constant
            propensity = Probit(T, add_constant(X)).fit().predict().reshape(-1,1)
            # correct for values of 0
            for j in range(len(propensity)):
                if propensity[j] == 0:
                    propensity[j] = 1e-5
                elif propensity[j] == 1:
                    propensity[j] = 0.99999
        elif propensity_score_model == 'GAM':
            from pygam import GAM, s
            propensity = GAM(s(0, spline_order=3, n_splines=6), distribution='poisson', link='log').fit(X, T).predict(X).reshape(-1,1)
            # correct for values of 0
            for j in range(len(propensity)):
                if propensity[j] == 0:
                    propensity[j] = 1e-5
                elif propensity[j] == 1:
                    propensity[j] = 0.99999
        elif propensity_score_model == 'SVC':
            from sklearn.svm import SVC
            propensity = SVC(probability=True).fit(X, T).predict_proba(X)[:,1].reshape(-1,1)
            # correct for values of 0
            for j in range(len(propensity)):
                if propensity[j] == 0:
                    propensity[j] = 1e-5
                elif propensity[j] == 1:
                    propensity[j] = 0.99999
        elif propensity_score_model == 'forest':
            from sklearn.ensemble import RandomForestRegressor
            propensity = RandomForestRegressor().fit(X, T.ravel()).predict(X).reshape(-1,1)
            # correct for values of 0
            for j in range(len(propensity)):
                if propensity[j] == 0:
                    propensity[j] = 1e-5
                elif propensity[j] == 1:
                    propensity[j] = 0.99999
        else:
            print('propensity_score_model must be either logit, probit, GAM, SVC, or forest')
        

        #stack the covariates and treatment into exog
        exog = np.append(np.concatenate((X,T), axis = 1),np.array(range(len(X))).reshape(-1, 1), axis = 1)

        #split into treated (T=1) and untreated (T=0) groups
        exog0 = exog[exog[:,-2] == 0]
        y0 = y[T.flatten() == 0]
        exog1 = exog[exog[:,-2] == 1]
        y1 = y[T.flatten() == 1]

        #run a regression model if needed
        if method == 'IPW':
            print("no regression required for IPW")
        elif model == 'linear':
            #import necessary libraries
            from statsmodels.regression.linear_model import OLS
            from statsmodels.tools import add_constant

            #fit a GLS model and predict values
            OLS1 = OLS(y1, add_constant(exog1[:,:n_covars])).fit()
            OLS0 = OLS(y0, add_constant(exog0[:,:n_covars])).fit()
            ey1 = OLS1.predict(add_constant(X)).reshape(-1,1)
            ey0 = OLS0.predict(add_constant(X)).reshape(-1,1)
        elif model == 'GAM':
            #import necessary libraries
            from pygam import GAM, s

            #fit a GAM model and predict values 
            GAM1 = GAM(s(0, spline_order=3, n_splines=6), distribution='poisson', link='log').fit(exog1[:,:n_covars], y1)
            GAM0 = GAM(s(0, spline_order=3, n_splines=6), distribution='poisson', link='log').fit(exog0[:,:n_covars], y0)
            ey1 = GAM1.predict(exog[:,:n_covars]).reshape(-1,1)
            ey0 = GAM0.predict(exog[:,:n_covars]).reshape(-1,1)
        elif model == 'forest':
            #import necessary libraries
            from sklearn.ensemble import RandomForestRegressor

            #fit a decision tree model and predict values
            forest1 = RandomForestRegressor().fit(exog1[:,:n_covars], y1.ravel())
            forest0 = RandomForestRegressor().fit(exog0[:,:n_covars], y0.ravel())
            ey1 = forest1.predict(exog[:,:n_covars]).reshape(-1,1)
            ey0 = forest0.predict(exog[:,:n_covars]).reshape(-1,1)
        else:
            print('model must be either linear, GAM, or forest')

        #return the values from the nonparametric bootstrap in an array
        if method == 'IPW':
            y0 = np.concatenate((y.reshape(-1,1),np.array(range(len(X))).reshape(-1, 1)), axis = 1)[T.flatten() == 0]
            y1 = np.concatenate((y.reshape(-1,1),np.array(range(len(X))).reshape(-1, 1)), axis = 1)[T.flatten() == 1]
            results[i] = (IPW(T, y1, y0, propensity))
        elif method == 'RA':
            results[i] = (RA(ey1, ey0))
        elif method == 'AIPW':
            results[i] = (AIPW(T, y, propensity, ey1, ey0))
        else:
            print('method must be either IPW, AIPW, or RA')

        print('iteration', i+1, 'run')


    #plot the distribution of estimated values to check distribution
    if graph == True:
        import matplotlib.pyplot as plt
        import seaborn as sns
    
        plt.figure(figsize=(10, 6))
        sns.histplot(results)
        plt.title('Distribution of Estimates')
        plt.xlabel('Estimates (Normality Check)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    return np.array(results), samples


def causal_regime(adata, pathwayname, genename, propensity_score_model = 'GAM', model = 'linear', method = 'AIPW', bootstrap = False, 
                  teststyle = 'Permutation', n_covars = 20, n_samples = 10000, graph = False):
    #import required libraries
    import numpy as np
    import scanpy as sc
    #from causal_function import IPW, AIPW, RA
    print(f"propensity_score_model: {propensity_score_model}")
    print(f"model: {model}")
    print(f"method: {method}")
    #print(f"SE: {SE}")

    #obtain pca values
    sc.tl.pca(adata, n_comps=n_covars)
    # define covariates (X), treatment (T) (through binarization, 1 if value > 0), and outcome (y)
    X = adata.obsm['emb_pca']
    T = np.where(adata.obsm['commot-cellchat-sum-receiver'][pathwayname].values.reshape(-1,1) != 0, 1, 0)
    y = adata.X[:, adata.var_names == genename].toarray().flatten()
   
    #The data is organized as covariates first, then treatment, then outcome. 
    data_array = np.concatenate((X, T.reshape(-1,1), 
                                 y.reshape(-1,1)), axis=1)
    


    #estimate propensity scores using a model if needed
    if method == 'RA':
        print("no propensity score required for RA")
    elif propensity_score_model == 'logit':
        from statsmodels.discrete.discrete_model import Logit
        from statsmodels.tools import add_constant
        propensity = Logit(T, X).fit().predict().reshape(-1,1)
        #correct for values of 0
        for j in range(len(propensity)):
                if propensity[j] == 0:
                    propensity[j] = 1e-5
                elif propensity[j] == 1:
                    propensity[j] = 0.99999
    elif propensity_score_model == 'probit':
        from statsmodels.discrete.discrete_model import Probit
        from statsmodels.tools import add_constant
        propensity = Probit(T, X).fit().predict().reshape(-1,1)
        #correct for values of 0
        for j in range(len(propensity)):
                if propensity[j] == 0:
                    propensity[j] = 1e-5
                elif propensity[j] == 1:
                    propensity[j] = 0.99999
    elif propensity_score_model == 'GAM':
        from pygam import GAM, s
        propensity = GAM(s(0, spline_order=3, n_splines=6)).fit(np.array(X), T).predict(X).reshape(-1,1)
        #correct for values of 0
        for j in range(len(propensity)):
               if propensity[j] <= 0:
                   propensity[j] = 1e-5
               elif propensity[j] >= 1:
                   propensity[j] = 0.99999
    elif propensity_score_model == 'SVC':
        from sklearn.svm import SVC
        propensity = SVC(probability=True).fit(X, T.ravel()).predict_proba(X)[:,1].reshape(-1,1)
        # correct for values of 0
        for j in range(len(propensity)):
            if propensity[j] == 0:
                propensity[j] = 1e-5
            elif propensity[j] == 1:
                propensity[j] = 0.99999
    elif propensity_score_model == 'forest':
        from sklearn.ensemble import RandomForestRegressor
        propensity = RandomForestRegressor().fit(X, T.ravel()).predict(X).reshape(-1,1)
        #correct for values of 0
        for j in range(len(propensity)):
                if propensity[j] <= 0:
                    propensity[j] = 1e-5
                elif propensity[j] >= 1:
                    propensity[j] = 0.99999
    else:
        print('propensity_score_model must be either logit, probit, GAM, SVC, or forest')


    #plot the distribution of propensity scores to check overlap
    if graph == True and method != 'RA':
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(10, 6))
        sns.histplot(propensity)
        plt.title('Distribution of Propensity scores')
        plt.xlabel('Propensity Score (Overlap Check)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
  
    #stack the covariates and treatment into exog
    exog = np.append(np.concatenate((X,T.reshape(-1,1)), axis = 1),np.array(range(len(X))).reshape(-1, 1), axis = 1)

    #split into treated (T=1) and untreated (T=0) groups
    exog0 = exog[exog[:,-2] == 0]
    y0 = y[T.flatten() == 0].reshape(-1,1)
    exog1 = exog[exog[:,-2] == 1]
    y1 = y[T.flatten() == 1].reshape(-1,1)

    #run a regression model if needed
    if method == 'IPW':
        print("no regression required for IPW")
    elif model == 'linear':
        #import necessary libraries
        from statsmodels.regression.linear_model import OLS
        from statsmodels.tools import add_constant
        #fit a GLS model and predict values
        OLS1 = OLS(y1, add_constant(exog1[:,:n_covars])).fit()
        OLS0 = OLS(y0, add_constant(exog0[:,:n_covars])).fit()
        ey1 = OLS1.predict(add_constant(X)).reshape(-1,1)
        ey0 = OLS0.predict(add_constant(X)).reshape(-1,1)
    elif model == 'GAM':
        #import necessary libraries
        from pygam import GAM, s

        #fit a GAM model and predict values 
        GAM1 = GAM(s(0, spline_order=3, n_splines=6), distribution='poisson', link='log').fit(exog1[:,:n_covars], y1)
        GAM0 = GAM(s(0, spline_order=3, n_splines=6), distribution='poisson', link='log').fit(exog0[:,:n_covars], y0)
        #changed exog1 to exog0 and y1 to y0 in the line above
        ey1 = GAM1.predict(exog[:,:n_covars]).reshape(-1,1)
        ey0 = GAM0.predict(exog[:,:n_covars]).reshape(-1,1)
    elif model == 'forest':
        #import necessary libraries
        from sklearn.ensemble import RandomForestRegressor

        #fit a decision tree model and predict values
        forest1 = RandomForestRegressor().fit(exog1[:,:n_covars], y1.ravel())
        forest0 = RandomForestRegressor().fit(exog0[:,:n_covars], y0.ravel())
        ey1 = forest1.predict(exog[:,:n_covars]).reshape(-1,1)
        ey0 = forest0.predict(exog[:,:n_covars]).reshape(-1,1)
    else:
        print('model must be either linear, GAM, or forest')


    #calculate the average treatment effect and variance
    #either using bootstrap
    if bootstrap == True:
        result, samples = causal_bootstrap(data_array, n_samples=n_samples, propensity_score_model = propensity_score_model, model = model, method = method, n_covars = n_covars, graph=graph)
        if method == 'IPW':
            y0 = np.concatenate((y.reshape(-1,1),np.array(range(len(X))).reshape(-1, 1)), axis = 1)[T.flatten() == 0]
            y1 = np.concatenate((y.reshape(-1,1),np.array(range(len(X))).reshape(-1, 1)), axis = 1)[T.flatten() == 1]
            ATE = IPW(T, y1, y0, propensity)
            #ATE = np.median(result)
            V = (np.std(result))**2
        elif method == 'RA':
            ATE = RA(ey1, ey0)
            #ATE = np.median(result)
            V = (np.std(result))**2
        elif method == 'AIPW':
            ATE = AIPW(T, y, propensity, ey1, ey0)
            #ATE = np.median(result)
            V = (np.std(result))**2
        else:
            print('method must be either IPW, AIPW, or RA')
    
    #or using a naive variance estimate
    elif bootstrap == False:
        if method == 'IPW':
            y0 = np.concatenate((y.reshape(-1,1),np.array(range(len(X))).reshape(-1, 1)), axis = 1)[T.flatten() == 0]
            y1 = np.concatenate((y.reshape(-1,1),np.array(range(len(X))).reshape(-1, 1)), axis = 1)[T.flatten() == 1]
            ATE = IPW(T, y1, y0, propensity)
            V = 1/((len(X)-n_covars)**2) * np.sum( (Individual_IPW(T, y1,y0, propensity) - ATE)**2)
        elif method == 'RA':
            ATE = RA(ey1, ey0)
            V = 1/((len(X)-n_covars)**2) * np.sum( (ey1-ey0 - ATE)**2)
        elif method == 'AIPW':
            ATE = AIPW(T, y, propensity, ey1, ey0)
            V = 1/((len(X)-n_covars)**2) * np.sum((Individual_AIPW(T, y, propensity, ey1, ey0) - ATE)**2)
        else:
            print('method must be either IPW, AIPW, or RA')


    #returns estimate, standard error, bootstrap info, and data array of X,T,y into a class
    if teststyle == 'Bootstrap':
        return CausalFunction(ATE, np.sqrt(V), 'Bootstrap', samples, result, data_array, method, model, propensity_score_model, n_covars), data_array
    elif teststyle == 'Permutation':
        return CausalFunction(ATE, np.sqrt(V), 'Permutation', None, None, data_array, method, model, propensity_score_model, n_covars), data_array
    elif teststyle == 'Naive':
        return CausalFunction(ATE, np.sqrt(V), 'Naive', None, None, data_array, method, model, propensity_score_model, n_covars), data_array
    else:
        return None