import scanpy as sc
import anndata
import pandas as pd
import numpy as np

#find a more limited number of highly variable genes
#this might be a bit redundant, but we make a new adata object that only uses the highly variable genes found in preprocessing and then we 
#look for the most highly variable genes in that dataset
def findgenes(adata, ngenes = 2000):
    """
    Find the most highly variable genes

    Parameters
    ----------
    adata
        The data matrix of shape ``n_obs`` × ``n_var``.
        Rows correspond to cells or spots and columns to genes.
    ngenes
        The number of genes to return. Default is 200
    
    Returns
    -------
    list
        A list of strings corresponding to the genes
    
    """
    #use the highly variable command on the adata to find the most variable genes
    adata2 = adata[:, adata.var.highly_variable]
    sc.pp.highly_variable_genes(adata2, n_top_genes=ngenes)
    adata3 = adata2[:, adata2.var.highly_variable]
    #return the names of the most variable genes
    return adata3.var_names


#find highly variable signalling pathways
def findsignals(adata, nsignals=100, filtermincells=20, style='most active'):
    """
    Find the most highly variable signalling pathways

    Parameters
    ----------
    adata
        The data matrix of shape ``n_obs`` × ``n_var``.
        Rows correspond to cells or spots and columns to genes.
    nsignals
        The number of signalling pathways to return. Default is 10
    filtermincells
        The minimum number of cells that a signalling pathway must be expressed in. Default is 20
    style
        The method used to find the signalling pathways. Default is 'most active'
        Options are 'usual' or 'most active'
    
    Returns
    -------
    list
        A list of strings corresponding to the signalling pathways
    
    """
    if style=='usual':
        #obtain signals, filter out those with low activity, and log transform
        signals = anndata.AnnData(X = adata.obsm['commot-cellchat-sum-receiver'])
        sc.pp.filter_genes(signals, min_cells=filtermincells)
        sc.pp.log1p(signals)
    
        #use the highly variable command on the signals to find the most variable signals
        sc.pp.highly_variable_genes(signals, n_top_genes = nsignals)
        signals2 = signals[:, signals.var.highly_variable]
        return signals2.var_names
    
    if style=='most active':
        #obtain signals, filter out those with low activity
        signals = anndata.AnnData(X = adata.obsm['commot-cellchat-sum-receiver'])
        sc.pp.filter_genes(signals, min_cells=filtermincells)
        #find the most active signals by summing the signals across cells. This method is preferred
        summed = adata.obsm['commot-cellchat-sum-receiver'].sum(axis=0)
        return summed.nlargest(nsignals).index
    


def masscausalcandidates(adata, interestgenes, interestsignals, threshold = 0.0):
    """
    Obtain a DataFrame of candidate signal-gene pairs

    Parameters
    ----------
    adata
        The data matrix of shape ``n_obs`` × ``n_var``.
        Rows correspond to cells or spots and columns to genes.
    interestgenes
        A list of strings corresponding to genes in adata
    interestsignals
        A list of strings corresponding to signals in adata after running COMMOT
    threshold
        The minimum gap between the maximum and minimum values of the GAM model. Default is 0.0

    Returns
    -------
    DataFrame : pandas
        A DataFrame with the columns 'Pathway', 'Gene', 'GAM_p', and 'Gaps'
        'GAM_p' is the p-value of the GAM model
        'Gaps' is the gap between the maximum and minimum values of the GAM model

    """
    #generate a GAM model for each signal-gene pair of interest. The model is a spline of degree 6.
    GAMresults = massGAMfitter(adata, interestgenes, interestsignals, nsplines = 6)
    #generate a dataframe with the p-values and gaps for each signal-gene pair
    pvalframe = pd.DataFrame(columns = ['Pathway', 'Gene', 'GAM_p', 'Gaps'])
    pathways = list()
    for i in interestsignals:
        #generate a pathway to correspond to each gene
        pathways = pathways + [i] * len(interestgenes)
    pvalframe['Pathway'] = pathways
    genes = list()
    #generate a gene to correspond to each pathway from before
    genes = genes + list(interestgenes) * len(interestsignals)
    pvalframe['Gene'] = genes
    pvals = list()
    #generate p-values corresponding to each signal-gene pair from the GAM model
    for signal in interestsignals:
        for gene in interestgenes:
            pvals.append(GAMresults[signal][gene]['p_value'])
    pvalframe['GAM_p'] = pvals
    gaps = list()
    #generate gaps
    for i in range(0, len(pvalframe)):
            signal = pvalframe.loc[i, 'Pathway']
            gene = pvalframe.loc[i, 'Gene']
            model = GAMresults[str(signal)][str(gene)]['model']
            X_grid = GAMresults[signal][gene]['X']
            predictions = model.predict(X_grid)
            gap = max(predictions) - min(predictions)
            gaps.append(gap)
    pvalframe['Gaps'] = gaps
    #filter out signal-gene pairs with small gaps
    pvalframe = pvalframe[pvalframe['Gaps'] >= threshold]  
    pvalframe = pvalframe.reset_index(drop=True)   
    #sort by p-values from the GAM model
    pvalframe = pvalframe.sort_values(by = 'GAM_p', ascending = True)
    pvalframe = pvalframe.reset_index(drop=True)    
    return pvalframe

#this does the causal function many times
#you need to input an adata object
#you also need to input a pandas dataframe called candidates
#the dataframe should have at least two columns - 'Pathway' and 'Gene'
#each row is a signal-gene pair for which we want to run the causal inference algorithm
#you can use the pandas dataframe output by the masscausalcandidates function, but you can also make your own
#the output is a new dataframe that will include columns for the ATE and the causal inference p-value
def masscausal(adata, candidates, propensity_score_model = 'GAM', model = 'linear', method = 'AIPW', bootstrap=False, teststyle='Naive', n_covars = 20, n_samples = 10000, graph=False):
    """
    Calculate the causal relationship between a pathway and gene from the adata file

    Parameters
    ----------
    adata
        The data matrix of shape ``n_obs`` × ``n_var``.
        Rows correspond to cells or spots and columns to genes.
    candidates
        DataFrame that has column called pathway and a column called gene
        Represents possible candidate pairs that are already paired up (no missing data, same number of rows)
    propensity_score_model:
        The model used to estimate the propensity score. Default is 'logit'
        Options are 'logit', 'probit', 'SVC, 'GAM', or 'forest'
    model:
        The model used to estimate the predicted regression results. Default is 'linear'
        Options are 'linear', 'GAM', or 'forrest'
    method:
        The method used to calculate the treatment effect. Default is 'AIPW'
        Options are 'AIPW', 'IPW', or 'RA'
    bootstrap:
        Whether to use bootstrapping to calculate the confidence intervals. Default is False
        Options are True or False
    teststyle:
        The method used to calculate the p-value. Default is 'Permutation'
        Options are 'Naive', 'Permutation' or 'Bootstrap'
    n_covars:
        The number of covariates to use in the propensity score model. Default is 15
    n_samples:
        The number of samples to use in the bootstrap. Default is 10000
    graph:
        Whether to generate plots for propensity score and bootstrap (if bootstrap is True). Default is False
        Options are True or False

    Returns
    -------
    DataFrame : pandas
        A DataFrame with the columns 'Pathway', 'Gene', 'ATEs', and 'p_values'
        'Pathway' and 'Gene' are the same as the input DataFrame
        'ATEs' is the average treatment effect
        'p_values' is the p-value of the causal inference test
        The DataFrame is sorted based on smallest p-value

    """

    #generate results dataframe, first 2 columns are pairs from the candidates dataframe
    causalresults = candidates
    ATEs = list()
    pvals = list()
    for i in range(0, len(candidates)):
        count = i
        denom = len(candidates)
        #to keep track of output
        print(count, count/denom)
        #run the causal inference algorithm
        causal_result = causal_regime(adata, candidates.loc[i, 'Pathway'], candidates.loc[i, 'Gene'], propensity_score_model = propensity_score_model, model = model, method = method, bootstrap = bootstrap, teststyle = teststyle, n_covars = n_covars, n_samples = n_samples, graph=graph)
        #append the results
        ATEs.append(causal_result[0].ATE)
        pvals.append(causal_result[0].test()['p_value'])
        print(causal_result[0])
    #add two columns to the resulting dataframe
    causalresults['ATEs'] = ATEs
    causalresults['p_values'] = pvals
    #sort by p-values
    causalresults = causalresults.sort_values(by='p_values', ascending=True)
    causalresults = causalresults.reset_index(drop=True)
    return causalresults   



#we want a way to rank the top signal-gene pairs
#many of them are going to have the same p-values
#we will calculate pearson correlation coefficients and use that to break ties
#this function takes in the dataframe generated by the masscausal function
#it adds a 'Pearson' column
#it also sorts the dataframe by AIPW p-value, but with the pearson correlation coefficients as tie-breakers
#it returns this sorted dataframe
def pearsonties(adata, causalresults):
    """
    Rank the top signal-gene pairs using Pearson correlation coefficients

    Parameters
    ----------
    adata
        The data matrix of shape ``n_obs`` × ``n_var``.
        Rows correspond to cells or spots and columns to genes.
    causalresults
        DataFrame that has column called pathway, gene, ATEs, and p_values
        Represents the results of the causal inference test
    
    Returns
    -------
    DataFrame : pandas
        A DataFrame with the columns 'Pathway', 'Gene', 'ATEs', 'p_values', and 'Pearson'
        'Pathway', 'Gene', 'ATEs', and 'p_values' are the same as the input DataFrame
        'Pearson' is the Pearson correlation coefficient
        The DataFrame is sorted based on smallest p-value and largest Pearson correlation coefficient

    """
    #first we will make some vectors to speed up calculations
    interestsignals = list(set(causalresults['Pathway']))
    interestgenes = list(set(causalresults['Gene']))
    adata.layers['counts'] = adata.X
    pearsons = pd.DataFrame(columns = interestsignals, index = interestgenes)
    signalmeans = {}
    signalvectors = {}
    signalsds = {}
    for signal in list(pearsons.columns):
        signalmeans[signal] = np.mean(adata.obsm['commot-cellchat-sum-receiver'][signal])
        signalvectors[signal] = adata.obsm['commot-cellchat-sum-receiver'][signal] - signalmeans[signal]
        signalsds[signal] = np.sqrt(np.sum(np.square(adata.obsm['commot-cellchat-sum-receiver'][signal] - signalmeans[signal])))

    genemeans = {}
    genevectors = {}
    genesds = {}
    for gene in list(pearsons.index):
        genemeans[gene] = np.mean(adata.layers['counts'][:, adata.var_names == gene].toarray().flatten())
        genevectors[gene] = adata.layers['counts'][:, adata.var_names == gene].toarray().flatten() - genemeans[gene]
        genesds[gene] = np.sqrt(np.sum(np.square(adata.layers['counts'][:, adata.var_names == gene].toarray().flatten() - genemeans[gene])))

    #now the actual pearsons calculations
    for signal in list(pearsons.columns):
        for gene in list(pearsons.index):
            pearsons.loc[gene, signal] = np.sum(np.multiply(signalvectors[signal], genevectors[gene]))/(signalsds[signal]*genesds[gene])
    
    #and we want to stack by absolute value (rank by absolute value)
    abs = pearsons.abs()
    stacked = abs.stack()
    stacked = stacked.reset_index()
    stacked.columns = ['Gene', 'Pathway', 'Pearson']
    sorted_stacked = stacked.sort_values(ascending=False, by='Pearson')
    sorted_stacked = sorted_stacked.reset_index(drop=True)

    #now we merge the pearson values with the causal results
    causal = pd.merge(causalresults, sorted_stacked, on=['Pathway', 'Gene'], how='inner')
    causal = causal.sort_values(by=['p_values','Pearson'], ascending=[True, False], ignore_index=True)

    return causal
