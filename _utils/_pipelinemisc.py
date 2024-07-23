
from pygam import GAM, s
import numpy as np


#GAM fitter function
#this is back-end and is just necessary to make the mass GAM fitter run
#alternatively, we can run it so that we can get plots
def GAMfitter(adata, genes_ordered, signal, nsplines=6):
    # Create a dictionary to store the results
    gene_gam_results = {}
    adata.layers['counts'] = adata.X
    
    # Extract signal values and ensure it is 2D with samples as rows
    signalling_values = adata.obsm['commot-cellchat-sum-receiver'][[signal]].values
    if len(signalling_values.shape) == 1:
        signalling_values = signalling_values.reshape(-1, 1)  # Reshape to 2D if necessary
    
    # Iterate over all genes
    #I could change this to only look at genes_ordered, which would probably be a lot faster
    for gene in genes_ordered:
        # Extract gene expression data
        gene_expression = adata.layers['counts'][:, adata.var_names == gene].toarray().flatten()
        
        # Initialize and fit the GAM model
        gam = GAM(s(0, spline_order=3, n_splines=nsplines), distribution='poisson', link='log')
        gam.fit(signalling_values, np.log1p(gene_expression))  
        X = gam.generate_X_grid(term=0)
        
        # Store the fitted model and its p-value
        gene_gam_results[gene] = {
            'X': X, # Store X_grid
            'model': gam,
            'p_value': gam.statistics_['p_values'][0],  
            'y': gene_expression,  # Store the gene expression data as well
        }
        
    return gene_gam_results


#this function fits GAM models for multiple signals
#to access, we need results[signal][gene]
#this is back-end code to make the masscausal candidates stuff run
def massGAMfitter(adata, genes_ordered, signal_list, nsplines = 6):
    gam_results = {}
    for i in signal_list:
        gam_results[i] = GAMfitter(adata, genes_ordered, signal=i, nsplines=nsplines)
    return gam_results