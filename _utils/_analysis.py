#the following functions are used to evaluate the effectiveness of our pipeline
import pandas as pd
from collections import defaultdict
import numpy as np

def pearsondf(adata, interestsignals, interestgenes):
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

    return sorted_stacked


#first, we must construct dictionaries of known biological information
#load in REACTOME
#REACTOME contains LR and TF pairs
REACTOMEhuman = pd.read_csv('REACTOMEhuman.csv')
REACTOMEhuman = REACTOMEhuman.drop(REACTOMEhuman.columns[[0, 2, 4, 5]], axis=1)
REACTOMEmouse = pd.read_csv('REACTOMEmouse.csv')
REACTOMEmouse = REACTOMEmouse.drop(REACTOMEmouse.columns[[0, 2, 4, 5]], axis=1)
#convert to dictionary
REACThuman_dict = REACTOMEhuman.to_dict(orient='records')
REACTmouse_dict = REACTOMEmouse.to_dict(orient='records')
#combine so that we have a list of TFs for each receptor
#human
combined_REACThuman = defaultdict(list)

for entry in REACThuman_dict:
    combined_REACThuman[entry['receptor']].append(entry['tf'])

REACThuman_dict = dict(combined_REACThuman)

#mouse
combined_REACTmouse = defaultdict(list)

for entry in REACTmouse_dict:
    combined_REACTmouse[entry['receptor']].append(entry['tf'])

REACTmouse_dict = dict(combined_REACTmouse)
#load in RegNet
#RegNet contains TF and gene pairs
#also convert to dictionary and remove nas
#human
RegNethuman = pd.read_csv('RegNetworkhuman.csv')
RegNethuman = RegNethuman.drop(RegNethuman.columns[[0]], axis=1)
RegNethuman_dict = RegNethuman.to_dict(orient='list')
for i in list(RegNethuman_dict.keys()):
    RegNethuman_dict[i] = [x for x in RegNethuman_dict[i] if str(x) != 'nan']

#mouse
RegNetmouse = pd.read_csv('RegNetworkmouse.csv')
RegNetmouse = RegNetmouse.drop(RegNetmouse.columns[[0]], axis=1)
RegNetmouse_dict = RegNetmouse.to_dict(orient='list')
for i in list(RegNetmouse_dict.keys()):
    RegNetmouse_dict[i] = [x for x in RegNetmouse_dict[i] if str(x) != 'nan']

#load in TRUST
#TRUST contains TF and gene pairs
#also convert to dictionary and remove nas
#human
TRUSThuman = pd.read_csv('TRUST.csv')
TRUSThuman = TRUSThuman.drop(TRUSThuman.columns[[0]], axis=1)
TRUSThuman_dict = TRUSThuman.to_dict(orient='list')
for i in list(TRUSThuman_dict.keys()):
    TRUSThuman_dict[i] = [x for x in TRUSThuman_dict[i] if str(x) != 'nan']

#mouse
TRUSTmouse = pd.read_csv('TRUSTmouse.csv')
TRUSTmouse = TRUSTmouse.drop(TRUSTmouse.columns[[0]], axis=1)
TRUSTmouse_dict = TRUSTmouse.to_dict(orient='list')
for i in list(TRUSTmouse_dict.keys()):
    TRUSTmouse_dict[i] = [x for x in TRUSTmouse_dict[i] if str(x) != 'nan']

#make dictionaries
#these dictionaries go from receptor to gene
#REACTREGhuman_dict combines REACTOME and RegNetwork (human)
#for every receptor in REACTOME, assign the genes from RegNetwork associated with the TFs in REACTOME/RegNetwork
comb_REACTREG_human = defaultdict()
for i in list(REACThuman_dict.keys()):
    comb_REACTREG_human[i] = list()
    for j in REACThuman_dict[i]:
        if j in list(RegNethuman_dict.keys()):
            comb_REACTREG_human[i] = comb_REACTREG_human[i] + RegNethuman_dict[j]
REACTREGhuman_dict = dict(comb_REACTREG_human)

#REACTREGmouse_dict combines REACTOME and RegNetwork (mouse)
comb_REACTREG_mouse = defaultdict()
for i in list(REACTmouse_dict.keys()):
    comb_REACTREG_mouse[i] = list()
    for j in REACTmouse_dict[i]:
        if j in list(RegNetmouse_dict.keys()):
            comb_REACTREG_mouse[i] = comb_REACTREG_mouse[i] + RegNetmouse_dict[j]
REACTREGmouse_dict = dict(comb_REACTREG_mouse)

#REACTTRUSThuman_dict combines REACTOME and TRUST (human)
comb_REACTTRUST_human = defaultdict()
for i in list(REACThuman_dict.keys()):
    comb_REACTTRUST_human[i] = list()
    for j in REACThuman_dict[i]:
        if j in list(TRUSThuman_dict.keys()):
            comb_REACTTRUST_human[i] = comb_REACTTRUST_human[i] + TRUSThuman_dict[j]
REACTTRUSThuman_dict = dict(comb_REACTTRUST_human)

#REACTTRUSTmouse_dict combines REACTOME and TRUST (mouse)
comb_REACTTRUST_mouse = defaultdict()
for i in list(REACTmouse_dict.keys()):
    comb_REACTTRUST_mouse[i] = list()
    for j in REACTmouse_dict[i]:
        if j in list(TRUSTmouse_dict.keys()):
            comb_REACTTRUST_mouse[i] = comb_REACTTRUST_mouse[i] + TRUSTmouse_dict[j]
REACTTRUSTmouse_dict = dict(comb_REACTTRUST_mouse)

#define function to remove duplicates
def remove_dups(dictionary):
    for key in list(dictionary.keys()):
        # Convert list to set to remove duplicates, then back to list
        dictionary[key] = list(set(dictionary[key]))

#use it
remove_dups(REACTREGmouse_dict)
remove_dups(REACTREGhuman_dict)
remove_dups(REACTTRUSTmouse_dict)
remove_dups(REACTTRUSThuman_dict)

#load in cellchat, which contains pathway names and receptors
CellChatmouse = pd.read_csv('CellChatDB.ligrec.mouse.csv')
CellChatmouse = CellChatmouse.drop(CellChatmouse.columns[[0, 1, 4]], axis=1)
CellChatmouse.columns = ['Receptor', 'Pathway']
CellChatmouse = CellChatmouse.drop_duplicates()
CellChathuman = pd.read_csv('CellChatDB.ligrec.human.csv')
CellChathuman = CellChathuman.drop(CellChathuman.columns[[0, 1, 4]], axis=1)
CellChathuman.columns = ['Receptor', 'Pathway']
CellChathuman = CellChathuman.drop_duplicates()

#human
#this dictionary goes from pathway to receptor
CellChathuman_dict = {}
for i in CellChathuman.index:
    if CellChathuman['Pathway'][i] not in CellChathuman_dict.keys():
        CellChathuman_dict[CellChathuman['Pathway'][i]] = [CellChathuman['Receptor'][i]]
    else:
        CellChathuman_dict[CellChathuman['Pathway'][i]] = CellChathuman_dict[CellChathuman['Pathway'][i]] + [CellChathuman['Receptor'][i]]

#mouse
CellChatmouse_dict = {}
for i in CellChatmouse.index:
    if CellChatmouse['Pathway'][i] not in CellChatmouse_dict.keys():
        CellChatmouse_dict[CellChatmouse['Pathway'][i]] = [CellChatmouse['Receptor'][i]]
    else:
        CellChatmouse_dict[CellChatmouse['Pathway'][i]] = CellChatmouse_dict[CellChatmouse['Pathway'][i]] + [CellChatmouse['Receptor'][i]]

def normalize_string(s):
    return s.replace('_', ',')

for key in CellChathuman_dict.keys():
    CellChathuman_dict[key] = [normalize_string(s) for s in CellChathuman_dict[key]]

for key in CellChatmouse_dict.keys():
    CellChatmouse_dict[key] = [normalize_string(s) for s in CellChatmouse_dict[key]]

#now make the full pathway to gene dictionary
#takes in a pathway name and reports all downstream genes
#human
full_humanRR = defaultdict()
for i in list(CellChathuman_dict.keys()):
    full_humanRR[i] = []
    for j in CellChathuman_dict[i]:
        if j in list(REACTREGhuman_dict.keys()):
            full_humanRR[i] = full_humanRR[i] + REACTREGhuman_dict[j]
humanRR = dict(full_humanRR)

#remove duplicates
full_humanRR = defaultdict()
for i in list(humanRR.keys()):
    full_humanRR[i] = list(set(humanRR[i]))
humanRR = dict(full_humanRR)

#mouse
full_mouseRR = defaultdict()
for i in list(CellChatmouse_dict.keys()):
    full_mouseRR[i] = []
    for j in CellChatmouse_dict[i]:
        if j in list(REACTREGmouse_dict.keys()):
            full_mouseRR[i] = full_mouseRR[i] + REACTREGmouse_dict[j]
mouseRR = dict(full_mouseRR)

#remove duplicates
full_mouseRR = defaultdict()
for i in list(mouseRR.keys()):
    full_mouseRR[i] = list(set(mouseRR[i]))
mouseRR = dict(full_mouseRR)

#repeat using the TRUST database instead of RegNet
full_humanRT = defaultdict()
for i in list(CellChathuman_dict.keys()):
    full_humanRT[i] = []
    for j in CellChathuman_dict[i]:
        if j in list(REACTTRUSThuman_dict.keys()):
            full_humanRT[i] = full_humanRT[i] + REACTTRUSThuman_dict[j]
humanRT = dict(full_humanRT)

full_humanRT = defaultdict()
for i in list(humanRT.keys()):
    full_humanRT[i] = list(set(humanRT[i]))
humanRT = dict(full_humanRT)

full_mouseRT = defaultdict()
for i in list(CellChatmouse_dict.keys()):
    full_mouseRT[i] = []
    for j in CellChatmouse_dict[i]:
        if j in list(REACTTRUSTmouse_dict.keys()):
            full_mouseRT[i] = full_mouseRT[i] + REACTTRUSTmouse_dict[j]
mouseRT = dict(full_mouseRT)

full_mouseRT = defaultdict()
for i in list(mouseRT.keys()):
    full_mouseRT[i] = list(set(mouseRT[i]))
mouseRT = dict(full_mouseRT)

#now we can search these dictionaries for the pairs we found
#the following function takes in a results dataframe
#it adds a column that indicates "yes" or "no" for whether or not a pair is in the database
def dictionarysearch(causalresults, pathwaydatabase, receptordatabase):
    """
    Parameters

    causalresults: the dataframe of results (i.e. the one produced by the masscausal function)

    pathwaydatabase: the database linking pathways with genes (either mouseRR, mouseRT, humanRR, or humanRT)

    receptordatabase: the database linking receptors with genes (either REACTREGmouse_dict, REACTTRUSTmouse_dict, 
    REACTREGhuman_dict, or REACTTRUSThuman_dict)
    """
    dictionary_search = list()
    for i in range(0,len(causalresults)):
        count_dashes = causalresults.loc[i, 'Pathway'].count('-')
        if count_dashes == 1:
            pathwayname = causalresults.loc[i, 'Pathway'][2:]
            if pathwayname in pathwaydatabase.keys():
                if causalresults.loc[i, 'Gene'] in pathwaydatabase[pathwayname]:
                    dictionary_search.append('yes')
                elif causalresults.loc[i, 'Gene'] not in pathwaydatabase[pathwayname]:
                    dictionary_search.append('no')
            elif pathwayname not in pathwaydatabase.keys():
                dictionary_search.append('no')
        
        elif count_dashes >= 2:
            first_dash_index = causalresults.loc[i, 'Pathway'].find('-')
            second_dash_index = causalresults.loc[i, 'Pathway'].find('-', first_dash_index + 1)
            receptorname = causalresults.loc[i, 'Pathway'][second_dash_index + 1:]
            receptorname = receptorname.replace('_', ',')
            if receptorname in receptordatabase.keys():
                if causalresults.loc[i, 'Gene'] in receptordatabase[receptorname]:
                    dictionary_search.append('yes')
                elif causalresults.loc[i, 'Gene'] not in receptordatabase[receptorname]:
                    dictionary_search.append('no')
            elif receptorname not in receptordatabase.keys():
                dictionary_search.append('no')
                        
    causalresults['dictionary search'] = dictionary_search
    return causalresults

#we can now calculate power metrics
#calculating how many true pairs are in the entire dataset
def findn(data, pathwaydatabase, receptordatabase):
    n = 0
    for signal in list(data.obsm['commot-cellchat-sum-receiver'].columns):
        for gene in list(data.var_names):
            count_dashes = signal.count('-')
            if count_dashes == 1:
                pathwayname = signal[2:]
                if pathwayname in pathwaydatabase.keys():
                    if gene in pathwaydatabase[pathwayname]:
                        n = n + 1
                    elif gene not in pathwaydatabase[pathwayname]:
                        n = n
                elif pathwayname not in pathwaydatabase.keys():
                    n = n
        
            elif count_dashes >= 2:
                first_dash_index = signal.find('-')
                second_dash_index = signal.find('-', first_dash_index + 1)
                receptorname = signal[second_dash_index + 1:]
                receptorname = receptorname.replace('_', ',')
                if receptorname in receptordatabase.keys():
                    if gene in receptordatabase[receptorname]:
                        n = n + 1
                    elif gene not in receptordatabase[receptorname]:
                            n = n
                elif receptorname not in receptordatabase.keys():
                    n = n
    return n

#we also need to calculate N
#this is the number of ALL possible pairs of a signalling pathway and a downstream gene
def calculateN(adata):
    #adata2 = adata[:, adata.var.highly_variable]
    genes = len(adata.var_names)
    return (len(adata.obsm['commot-cellchat-sum-receiver'].columns)) * genes

# finally, we need a value for ns, which is the number of selected receptor-downstream gene pairs that are actually true
# we will define true as being present in the databases
def calculatens(results):
    """
    results: the dataframe consisting of only the top ranked signal-gene pairs
    """
    count_value = results['dictionary search'].value_counts().get('yes', 0)
    return count_value  

#now the calculation
def powermetric(results, n, Ns, N):
    """
    results: the dataframe consisting of only the top ranked signal-gene pairs

    n: from above

    Ns: number of rows in results dataframe

    N: from above
    """
    n = n
    ns = calculatens(results)
    Ns = Ns
    tpr = ns/n
    fpr = (Ns-ns)/(N-n)
    pm = tpr/(tpr+fpr)
    print(n, N, ns, Ns, pm)
    return pm

#some other options
#sometimes a pathway promotes the expression of its own receptor
#maybe we want to remove these
def receptorfilter(results, cellchatchoice):
    """
    cellchatchoice should be either CellChatmouse_dict or CellChathuman_dict
    """
    overlap = list()
    for i in range(len(results)):
        gene = results.loc[i, 'Gene']
        dashcount = results.loc[i, 'Pathway'].count('-')
        if dashcount == 1:
            pathwayname = results.loc[i, 'Pathway'][2:]
            if pathwayname in cellchatchoice.keys():
                if gene in cellchatchoice[pathwayname]:
                    overlap.append('receptor')
                elif gene not in cellchatchoice[pathwayname]:
                    overlap.append('no')
            elif pathwayname not in cellchatchoice.keys():
                overlap.append('???')
        elif dashcount == 2:
            first_dash_index = results.loc[i, 'Pathway'].find('-')
            second_dash_index = results.loc[i, 'Pathway'].find('-', first_dash_index + 1)
            count_underscores = results.loc[i, 'Pathway'][second_dash_index + 1:].count('_')
            if count_underscores == 0:
                receptorname = results.loc[i, 'Pathway'][second_dash_index + 1:]
                if gene == receptorname:
                    overlap.append('receptor')
                elif gene != receptorname:
                    overlap.append('no')
            elif count_underscores == 1:
                first_underscore_index = results.loc[i, 'Pathway'][second_dash_index + 1:].find('_')
                receptorname1 = results.loc[i, 'Pathway'][second_dash_index+1:][:first_underscore_index]
                receptorname2 =  results.loc[i, 'Pathway'][second_dash_index+1:][first_underscore_index+1:]
                if gene in [receptorname1, receptorname2]:
                    overlap.append('receptor')
                elif gene not in [receptorname1, receptorname2]:
                    overlap.append('no')
            elif count_underscores == 2:
                first_underscore_index = results.loc[i, 'Pathway'][second_dash_index + 1:].find('_')
                second_underscore_index = results.loc[i, 'Pathway'][second_dash_index+1:][first_underscore_index +1:].find('_')
                receptorname1 = results.loc[i, 'Pathway'][second_dash_index+1:][:first_underscore_index]
                receptorname2 =  results.loc[i, 'Pathway'][second_dash_index+1:][first_underscore_index+1:second_underscore_index]
                receptorname3 = results.loc[i, 'Pathway'][second_dash_index+1:][first_underscore_index+1:][second_underscore_index:]
                if gene in [receptorname1, receptorname2, receptorname3]:
                    overlap.append('receptor')
                elif gene not in [receptorname1, receptorname2, receptorname3]:
                    overlap.append('no')
            elif count_underscores >= 3:
                overlap.append('too many')
    results['receptor overlap'] = overlap
    return results

#maybe we want to see if the transcription factors that are the intermediaries between a particular signal and 
# a particular gene are actually active in the dataset
def tfactivitysearch(data, causalresults, pathwaydatabase, receptordatabase, receptortfdatabase, tfgenedatabase, pathwayreceptordatabase):
    """
    data = filtered adata file
    causalresults = dataframe of results
    pathwaydatabase = mouseRR, mouseRT, humanRR, or humanRT
    receptordatabase = REACTREGmouse_dict, REACTTRUSTmouse_dict, REACTREGhuman_dict, REACTTRUSThuman_dict
    receptortfdatabase = REACTmouse_dict or REACThuman_dict
    tfgenedatabase = RegNetmouse_dict, TRUSTmouse_dict, RegNethuman_dict, TRUSThuman_dict
    pathwayreceptordatabase = CellChatmouse_dict, CellChathuman_dict
    """
    #make a new column to record our information

    for i in range(0,len(causalresults)):
        if causalresults.loc[i, "dictionary search"] == "yes":
            #we need two things to have an "interesting" TF
            #we need it to be in the set of TFs that correspond to the signal pathway
            #we also need the gene in question to be one of the TFs downstream genes
            genename = causalresults.loc[i, 'Gene']
            count_dashes = causalresults.loc[i, 'Pathway'].count('-')
            if count_dashes == 1:
            #if there's only one dash, then we're working with a pathway name
                pathwayname = causalresults.loc[i, 'Pathway'][2:]
                #go and find the list of receptors for this pathway
                receptorlist = pathwayreceptordatabase[pathwayname]
                tflist = list()
                #go find all the TFs for those receptors
                for receptorname in receptorlist:
                    #check to see if they correspond with the gene in question
                    if receptorname in receptortfdatabase.keys():
                        for tf in receptortfdatabase[receptorname]:
                            if tf in tfgenedatabase.keys():
                                if genename in tfgenedatabase[tf]:
                                    tflist.append(tf)

            elif count_dashes >= 2:
                #if there are multiple dashes, then we're dealing with a receptor
                first_dash_index = causalresults.loc[i, 'Pathway'].find('-')
                second_dash_index = causalresults.loc[i, 'Pathway'].find('-', first_dash_index + 1)
                receptorname = causalresults.loc[i, 'Pathway'][second_dash_index + 1:]
                receptorname = receptorname.replace('_', ',')
                #after identifying the receptor's name, we go and look for all the TFs that correspond to it
                tflist = list()
                for tf in receptortfdatabase[receptorname]:
                    #check if the gene in question matches those TFs
                    if tf in tfgenedatabase.keys():
                        if genename in tfgenedatabase[tf]:
                            tflist.append(tf)
            
            #filter to cells receiving this signal
            thissignal = causalresults.loc[i, "Pathway"]
            signalonlydata = data[data.obsm['commot-cellchat-sum-receiver'][thissignal] > 0]
            #sum up columns
            columnsums = signalonlydata.X.sum(axis=0)
            activitysums = pd.DataFrame(data=columnsums, columns=data.var_names)
            #threshold = 1000000000000000000
            #threshold = activitysums.iloc[0].quantile(0.75)
            #threshold = 0
            #use the median (probs also try the 1st and third quartile)
            ## also try making some plots that are p-value (or neg log of p-value) on the x-axis and have signal*tf on the y-axis
            ## we will hope to see positive linear relationship
            activityvalue = 0

            for TF in tflist:
                #for every TF that corresponds to the signal-gene path, check to see if its being expressed in the dataset
                if TF in data.var_names:
                    activityvalue = activityvalue + activitysums.loc[0, TF]

            causalresults.loc[i, "TF sum"] = activityvalue

# sometimes we also want to know the total amount of signal
def signalsum(results, data):
    for i in range(0,len(results)):
        if results.loc[i, "dictionary search"] == "yes":
            results.loc[i, "Signal sum"] = sum(data.obsm['commot-cellchat-sum-receiver'][results.loc[i, "Pathway"]])