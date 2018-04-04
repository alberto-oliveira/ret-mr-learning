### ADDING NEW COLLECTION TO THE EXPERIMENTS ###
###                                          ###
################################################

-> A new experiment is performed on the resulting ranks obtained by applying a 
retrieval methodology (summarized as "descriptor") on a collection, using a set 
of queries. To perform relevance prediction, two results are expected (1) A set
of ranks, one for each query used, and a set of relevance labels, one for each
rank. The relevance labels indicate which of the ranked results are relevant or
not. To add a new collection to the experimental workflow, the following steps
must be followed:
* Suppose the dataset used is named 'imagesA' and the descriptor used is named
  'descriptorA'. Consider that 'descriptorA' is the first being added for this
  dataset, and thus is desc1 in order.
(1) Create a new directory to hold the subdirectories for the cross-validation
    division of the ranks. The subdirectories shall be named fold_f, where f
	are sequential integers numbering each of the folds. The .rk files obtained
	by performing the queries on the collection are added to those folders.
	
(2) Create a new folder to hold the relevance label files for each of the .rk 
    files. The label files are numpy array (.npy) files, and there should be a 
	total of f, with f being the number of folds. Each of the .npy files holds
	a (n, r) array, n being the total number of ranks in that fold, and r being
	the number of returned values for the respective rank (the rows should be in
	sorted order). Each position is 1 or 0, to indicate if the respective position
	of the respective rank is relevant or not.
	
(3) Add the imagesA_desc1 value to each of the sections in the paths.cfg file,
    indicating the respective paths to the files resulting to the experiments
	in that collection.
	
(4) Add the imagesA_desc1 section to the dbparams.cfg file. The section must
    contain a 'pK', which contains the Precision at K value for that collection.
	The PK listed depends on the k used for prediction, for example, if you want
	to predict the relevance of the top 10 positions, then p10 should be listed.
	However, for all collections, pK should be the same. If wished to employ a
	different K, a new dbparams.cfg file should be created
	
(5) Add to the descriptor_map in mappings.py the keyword 'imagesA', respective
    to the collection, with value [1], as 'descriptorA' is the first descriptor,
	that is, 'desc1', used for the respective collection. If 'descriptorA' is not
	the first, you add to the keyword of 'imagesA' the respective value.
	