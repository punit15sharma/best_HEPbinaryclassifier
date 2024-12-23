Hello there, Ranger,
This is a sample class for training a BDT to do binary classification on the list of samples provided in the YAML file. 

The YAML file also takes the features to train on and the hyperparameters of the BDT.

The best part of this class is that it trains the BDT and then can dump TTrees with the categorized arrays after the BDT score cut while also providing the best threshold value to maximize the significance. 

Simple and easy for a quick analysis, these mini-trees can be then passed onto making XML files for the fit and then one can run NLL fits on it.


cheers.
