from copy import deepcopy
from math import *
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

algorithms = ['MultiLayerPerceptron', 'LR', 'RandomForest', 'GaussianNB']
fair_algorithms = ['Feldman-' + algorithm for algorithm in algorithms]
all_algorithms = algorithms + fair_algorithms

metrics = ['predicted_delta', 'true_delta', 'labeled_delta', 'calibrated_delta']


class Dataset:
    def __init__(self, df, dataset_name) -> None:
        self.df = df
        self.df_test = deepcopy(df)  # use the same df_test for all, preserve the same ordering
        self.add_information(dataset_name)

    def __len__(self):
        return self.df.shape[0]

    def add_information(self, dataset_name):
        if dataset_name == 'adult':
            self.dataset_name = 'adult'
            self.class_attr = 'income-per-year'
            self.positive_class_val = 1
            self.sensitive_attrs = ['race', 'sex']  # privilidged: group 0
        elif dataset_name == 'bank':
            self.dataset_name = 'bank'
            self.class_attr = 'y'
            self.positive_class_val = 1
            self.sensitive_attrs = ['age']  # privilidged: group 0
        elif dataset_name == 'propublica-violent-recidivism':
            self.dataset_name = 'propublica-violent-recidivism'
            self.class_attr = 'two_year_recid'
            self.positive_class_val = 1
            self.sensitive_attrs = ['race', 'sex']  # privilidged: group 0
        elif dataset_name == 'propublica-recidivism':
            self.dataset_name = 'propublica-recidivism'
            self.class_attr = 'two_year_recid'
            self.positive_class_val = 1
            self.sensitive_attrs = ['race', 'sex']  # privilidged: group 0
        elif dataset_name == 'german':
            self.dataset_name = 'german'
            self.class_attr = 'credit'
            self.positive_class_val = 1
            self.sensitive_attrs = ['sex', 'age']  # privilidged: group 0
        elif dataset_name == 'ricci':  # might take out later
            self.dataset_name = 'ricci'
            self.class_attr = 'Class'
            self.positive_class_val = 1
            self.sensitive_attrs = ['Race']  # privilidged: group 0
        else:
            raise NotImplementedError

    def shuffle(self, random_state=0, attribute=None) -> None:
        self.df = shuffle(self.df, random_state=random_state)
        # important!!
        self.df.reindex()
        # make sure the self.df[:budget] contains at least one instance for each group
        if attribute is not None:
            list_attribute = self.df[attribute].tolist()
            idx0 = [list_attribute.index(elem) for elem in set(list_attribute)]
            idx1 = [i for i in range(self.__len__()) if i not in idx0]
            self.df = self.df.iloc[idx0 + idx1]

    @classmethod
    def load_from_file(cls, fname: Path, dataset_name: str) -> 'Dataset':
        return cls(pd.read_csv(fname), dataset_name)

    def get_group_frequency(self, attribute, num_groups):
        counts = np.zeros((num_groups,))
        for k in range(num_groups):
            counts[k] = (self.df[attribute] == k).sum()
        return counts

    def get_groupwise_accuracy(self, algorithm_name, attribute, group_id, budget=None):
        if budget is None:
            budget = self.__len__()

        s = self.df['score_' + algorithm_name][:budget]
        predicted_y = s > 0.5
        y = self.df[self.class_attr][:budget]

        return np.mean((predicted_y == y)[self.df[attribute][:budget] == group_id] * 1.0)

    def get_accuracy_difference(self, algorithm_name, attribute, group_idx_i, group_idx_j, budget=None):
        """
        Returns accuracy difference of the algorithm between two groups, computed with the top #budget rows.
        """
        if budget is None:
            budget = self.__len__()

        s = self.df['score_' + algorithm_name][:budget]
        predicted_y = s > 0.5
        y = self.df[self.class_attr][:budget]

        if group_idx_i == -1 and group_idx_j == -1:
            raise ValueError("at least one of group id is not -1")

        if group_idx_i == -1:
            acc_i = np.mean((predicted_y == y)[self.df[attribute][:budget] != group_idx_j] * 1.0)
        else:
            acc_i = np.mean((predicted_y == y)[self.df[attribute][:budget] == group_idx_i] * 1.0)

        if group_idx_j == -1:
            acc_j = np.mean((predicted_y == y)[self.df[attribute][:budget] != group_idx_i] * 1.0)
        else:
            acc_j = np.mean((predicted_y == y)[self.df[attribute][:budget] == group_idx_j] * 1.0)

        return acc_i - acc_j

    def get_conditional_metrics_difference(self, algorithm_name, attribute, group_idx_i, group_idx_j,
                                           metric, budget=None):

        if budget is None:
            budget = self.__len__()

        s = self.df['score_' + algorithm_name][:budget]
        predicted_y = s > 0.5
        y = self.df[self.class_attr][:budget]
        ct = self.df[attribute][:budget]

        if group_idx_i == -1 and group_idx_j == -1:
            raise ValueError("at least one of group id is not -1")

        if group_idx_i == -1:
            y_i = y[ct != group_idx_j] * 1.0
            predicted_y_i = predicted_y[ct != group_idx_j] * 1.0
        else:
            y_i = y[ct == group_idx_i] * 1.0
            predicted_y_i = predicted_y[ct == group_idx_i] * 1.0

        if group_idx_j == -1:
            y_j = y[ct != group_idx_i] * 1.0
            predicted_y_j = predicted_y[ct != group_idx_i] * 1.0
        else:
            y_j = y[ct == group_idx_j] * 1.0
            predicted_y_j = predicted_y[ct == group_idx_j] * 1.0

        if metric == 'tpr':
            tpr_i = np.mean(predicted_y_i[y_i == 1])
            tpr_j = np.mean(predicted_y_j[y_j == 1])
            delta = tpr_i - tpr_j
            if isnan(delta):
                return 0
            return delta

        elif metric == 'fnr':
            fnr_i = np.mean(predicted_y_i[y_i == 0])
            fnr_j = np.mean(predicted_y_j[y_j == 0])
            delta = fnr_i - fnr_j
            if isnan(delta):
                return 0
            return delta
        else:
            raise NotImplementedError


code_llo_calibration = '''
# LLO Calibration model
#
# Observed data / constants
# n     total number of instances
# s     n x 1 vector of model scores
# y     n x 1 vector of true outcomes (the label observed; NOT an indicator of accuracy)
#
# Latent variables
# a,b   calibration scaling parameters (scalars)
# p     estimated p( label=1 | Score )
#
# Note:
# Computational issues arise when there are scores that have exact values of 0 or 1
# To make this code run without JAGS errors, it helps to set the minimum score to 0.001 and the maximum score to 0.999
model{
   # Loop over all instances
   for (i in 1:n) {
      p[i] <- ( exp(b) * ( s[i]^a )) / ( exp(b) * ( s[i]^a )+ (1-s[i])^a )
      # Draw a label according to the recalibrated probability
      y[i] ~ dbern( p[i] )
   }
   # Priors -- these will yield perfect calibration p(label=1|s)=s a priori
   a  <- exp( aa )
   b  ~  dnorm(0,0.1)
   aa ~  dnorm(0,0.1)
}
'''

code_beta_binomial = '''
# Beta-binomial model
#
# Observed data / constants
# nl    number of instances with observed labels
# nc    number of categories
# ct    nl x 1 vector of categories 
# s     nl x 1 vector of model scores 
# y     nl x 1 vector of labels
#
# Latent variables
# theta nc x 1  accuracy for each category


data
{
    for (i in 1:nl) {
        # Create an indicator for correctness of the i-th instance
        # Assume that a score > 0.5 will lead to a predicted label of "1"
        iscorrect[i] <- ((s[i] > 0.5) == y[i])
    }
}
model
{
    # Loop over all instances with observed labels
    for (i in 1:nl) {
        # Draw indicator of correctness with probability theta according to category
        iscorrect[i] ~ dbern( theta[ct[i]] )
    }
    # Sample the calibration parameters for each category
    for (j in 1:nc) {
        theta[j] ~ dbeta(2, 2)
    }
}
'''

code_hierarchical_beta_binomial = '''
# Hierarchical Beta-binomial model
#
# Observed data / constants
# nl    number of instances with observed labels
# nc    number of categories
# ct    nl x 1 vector of categories 
# s     nl x 1 vector of model scores 
# y     nl x 1 vector of labels
#
# Latent variables
# theta nc x 1  accuracy for each category
# mu    prior mean accuracy at group level
# c     strength of prior mean at group level

data
{
    for (i in 1:nl) {
        # Create an indicator for correctness of the i-th instance
        # Assume that a score > 0.5 will lead to a predicted label of "1"
        iscorrect[i] <- ((s[i] > 0.5) == y[i])
    }
}
model
{
    # Loop over all instances with observed labels
    for (i in 1:nl) {
        # Draw indicator of correctness with probability theta according to category
        iscorrect[i] ~ dbern( theta[ct[i]] )
    }
    # Sample the calibration parameters for each category
    for (j in 1:nc) {
        theta[j] ~ dbeta(c * mu, c * (1 - mu)) T(0.001,0.999)
    }
    # prior of overall accuracy
    mu ~ dbeta(2, 2) 
    c ~ dnorm( 0 , 1 ) T(0.1,)
}
'''

code_hierarchical_beta_calibration = '''
# Hierarchical Beta Calibration model
#
# Observed data / constants
# n     total number of instances 
# nl    number of instances with observed labels
# nc    number of categories
# ct    n x 1 vector of categories 
# s     n x 1 vector of model scores 
# y     nl x 1 vector of labels
# mumua,mumub,mumuc       prior means at group level
# stda,stdb,stdc          standard deviations at category level
# stdstda,stdstdb,stdstdc standard deviations at group level
#
#
# Latent variables
# a,b,c   nc x 1 vector of calibration scaling parameters for each category
# p       n x 1 vector with estimated p( label=1 | Score )
# pc      n x 1 vector with estimated accuracy 
# theta   nc x 1 vector with mean of pointwise accuracy over all instances that belong to a category
#
# Note:
# Computational issues arise when there are scores that have exact values of 0 or 1
# To make this code run without JAGS errors, it helps to set the minimum score to 0.001 and the maximum score to 0.999 
data {
   for (i in 1:n) {
      # Log transform the scores  
      logs1[i] <- log( s[i]   )
      logs0[i] <- log( 1-s[i] )
      # Create a boolean variable for predicted label associated with the i-th instance
      # Assume that a score > 0.5 will lead to a predicted label of "1"
      predl[i] <- ( s[i]>0.5 ) 
      # Create indicator matrix iscat[j,i] where iscat[j,i]=1 when the i-th instance belongs to category j
      for (j in 1:nc) {
         iscat[j,i] <- ( ct[i] == j ) 
      }
   }
   for (i in 1:nl) {
      # Create an indicator for correctness of the i-th instance
      iscorrect[i] <- (predl[i] == y[i])
   }
   for (i in (nl+1):n) {
      # We don't need the correctness labels for these instances, but we add them to avoid
      # JAGS from compilation errors
      iscorrect[i] <- 0
   }
}
model{
   # Loop over all instances 
   for (i in 1:n) {
      # The Beta calibration step (avoid log odds larger than 15) 
      logoddsp[i] <- min( c[ct[i]] + exp( a[ct[i]] )*logs1[i] - exp( b[ct[i]] )*logs0[i] , 15 )
      # p is the model probability of label "1" given this score
      p[i] <- exp( logoddsp[i] ) / ( exp( logoddsp[i] ) + 1 )
      # pc is the pointwise probability that the instance is classified correctly
      pc[i] <- p[i]*predl[i] + (1-p[i])*(1-predl[i])
      # pc2 is the pointwise probability of classifying an instance correctly 
      # For the labeled instances, we take observed indicator of correctness
      pc2[i] <- ifelse( i<=nl , iscorrect[i] , pc[i] ) 
   }
   # Loop over all instances with observed labels
   for (i in 1:nl) {  
      # Draw a label according to the LLO probability
      y[i] ~ dbern( p[i] )              
   }  
   # Sample the calibration parameters for each category
   for (j in 1:nc) {     
      a[j] ~ dnorm( mua , 1/(stda^2))
      b[j] ~ dnorm( mub , 1/(stdb^2))
      c[j] ~ dnorm( muc , 1/(stdc^2))
   }
   # Compute the mean accuracy per category
   for (j in 1:nc) {
      # The inner product (inprod) computes the sum of the accuracies for all instances belonging to category j, 
      # and the denominator computes the sum of the number of instances belonging to category j
      theta[j] <- inprod( pc2 , iscat[j,] ) / sum( iscat[j,] )      
   }
   # Normal priors on the group means   
   mua ~  dnorm( mumua , 1/(stdmua^2))
   mub ~  dnorm( mumub , 1/(stdmub^2))  
   muc ~  dnorm( mumuc , 1/(stdmuc^2)) 
   # Half-normal (truncated) priors for the standard deviations   
   stda ~ dnorm(0, 1/(stdstda^2) ) T(0,)
   stdb ~ dnorm(0, 1/(stdstdb^2) ) T(0,) 
   stdc ~ dnorm(0, 1/(stdstdc^2) ) T(0,)  
}
'''

code_hierarchical_llo_calibration = '''
# Hierarchical LLO Calibration model, arrows from p_i to theta
#
# Observed data / constants
# n     total number of instances 
# nl    number of instances with observed labels
# nc    number of categories
# ct    n x 1 vector of categories 
# s     n x 1 vector of model scores 
# y     nl x 1 vector of labels
# mumua,mumub     prior means at group level
# stda,stdb       standard deviations at category level
# stdstda,stdstdb standard deviations at group level
#
#
# Latent variables
# a,b   nc x 1 vector with calibration scaling parameters for each category
# p     n x 1 vector with estimated p( label=1 | Score )
# pc    n x 1 vector with estimated accuracy 
# theta nc x 1 vector with mean of pointwise accuracy over all instances that belong to a category
#
# Note:
# Computational issues arise when there are scores that have exact values of 0 or 1
# To make this code run without JAGS errors, it helps to set the minimum score to 0.001 and the maximum score to 0.999 
data {
   for (i in 1:n) {
      # Transform the scores to log odds 
      logoddss[i] <- log( s[i] / ( 1-s[i] ))
      # Create a boolean variable for predicted label associated with the i-th instance
      # Assume that a score > 0.5 will lead to a predicted label of "1"
      predl[i] <- ( s[i]>0.5 ) 
      # Create indicator matrix iscat[j,i] where iscat[j,i]=1 when the i-th instance belongs to category j
      for (j in 1:nc) {
         iscat[j,i] <- ( ct[i] == j ) 
      }
   }
   for (i in 1:nl) {
      # Create an indicator for correctness of the i-th instance
      iscorrect[i] <- (predl[i] == y[i])
   }
   for (i in (nl+1):n) {
      # We don't need the correctness labels for these instances, but we add them to avoid
      # JAGS from compilation errors
      iscorrect[i] <- 0
   }
}
model{
   # Loop over all instances 
   for (i in 1:n) {
      # The LLO model is a linear transformation of the log odds 
      # Place an upper bound of 15 to avoid numerical problems with conversion to probability
      logoddsp[i] <- min( b[ct[i]] + exp( a[ct[i]] ) * logoddss[i] , 15 )
      # p is the model probability of label "1" given this score
      p[i] <- exp( logoddsp[i] ) / ( exp( logoddsp[i] ) + 1 )
      # pc is the pointwise probability that the instance is classified correctly
      pc[i] <- p[i]*predl[i] + (1-p[i])*(1-predl[i])
      # pc2 is the pointwise probability of classifying an instance correctly 
      # For the labeled instances, we take observed indicator of correctness
      pc2[i] <- ifelse( i<=nl , iscorrect[i] , pc[i] ) 
   }
   # Loop over all instances with observed labels
   for (i in 1:nl) {  
      # Draw a label according to the LLO probability
      y[i] ~ dbern( p[i] )              
   }  
   # Sample the calibration parameters for each category
   for (j in 1:nc) {
      b[j] ~ dnorm( mub , 1/(stdb^2))
      a[j] ~ dnorm( mua , 1/(stda^2))
   }
   # Compute the mean accuracy per category
   for (j in 1:nc) {
      # The inner product (inprod) computes the sum of the accuracies for all instances belonging to category j, 
      # and the denominator computes the sum of the number of instances belonging to category j
      theta[j] <- inprod( pc2 , iscat[j,] ) / sum( iscat[j,] )      
   }
   # Normal priors on the group means    
   mua ~  dnorm( mumua , 1/(stdmua^2)) 
   mub ~  dnorm( mumub , 1/(stdmub^2))
   # Half-normal (truncated) priors for the standard deviations
   stda ~ dnorm(0, 1/(stdstda^2) ) T(0,)
   stdb ~ dnorm(0, 1/(stdstdb^2) ) T(0,) 
}

'''
