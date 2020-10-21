## Getting started
To setup virtual environment install dependencies in `requirements.txt`:
```
conda create -n bayesian-fairness python=3.7
source activate bayesian-fairness
pip install -r requirements.txt
```


## Datasets
- Download .csv files for 6 datasets used in the paper form `data` directory

## Preprocess and benchmark
- Preprocess with `python fairness_comparison/preprocess.py`
- Train classifers and write predicted scores to file with `python fairness_comparison/benchmark.py`

## Train calibration models

Our implementation supports 4 different methods to provide estimates about groupwise fairness metrics ($\Delta$ Accuracy, $\Delta$ TPR, $\Delta$ FNR, etc), including:
- hierarchical beta binomial model: `src/hierarchical_beta_binomial.py`
- hierarchical llo calibration: `src/hierarchical_llo_calibration.py`
- beta binomial: `src/beta_binomial.py`
- hierarchical beta calibration: `src/hierarchical_beta_calibration.py`

We provide an example to train calibration models on `adult` data with sensitive attribute `sex`.
```
# navigate to source code directory
cd src

# define the list of methods
declare -a MethodArray=("hierarchical_beta_binomial" "hierarchical_llo_calibration" "beta_binomial" "hierarchical_beta_calibration")

# train calibration models with different values of n_L, 100 independent runs
for method in "${MethodArray[@]}"
do
    for n in {2,5,10,20,40,60,80,100,200,400,600,800,1000,10054}
    do
        for i in {0..100}
        do
            python $method.py -dataset adult -attribute sex -num_groups 2 -budget $n -run_id $i
        done
    done
done
```

## Infer fairness metrics

To infer fairness metrics including $\Delta$ TPR and $\Delta$ FNR  on `adult` data with sensitive attribute `sex`, $n_L$=100:
```
python hierarchical_beta_conditional_metrics.py  -dataset adult -attribute sex -num_groups 2 -num_runs 100 -budget 100
```



## To reproduce the results reported in the paper and supplements
- `cd src`
- `bash script`