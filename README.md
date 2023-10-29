# Decision tree parallelization in Go

The goal is to design and implement parallelism in the tree growth step of the decision tree algorithm. To provide some background on our version of the decision tree algorithm, we can imagine that we are given some dataset with a categorical target feature that we wish to predict, by using a subset of all non-target features in a certain way. Assume that all features are categorical, then in the best case it is possible that there exists one perfect feature which can help you predict the target feature value without peeking at other features. Normally, however, we would need a combination of features to be certain of our prediction. For more details about decision trees, please see Section 18.3 of ```Stuart Russell and Peter Norvig, Artificial Intelligence: A Modern Approach (Third edition), Prentice Hall (2009).```

At each non-leaf node, the decision tree algorithm needs to compare across all possible features before choosing a best feature to use at a non-leaf node. We can think of the calculation involved as counting how the data would be separated if the node were to use a certain feature. The goal is to split the dataset as “unevenly” as possible. Since the analysis for each potential feature only involves the feature itself, there is room for parallelization.

# Implementation details

**Assumptions of the decision tree algorithm**

We assume that all features of the dataset are categorical, and that the target feature is binary. This enables us to focus more on the parallelism rather than data cleaning and is consistent with the example in Figure 18.2 of Russell and Norvig.

**Criteria for choosing the best feature**

Possible criteria include the Gini index and the information entropy. In this project, we use the Gini index defined as follows:
```math
$$\begin{align}
Gini_{f_i,v_j} = 1 - P^2(Target=Y|f_i=v_j) - P^2(Target=N|f_i=v_j)
\end{align}$$
```
where $f_i$ is the i-th feature and $v_j$ is the j-th unique value of that feature. Then, we take a weighted average of the Gini index of feature i across its unique values j and return the result to compare against the results for other features. The feature with the best average Gini index will be used in the current node of the decision tree. The parallelization of Gini index calculation is the core of the parallel version of our program.

**Overall structure**

Sequential version: The builder of the tree simply grows the tree level by level, where the calculations for each node take place sequentially, in other words:
```
for each level of the tree:
  for each node in the level:
    for each potential feature for the node:
      calculate Gini index
    compare and choose the best feature
```
Parallel version: parallelization is achieved by using futures backed by thread pools, and takes place in two steps. The high-level picture is that we still complete the entire level of the tree before moving on to the next level, but this time, nodes at the same level are processed in parallel, and further, the Gini index calculations for each node also take place in parallel.
In other words, there are two steps of fanning out. In the first step, the ```growTree``` method spawns a future for each node in the current set of leaves, to find the best feature to partition that subtree. Note that for this, we need to compare across the Gini indices for all candidate features. We call this step the “aggregator task.” But before an aggregator can do the comparison, someone needs to calculate the Gini indices. For this the aggregators fan out again by generating multiple “evaluation tasks.” This fanning out constitutes the second step, where the Gini index calculations take place. Each evaluation task is a future (discussed in the next section). It can inform the aggregator task that spawned it when it is done. Eventually, each aggregator task will hear back from the evaluation tasks and be able to proceed and compare across features.

**Advanced Feature**

In our program, each evaluation task is a future, and each aggregator is also a future. They are both implemented by a done channel. Take the evaluation future object for example, when the aggregator fans out the evaluation tasks, it does not need to worry about when the evaluation will return. The aggregator simply needs to initialize an evaluation task object and call the wait method, and eventually compare the values returned by all the future objects it has spawned.

# Code availability
Only part of the code is made publicly available for demonstration purposes, and may be shared fully upon request.
