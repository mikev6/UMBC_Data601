# Model Evaluation and Model Selection


[From Introduction to Statistical Learning with R (ISLR)](https://faculty.marshall.usc.edu/gareth-james/ISL/ISLR%20Seventh%20Printing.pdf)


Read 5.1: Cross Validation p175 - p187

__Some points to keep in your mind as you're reading__

- Make sure you really understand the difference between `test error rate` and `train error rate`.

- What is the challenge of estimating the `test error rate`?

- Can we use `train error rate` as an estimate for `test error rate`? Why (Why not)?

- Explain `validation set approach`.

- What is the difference between `train set` and `validation set`?

- Do you fit the model on `validation set`?

- Understand figure 5.2

- What is `validation error rate`? Does `validation error rate` overestimate or underestimate `test error rate`?

- What weakness of `validation set approach` is addressed by `Leave-One-Out Cross Validation` (LOOC) and how?

- What is the major advantage of the LOOC with respect to `validation set approach`?

- What is the major disadvantage of the LOOC with respect to `validation set approach`?

- What is magic formula in the context of LOOC? Why is it magical?

- Explain `k-fold cross validation` approach.

- Compare the advantages and disadvantages of `k-fold cross validation` with respect to previous approached.

- What might be some possible goals when we perform cross-validation?

- Compare LOOC and cross-validation with respect to concerns about `bias`?

- Which one of the approaches between LOOC and cross-validation has smaller variance? Do you understand why?

Read 6.2: Shrinkage Methods p214 - p230

- How Ridge regression adjusts the RSS formula and why?

- What is `shrinkage penalty` and what is it role in formula 6.5?

- What happens if you set $\lambda$ very big? What if very small?

- Explain $\ell_{2}$-norm. Consider the vector $v = (3,4,5)$ what would be the $\ell_{2}$-norm of this vector?

- Explain `scale equivariance`. Is Ridge algorithm `scale equivariant`?

- Compare Ridge regression with respect to least squares approach. Which one has less variance? Which one would have more bias?

- When does Ridge regression work best?

- Compare Lasso and Ridge regression.

- Explain what is a sparse model.

- You can stop at p220 in your first reading as things get a little bit technical after that.
