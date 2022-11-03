---
jupytext:
  formats: ipynb///ipynb,myst///md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"nbsphinx": "hidden", "slideshow": {"slide_type": "skip"}}

<div class="alert alert-warning">
warning

The ipynb version of this notebook should not be manually edited.
If you want to make modification please modify the .md version

</div>

+++ {"slideshow": {"slide_type": "skip"}}

<div class="alert alert-warning">
warning

This code is in a pre-release state. It may not be fully functional and breaking changes
can occur without notice.

</div>

+++ {"slideshow": {"slide_type": "slide"}}

# Adversarial Machine Learning

In this example, we show how to use Gurobi Machine Learning to construct an
adversarial example for a trained neural network.

We use the MNIST handwritten digit database (http://yann.lecun.com/exdb/mnist/)
for this example.

+++ {"slideshow": {"slide_type": "subslide"}}

For this problem, we are given a trained neural network and one well classified
example $\bar x$. Our goal is to construct another example $x$ _close to_ $\bar
x$ that is classified with another label.

For the hand digit recognition problem, the input is a grayscale image of $28
\times 28$ ($=784$) pixels and the output is a vector of length 10 (each entry
corresponding to a digit). We denote the output vector by $y$. The image is
classified according to the largest entry of $y$.

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
import gurobipy as gp
import numpy as np
from joblib import load
from matplotlib import pyplot as plt

from gurobi_ml.sklearn import add_mlp_regressor_constr
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
# Load the trained network and the examples
mnist_data = load("../../../tests/predictors/mnist__mlpclassifier.joblib")
X = mnist_data["data"]
nn = mnist_data["predictor"]
# Choose an example
exampleno = 26
example = X[exampleno : exampleno + 1, :]
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
pixels = example.reshape((28, 28))
plt.imshow(pixels, cmap="gray")
plt.show()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
print(f"Predicted label {nn.predict(example)}")
```

+++ {"slideshow": {"slide_type": "subslide"}}

We use `predict_proba` to get the weight for each label given by the neural
network.

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
ex_prob = nn.predict_proba(example)
ex_prob
```

+++ {"slideshow": {"slide_type": "subslide"}}

For this training example, coordinate $l=4$ of the output vector is
the one with the largest value giving the correct label. We pick a coordinate
corresponding to another label, denoted $w$, and we want the difference between
$y_w - y_l$ to be as large as possible.

If we can find a solution where this difference is positive, then $x$ is a
counter-example that will receive a different label. If instead we can show that
the difference is non-positive, no such counter-example exists.

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
sorted_labels = np.argsort(ex_prob)[0]
right_label = sorted_labels[-1]
wrong_label = sorted_labels[-2]
print(f"The wrong label we choose is {wrong_label}")
```

+++ {"slideshow": {"slide_type": "subslide"}}

We define the neighborhood using the $l1-$norm $|| x - \bar x
||_1$. The size of the neighborhood is defined by a fixed parameter $\delta$. We
want

$$ || x - \bar x ||_1 \le \delta. $$

+++ {"slideshow": {"slide_type": "subslide"}}

If we denote by $g$ the prediction function of the neural network. Our full
optimization model reads:

$$ \begin{aligned} &\max y_w - y_l \\
&\text{subject to:}\\
&|| x - \bar x ||_1 \le \delta,\\
& y = g(x). \end{aligned} $$


Our model is inspired by <cite data-cite="fischetti_jo_2018">Fischet al.
(2018)</cite>.

+++ {"slideshow": {"slide_type": "slide"}}

## Optimization Model

Now build the model with gurobipy

+++ {"slideshow": {"slide_type": "slide"}}

Create a matrix variable `x` corresponding to the new input of the
neural network we want to compute and a `y` variables for the output of the
neural network. Those variables should have respectively the shape of the
example we picked and the shape of the return value of `predict_proba`.

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
m = gp.Model()
delta = 5

x = m.addMVar(example.shape, lb=0.0, ub=1.0, name="x")
y = m.addMVar(ex_prob.shape, lb=-gp.GRB.INFINITY, name="y")
```

+++ {"slideshow": {"slide_type": "slide"}}

Set the objective to maximize the difference between the
_wrong_ label and the _right_ label.

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
m.setObjective(y[0, wrong_label] - y[0, right_label], gp.GRB.MAXIMIZE)
```

+++ {"slideshow": {"slide_type": "slide"}}

Crete additional variables to model the $l1-$norm constraint.

For each pixel in the image, we need to model the absolute difference between $x$
and $\bar x$.

Denote by $\eta$ the matrix variable measuring the absolute difference.

+++ {"slideshow": {"slide_type": "slide"}}

The $l1-$norm constraint is formulated with:

$$ \eta \ge x - \bar x \\
\eta \ge \bar x - x \\
\sum \eta \le \delta $$

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
# Bound on the distance to example in norm-1
eta = m.addMVar(example.shape, lb=0, ub=1, name="abs_diff")
m.addConstr(eta >= x - example)
m.addConstr(eta >= -x + example)
m.addConstr(eta.sum() <= delta);
```

+++ {"slideshow": {"slide_type": "slide"}}

Finally, insert the neural network in the `gurobipy` model to link `x` and
`y`.

Note that the neural network is trained for classification with a `"softmax"` activation in
the last layer. But in this model we are using the network without activation in
the last layer.

For this reason, we change manually the last layer activation before adding the
network to the Gurobi model.

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
# Change last layer activation to identity
nn.out_activation_ = "identity"
# Code to add the neural network to the constraints
pred_constr = add_mlp_regressor_constr(m, nn, x, y)

# Restore activation
nn.out_activation_ = "softmax"
```

+++ {"slideshow": {"slide_type": "subslide"}}

The model should be complete. We print the statistics of what was added to
insert the neural network into the optimization model.

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
pred_constr.print_stats()
```

+++ {"slideshow": {"slide_type": "slide"}}

## Solving the model

We now turn to solving the optimization model. Solving the adversarial problem,
as we formulated it above, doesn't actually require computing a provably optimal
solution. Instead, we need to either:

   - find a feasible solution with a positive objective cost (i.e. a
     counter-example), or
   - prove that there is no solution of positive cost (i.e. no counter-example
     in the neighborhood exists).

 We can use Gurobi parameters to limit the optimization to answer those
 questions: setting
 [BestObjStop](https://www.gurobi.com/documentation/current/refman/bestobjstop.html#parameter:BestObjStop)
 to 0.0 will stop the optimizer if a counter-example is found, setting
 [BestBdStop](https://www.gurobi.com/documentation/current/refman/bestobjstop.html#parameter:BestBdStop)
 to 0.0 will stop the optimization if the optimizer has shown there is no
 counter-example.

We set the two parameters and optimize.

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
m.Params.BestBdStop = 0.0
m.Params.BestObjStop = 0.0
m.optimize()
```

+++ {"slideshow": {"slide_type": "slide"}}

## Results

Normally, for the example and $\delta$ we chose, a counter example that gets the
wrong label is found. We finish this notebook by plotting the counter example
and printing how it is classified by the neural network.

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
pixels = x.X.reshape((28, 28))
plt.imshow(pixels, cmap="gray")
plt.show()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
print(f"Solution is classified as {nn.predict(x.X)}")
```

+++ {"nbsphinx": "hidden", "slideshow": {"slide_type": "slide"}}

Copyright © 2022 Gurobi Optimization, LLC
