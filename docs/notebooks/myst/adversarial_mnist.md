---
jupytext:
  formats: ipynb///ipynb,myst///md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
from traitlets.config.manager import BaseJSONConfigManager
from pathlib import Path
path = Path.home()/".jupyter"/"nbconfig"
cm = BaseJSONConfigManager(config_dir=str(path))
cm.update(
    "rise",
    {
        "transition": "linear",
        "start_slideshow_at": "selected",
        "scroll": True,
        "enable_chalkboard": False,  # scrollable slides are only available without chalkboard
        "header": "<div class=logo><img src=images/gurobi-light.svg alt=Gurobi></div>"
    }
)
```

+++ {"slideshow": {"slide_type": "subslide"}}

# Adversarial Machine Learning with Gurobi

In this example, we show how to use Gurobi Machine Learning to construct an
adversarial example for a trained neural network.

We use the MNIST handwritten digit database (http://yann.lecun.com/exdb/mnist/).

+++ {"slideshow": {"slide_type": "slide"}}

We are given a (small) trained neural network and one well classified
example $\bar x$. Our goal is to construct another example $x$ _close to_ $\bar
x$ that is classified with a different label.

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
  slide_type: fragment
---
# Load the trained network and the examples
mnist_data = load("../../../tests/predictors/mnist__mlpclassifier.joblib")
X = mnist_data["data"]
nn = mnist_data["predictor"]
# Choose an example
exampleno = 26
example = X[exampleno : exampleno + 1, :]
```

+++ {"slideshow": {"slide_type": "subslide"}}

The example is a grayscale image of $28
\times 28$ ($=784$) pixels.

The output is a vector of length 10 (each entry
corresponding to a digit).

We denote the output vector by $y$. The image is
classified according to the largest entry of $y$.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
plt.imshow(example.reshape((28, 28)), cmap="gray")
print(f"Predicted label {nn.predict(example)}")
```

+++ {"slideshow": {"slide_type": "slide"}}

The neural network has two hidden layers of 50 neurons each

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
nn
```

+++ {"slideshow": {"slide_type": "subslide"}}

We use `predict_proba` to get the weight for each label given by the neural
network.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
ex_prob = nn.predict_proba(example)

print(f"Label weights for training example:\n {ex_prob}")
```

+++ {"slideshow": {"slide_type": "subslide"}}

For this training example, coordinate $l=4$ of the output vector is
the one with the largest value giving the correct label. We pick a coordinate
corresponding to another label $w=9$, and we want the difference between
$y_9 - y_4$ to be as large as possible.

If we find a $x$ where this difference is positive, then $x$ is a
counter-example that receives a different label. If instead we show that
the difference is negative, no such counter-example exists.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
right_label = 4
wrong_label = 9
```

+++ {"slideshow": {"slide_type": "subslide"}}

We define the neighborhood using the $l1-$norm $|| x - \bar x
||_1$, it is defined by a fixed parameter $\delta$.

+++ {"slideshow": {"slide_type": "fragment"}}

If we denote by $g$ the prediction function of the neural network. Our full
optimization model reads:

$$ \begin{aligned} &\max y_w - y_l \\
&\text{subject to:}\\
&|| x - \bar x ||_1 \le \delta,\\
& y = g(x). \end{aligned} $$


Note that our model is inspired by <cite data-cite="fischetti_jo_2018">Fischet al.
(2018)</cite>.

+++ {"slideshow": {"slide_type": "slide"}}

## Optimization Model

Now build the model with gurobipy

+++ {"slideshow": {"slide_type": "slide"}}

Create a matrix variable `x` corresponding to input of the
neural network and a matrix variable `y` corresponding to the output of the
example we picked and the shape of the return value of `predict_proba`.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
m = gp.Model()
delta = 5

x = m.addMVar(example.shape, lb=0.0, ub=1.0, name="x")
y = m.addMVar(ex_prob.shape, lb=-gp.GRB.INFINITY, name="y")
```

+++ {"slideshow": {"slide_type": "subslide"}}

Set the objective to maximize the difference between the
_wrong_ label and the _right_ label.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
m.setObjective(y[0, wrong_label] - y[0, right_label], gp.GRB.MAXIMIZE)
```

+++ {"slideshow": {"slide_type": "slide"}}

Crete additional variables to model the $l1-$norm constraint.

For each pixel in the image, we need to model the absolute difference between $x$
and $\bar x$.

Denote by $\eta$ the matrix variable measuring the absolute difference.

+++ {"slideshow": {"slide_type": "subslide"}}

The $l1-$norm constraint is formulated with:

$$ \eta \ge x - \bar x \\
\eta \ge \bar x - x \\
\sum \eta \le \delta $$

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
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
  slide_type: fragment
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
  slide_type: fragment
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

 We use Gurobi parameters to limit the optimization to limit the optimization to answering those
 questions.

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
  slide_type: fragment
---
plt.imshow(x.X.reshape((28, 28)), cmap="gray")
print(f"Solution is classified as {nn.predict(x.X)}")
```

+++ {"nbsphinx": "hidden", "slideshow": {"slide_type": "slide"}}

<div class="titlepage">

# Thank you!

<br>
<br>
<br>

Find out more [Gurobi Machine Learning](https://gurobi-optimization-gurobi-machine-learning.readthedocs-hosted.com/en/stable/)

</div>

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
m.write('ReLU.lp')
```

```{code-cell} ipython3

```
