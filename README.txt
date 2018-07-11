

----------------------------
Flexible Modulated Poisson
----------------------------

**Description:** 
This code repository provides a MATLAB implementation 
of the flexible over-dispersed Poisson model in 

"Dethroning the Fano Factor: a flexible, model-based 
approach to partitioning neural variability" published 
in Neural Computation, 2018.  [[link]](https://www.mitpressjournals.org/doi/full/10.1162/neco_a_01062)

The models fit in this code base are modulated Poisson
models where the data is described by the generative
model

    r = Poiss(f(x+n))

where x is the stimulus-related response, n is trial
dependent noise that is normally distributed as
 
    n ~ N(0,sigma^2)

and f() is a nonlinearity. The three possible 
nonlinearities in thiscode package are the 
exponential, soft-rectification and 
power-soft-rectification functions. The model fits
the per-stimulus values x, the noise variance sigma^2
and the parameters of the nonlinearity using a Laplace
approximation method. 

**Code desctiption:** 

The main file to use in this repository is 

     negLfun_latentPoiss.m

which is a function that takes in a parameter set, 
event count data, and a handle to a link function and
returns the negative log-likelihood value for that data.
The intended use is to use this function in conjunction
with fmincon.m or a similar optimization function in 
order to optimize the negative log-likelihood with 
respect to the parameter set. 

**Usage**

* Launch matlab and cd into the `flexibleModulatedPoisson` directory.
 
* Examine the demo script `demo.m` for an example on the 
appropriate use of this function. `demo.m` generates a set 
of synthetic data that mimics the format of the data used
in the related paper. It then demonstrates the correct use
of the `negLfun_latentPoiss.m`  function in fitting the 
three models included in the package. The results are then 
compared, showing how to extract the fit parameters correctly
from the fminunc output, as some of the parameters are fit
under an exponential transformation to ensure positivity.

**Code download:** 

* **Clone**: clone the repository from github: ```git clone https://github.com/adamshch/flexibleModulatedPoisson.git```

----------------------------
