

----------------------------
Flexible Modulated Poisson
----------------------------

This code repository provides a MATLAB implementation 
of the flexible over-dispersed Poisson model in 
"Dethroning the Fano Factor: a flexible, model-based 
approach to partitioning neural variability" published 
in Neural Computation, 2018.


The main file to use in this repository is 

     negLfun_latentPoiss.m

which is a function that takes in a parameter set, 
event count data, and a handle to a link function and
returns the negative log-likelihood value for that data.
The intended use is to use this function in conjunction
with fmincon.m or a similar optimization function in 
order to optimize the negative log-likelihood with 
respect to the parameter set. 


----------------------------
EOF
----------------------------
