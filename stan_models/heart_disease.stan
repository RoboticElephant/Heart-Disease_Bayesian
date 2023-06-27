// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0> N;       // observations
  int<lower=0> N_t;     // test oberservations
  int<lower=1> K;       // coefficients
  
  matrix[N, K] X;
  
  matrix[N_t, K] X_test;
  array[N] int<lower=0, upper=1> y;
  vector[N_t] y_test;
}

parameters {
  real alpha;    // intercept
  vector[K] betas;
}

model {
  alpha ~ uniform(-50, 50);
  // alpha ~ normal(0, 10);
  betas ~ normal(0, 10);
  y ~ bernoulli_logit(alpha + X * betas);
}

generated quantities {
  vector[N_t] y_pred;
  
  for (n in 1:N_t) {
    y_pred[n] = bernoulli_logit_rng(alpha + X_test[n] * betas);
  }
  
  // In order to calculate Bayesian R2:
  // http://www.stat.columbia.edu/~gelman/research/unpublished/bayes_R2.pdf
  real BR2;
  BR2 = variance(y_pred) / (variance(y_pred) + variance(y_test - y_pred));
}
