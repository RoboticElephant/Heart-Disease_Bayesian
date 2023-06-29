// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0> N;       // observations
  int<lower=0> N_obs;   // observed variables
  int<lower=0> N_mis;   // missing variables
  array[N_obs] int<lower=1, upper=N_obs + N_mis> ix_obs;  // index of obs x's
  array[N_mis] int<lower=1, upper=N_obs + N_mis> ix_mis;  // index of mis x's
  
  int<lower=1> K;       // predictor variables (reduced by one for Cholesterol)
  
  matrix[N, K] X;       // The X's without the imputed column (Cholesterol)
  vector[N_obs] x_obs;  // Actual x values that were observed
  array[N] int<lower=0, upper=1> y;
}

parameters {
  real alpha;           // intercept
  vector[K] betas;      // coefficients
  real b_chol;          // Cholesterol coefficient
  vector[N_mis] x_mis;  // Missing values for cholesterol
  
  real mu;
  real<lower=0> sigma;
}

transformed parameters {
  vector[N] X_chol;        // Imputed Cholesterol values
  X_chol[ix_obs] = x_obs;  // Observed Cholesterol values
  X_chol[ix_mis] = x_mis;  // Missing Cholesterol values
}

model {
  alpha ~ normal(0, 100);
  betas ~ normal(0, 100);  // Non-informative prior
  b_chol ~ normal(200, 100); // Non-informative prior

  mu ~ normal(300, 50);    // Non-informative prior, but expecting to be around the mean of dataset
  sigma ~ normal(0, 10);
  X_chol ~ normal(mu, sigma);  // Impute the missing X values
  
  y ~ bernoulli_logit(alpha + X * betas + X_chol * b_chol);
}
