// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0> N;       // observations
  
  vector[N] age;
  vector[N] sex;  // M: 0, F: 1
  vector[N] cpt;
  vector[N] rbp;
  vector[N] c_;
  vector[N] fps;
  vector[N] recg;
  vector[N] mhr;
  vector[N] ea;
  vector[N] op;
  vector[N] sts;
  
  array[N] int<lower=0, upper=1> y;
}

parameters {
  real b0;      //intercept
  real b1;
  real b2;
  real b3;
  real b4;
  real b5;
  real b6;
  real b7;
  real b8;
  real b9;
  real b10;
  real b11;
}

model {
  b0 ~ normal(0, 10);
  b1 ~ normal(0, 10);
  b2 ~ normal(0, 10);
  b3 ~ normal(0, 10);
  b4 ~ normal(0, 10);
  b5 ~ normal(0, 10);
  b6 ~ normal(0, 10);
  b7 ~ normal(0, 10);
  b8 ~ normal(0, 10);
  b9 ~ normal(0, 10);
  b10 ~ normal(0, 10);
  b11 ~ normal(0, 10);
  
  target += bernoulli_logit_lupmf(y | b0 + age * b1 + sex * b2 + cpt * b3 +
                                      rbp * b4 + c_ * b5 + fps * b6 +
                                      recg * b7 + mhr * b8 + ea * b9 +
                                      op * b10 + sts * b11);
}
