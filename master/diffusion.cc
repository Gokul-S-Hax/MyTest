#include <mkl.h>
#include "distribution.h"


//vectorize this function based on instruction on the lab page
int diffusion(const int n_particles, 
              const int n_steps, 
              const float x_threshold,
              const float alpha, 
              VSLStreamStatePtr rnStream) {
  int n_escaped=0;
  float positions[n_particles];
  for (int i = 0; i < n_steps; i++) {
    float x = 0.0f;
	float rn[n_particles];
	vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD,rnStream, n_particles, rn, -1.0, 1.0);
	#pragma omp simd reduction(+: positions)
    for (int j = 0; j < n_particles; j++) {
		if(i == 0)positions[j] = 0;
      positions[j] += dist_func(alpha, rn[j]); 
	  if(i == n_steps - 1)
		if(positions[j]>x_threshold)n_escaped++;
    }
  }
  return n_escaped; 
}