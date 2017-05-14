#ifndef _AVX_H_
#define _AVX_H_

void avx_add(double *, double *);
void avx_mac(double *, double *);

// Mutex crap
extern pthread_mutex_t runtime_mutex;
#endif
