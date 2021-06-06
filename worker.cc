#include <mkl.h>
#include <hbwmalloc.h>


//implement scratch buffer on HBM and compute FFTs, refer instructions on Lab page
void runFFTs( const size_t fft_size, const size_t num_fft, MKL_Complex8 *data, DFTI_DESCRIPTOR_HANDLE *fftHandle) {
#pragma omp parallel for
  for(size_t i = 0; i < num_fft; i++) {
	const long buff_size = 1000;
	MKL_Complex8 *buff;
	hbw_posix_memalign((void**) &buff, 4096, sizeof(MKL_Complex8)*buff_size);
	&buff = data[i*fft_size];
    DftiComputeForward (*fftHandle, buff);
  }
}