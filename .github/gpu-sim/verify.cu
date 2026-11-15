// Device-side check for the CUDA kernels in src/kernels.cu.
//
// The NVCC build of bwbench only prints bandwidth numbers (check() in main.c
// is compiled out), so this runs every kernel on the device, copies the
// arrays back and compares them with values computed on the host. It is
// linked against the objects from `make TOOLCHAIN=NVCC`.
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
#include "cli.h"
#include "kernels.h"
}

static int failures = 0;

static void expect(const char *what, const double *v, size_t n, double want)
{
  for (size_t i = 0; i < n; i++) {
    if (fabs(v[i] - want) > 1e-12) {
      printf("FAIL %s: [%zu] = %g, expected %g\n", what, i, v[i], want);
      failures++;
      return;
    }
  }
  printf("ok   %s == %g\n", what, want);
}

int main(int argc, char **argv)
{
  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
    printf("FAIL: no CUDA device found\n");
    return 1;
  }
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("device: %s\n", prop.name);
  if (argc > 1 && !strstr(prop.name, argv[1])) {
    printf("FAIL: expected a device matching '%s'\n", argv[1]);
    return 1;
  }

  // The kernels take an int scalar internally, so use an integral one.
  const double s = 3.0;
  const size_t n = 1 << 18;
  const size_t pad = 4096; // sum() reads one block beyond n
  double *a, *b, *c, *d;

  allocateArrays(&a, &b, &c, &d, n + pad);
  initArrays(a, b, c, d, n + pad); // a=2 b=2 c=0.5 d=1
  double *h = (double *)malloc((n + pad) * sizeof(double));
  if (!h) {
    return 1;
  }
#define CHECK(name, ptr, want)                                                           \
  do {                                                                                   \
    cudaMemcpy(h, ptr, n * sizeof(double), cudaMemcpyDeviceToHost);                      \
    expect(name, h, n, want);                                                            \
  } while (0)

  CHECK("initArrays a", a, 2.0);
  CHECK("initArrays d", d, 1.0);
  init(b, s, n);
  CHECK("init b", b, 3.0);
  copy(c, a, n);
  CHECK("copy c", c, 2.0);
  update(a, s, n);
  CHECK("update a", a, 6.0);
  triad(a, b, c, s, n);
  CHECK("triad a", a, 9.0); // b + s*c
  daxpy(a, b, s, n);
  CHECK("daxpy a", a, 18.0); // a + s*b
  striad(a, b, c, d, n);
  CHECK("striad a", a, 5.0); // b + d*c
  sdaxpy(a, b, c, n);
  CHECK("sdaxpy a", a, 11.0); // a + b*c
  CHECK("b untouched", b, 3.0);
  CHECK("c untouched", c, 2.0);
  CHECK("d untouched", d, 1.0);

  // sum() leaves one partial sum per block in a[0..blocks-1]
  initArrays(a, b, c, d, n + pad);
  sum(a, n);
  const size_t blocks = n / (THREAD_BLOCK_SIZE * 2) + 1;
  cudaMemcpy(h, a, blocks * sizeof(double), cudaMemcpyDeviceToHost);
  expect("sum partials", h, blocks, 4.0 * THREAD_BLOCK_SIZE);

  free(h);
  if (failures) {
    printf("%d check(s) failed\n", failures);
    return 1;
  }
  printf("all kernel checks passed\n");
  return 0;
}
