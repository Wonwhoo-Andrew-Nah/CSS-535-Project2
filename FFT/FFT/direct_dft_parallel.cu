#include <stdio.h>
#include <math.h>
#include <cuComplex.h>

#define PI 3.14159265358979323846

__global__ void direct_dft(cuDoubleComplex *in, cuDoubleComplex *out, int n)
{
        int threadID = blockIdx.x * blockDim.x + threadIdx.x;

        if (threadID < n)
        {
                cuDoubleComplex sum = make_cuDoubleComplex(0, 0);

                for (int i = 0; i < n; i++)
                {
                        double angle = -2.0 * PI * threadID * i / n;

                        cuDoubleComplex tSub1 
                                = make_cuDoubleComplex(cos(angle), 0);

                        cuDoubleComplex tSub2 
                                = cuCmul(make_cuDoubleComplex(0, 1), 
                                         make_cuDoubleComplex(sin(angle), 0));

                        cuDoubleComplex t1 = cuCmul(in[i], tSub1);
                        cuDoubleComplex t2 = cuCmul(in[i], tSub2);

                        cuDoubleComplex t = cuCadd(t1, t2);

                        sum = cuCadd(sum, t);
                }

                out[threadID] = sum;
        }
}

int main()
{
        int n = 16;

        cuDoubleComplex *h_in = (cuDoubleComplex *)malloc(n * sizeof(cuDoubleComplex));
        cuDoubleComplex *h_out = (cuDoubleComplex *)malloc(n * sizeof(cuDoubleComplex));

        cuDoubleComplex *d_in;
        cudaMalloc(&d_in, n * sizeof(cuDoubleComplex));

        cuDoubleComplex *d_out;
        cudaMalloc(&d_out, n * sizeof(cuDoubleComplex));


        printf("in:\n");
        h_in[0] = make_cuDoubleComplex((double) 0, 0);
        h_in[1] = make_cuDoubleComplex((double) 1, 0);
        h_in[2] = make_cuDoubleComplex((double) 2, 0);
        h_in[3] = make_cuDoubleComplex((double) 3, 0);
        h_in[4] = make_cuDoubleComplex((double) 4, 0);
        h_in[5] = make_cuDoubleComplex((double) 5, 0);
        h_in[6] = make_cuDoubleComplex((double) 6, 0);
        h_in[7] = make_cuDoubleComplex((double) 7, 0);
        h_in[8] = make_cuDoubleComplex((double) 8, 0);
        h_in[9] = make_cuDoubleComplex((double) 9, 0);
        h_in[10] = make_cuDoubleComplex((double) 10, 0);
        h_in[11] = make_cuDoubleComplex((double) 11, 0);
        h_in[12] = make_cuDoubleComplex((double) 12, 0);
        h_in[13] = make_cuDoubleComplex((double) 13, 0);
        h_in[14] = make_cuDoubleComplex((double) 14, 0);
        h_in[15] = make_cuDoubleComplex((double) 15, 0);

        for (int i = 0; i < n; i++)
        {
                printf("%f, %f\n", cuCreal(h_in[i]), cuCimag(h_in[i]));
        }

        cudaMemcpy(d_in, h_in, n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

        direct_dft<<<4, 4>>>(d_in, d_out, n);

        cudaMemcpy(h_out, d_out, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

        printf("\nout:\n");
        for (int i = 0; i < n; i++)
        {
                printf("%f, %f\n", cuCreal(h_out[i]), cuCimag(h_out[i]));
        }

        free(h_in);
        free(h_out);
        cudaFree(d_in);
        cudaFree(d_out);

        return 0;
}
