/* Copyright (c) 2013, Devin Matthews
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following
 * conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL DEVIN MATTHEWS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE. */

#ifndef _AQUARIUS_UTIL_H_
#define _AQUARIUS_UTIL_H_

#define DIV_FLOPS 30
#define SQRT_FLOPS 30
#define EXP_FLOPS 30
#define LOG_FLOPS 30

#define IS_POWER_OF_TWO(x) (((x)>0)&&(((x)&((x)-1))==0))

/*
#define INSTANTIATE_SPECIALIZATIONS(name) \
template class name<double>; \
template class name<float>; \
template class name<std::complex<double> >; \
template class name<std::complex<float> >;
*/

#ifdef __cplusplus

#include <algorithm>
#include <cstdio>

inline void printmatrix(int nc, int nr, const double *m,
                        int width, int prec, int maxwidth)
{
    if (nr == 0 || nc == 0)
    {
        printf("{empty}\n");
        return;
    }

    width = std::max(width,prec+3);

    int maxcol = std::max(1,(maxwidth-3-2)/(width+1));

    for (int cb = 0;cb < nc;cb += maxcol)
    {
        printf("     ");
        for (int c = cb;c < std::min(nc,cb+maxcol);c++)
        {
            printf(" %*d", width, c+1);
        }
        printf("\n");

        printf("    +");
        for (int i = 0;i < std::min(nc-cb,maxcol)*(width+1);i++) printf("-");
        printf("\n");

        for (int r = 0;r < nr;r++)
        {
            printf("%3d |", r+1);
            for (int c = cb;c < std::min(nc,cb+maxcol);c++)
            {
                printf(" % *.*f", width, prec, m[c*nr+r]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

#endif

#define INSTANTIATE_SPECIALIZATIONS(name) \
template class name<double>;

#define INSTANTIATE_SPECIALIZATIONS_2(name,extra1) \
template class name<double,extra1>;

#define INSTANTIATE_SPECIALIZATIONS_3(name,extra1,extra2) \
template class name<double,extra1,extra2>;

#define CONCAT(...) __VA_ARGS__

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define ABS(a) ((a) < 0 ? -(a) : (a))

#ifdef DEBUG

#define DPRINTF(...) \
do \
{ \
    printf("%s(%d): ", __FILE__, __LINE__); \
    printf(__VA_ARGS__); \
} while (0)

#define DPRINTFC(...) \
do \
{ \
    printf(__VA_ARGS__); \
} while (0)

#else

#define DPRINTF(...)

#define DPRINTFC(...)

#endif

#endif
