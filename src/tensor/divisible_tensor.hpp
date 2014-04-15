/* Copyright (c) 2013, Devin Matthews
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following
 * conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice,This list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice,This list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL DEVIN MATTHEWS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE. */

#ifndef _AQUARIUS_DIVISIBLE_TENSOR_HPP_
#define _AQUARIUS_DIVISIBLE_TENSOR_HPP_

#include "tensor.hpp"

namespace aquarius
{
namespace tensor
{

class InvertedTensor;
class TensorDiv;

template <> class TensorImpl<DIVISIBLE>
{
    public:
        virtual ~TensorImpl() {}

        /*
         * this = beta*this + alpha*A/B
         */
        virtual void div(const Scalar& alpha, bool conja, const TensorImpl<DIVISIBLE>& A,
                                              bool conjb, const TensorImpl<DIVISIBLE>& B, const Scalar& beta) = 0;

        /*
         * this = beta*this + alpha/A
         */
        virtual void invert(const Scalar& alpha, bool conja, const TensorImpl<DIVISIBLE>& A, const Scalar& beta) = 0;
};

template <> class TensorWrapper<DIVISIBLE>
{
    friend class InvertedTensor;
    friend class TensorDiv;

    protected:
        Tensor<>& base;

    public:
        TensorWrapper(Tensor<>& base) : base(base) {}

        /**********************************************************************
         *
         * Binary operations (multiplication and division)
         *
         *********************************************************************/

        Tensor<>& operator=(const TensorDiv& other);

        Tensor<>& operator+=(const TensorDiv& other);

        Tensor<>& operator-=(const TensorDiv& other);

        /**********************************************************************
         *
         * Unary operations (assignment, summation, scaling, and inversion)
         *
         *********************************************************************/

        Tensor<>& operator/=(const TensorWrapper<DIVISIBLE>& other);

        Tensor<>& operator=(const InvertedTensor& other);

        Tensor<>& operator+=(const InvertedTensor& other);

        Tensor<>& operator-=(const InvertedTensor& other);

        Tensor<>& operator*=(const InvertedTensor& other);

        Tensor<>& operator/=(const InvertedTensor& other);

        /**********************************************************************
         *
         * Intermediate operations
         *
         *********************************************************************/

        friend InvertedTensor operator/(const Scalar& factor, const TensorWrapper<DIVISIBLE>& other);

        TensorDiv operator/(const TensorWrapper<DIVISIBLE>& other) const;
};

class InvertedTensor
{
    private:
        const InvertedTensor& operator=(const InvertedTensor& other);

    public:
        const TensorImplementation<>& tensor;
        Scalar factor;
        bool conj;

        InvertedTensor(const TensorImplementation<>& tensor,
                       const Scalar& factor, bool conj=false)
        : tensor(tensor), factor(factor), conj(conj) {}

        /**********************************************************************
         *
         * Unary negation, conjugation
         *
         *********************************************************************/

        InvertedTensor operator-() const;

        friend InvertedTensor conj(const InvertedTensor& tm);

        /**********************************************************************
         *
         * Operations with scalars
         *
         *********************************************************************/

        InvertedTensor operator*(const Scalar& factor) const;

        InvertedTensor operator/(const Scalar& factor) const;

        friend InvertedTensor operator*(const Scalar& factor, const InvertedTensor& other);
};

class TensorDiv
{
    private:
        const TensorDiv& operator=(const TensorDiv& other);

    public:
        const TensorImplementation<>& A;
        const TensorImplementation<>& B;
        bool conja;
        bool conjb;
        Scalar factor;

        TensorDiv(const ConstScaledTensor& A, const ConstScaledTensor& B)
        : A(A.tensor), conja(A.conj),
          B(B.tensor), conjb(B.conj),
          factor(A.factor/B.factor) {}

        /**********************************************************************
         *
         * Unary negation, conjugation
         *
         *********************************************************************/

        TensorDiv operator-() const;

        friend TensorDiv conj(const TensorDiv& tm);

        /**********************************************************************
         *
         * Operations with scalars
         *
         *********************************************************************/

        TensorDiv operator*(const Scalar& factor) const;

        TensorDiv operator/(const Scalar& factor) const;

        friend TensorDiv operator*(const Scalar& factor, const TensorDiv& other);
};

}
}

#endif
