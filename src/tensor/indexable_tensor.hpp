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

#ifndef _AQUARIUS_INDEXABLE_TENSOR_HPP_
#define _AQUARIUS_INDEXABLE_TENSOR_HPP_

#include <vector>
#include <string>
#include <algorithm>

#include "util/stl_ext.hpp"

#include "tensor.hpp"

namespace aquarius
{
namespace tensor
{

class ConstIndexedTensor;
class IndexedTensor;
class IndexedTensorMult;

std::string implicit(int ndim)
{
    std::string inds(ndim, ' ');
    for (int i = 0;i < ndim;i++) inds[i] = (char)('A'+i);
    return inds;
}

template <> class TensorInitializer<INDEXABLE>
{
    public:
        int ndim;

        TensorInitializer(int ndim) : ndim(ndim) {}
}

template <> class TensorImpl<INDEXABLE>
{
    public:
        virtual ~TensorImpl() {}

        virtual int getDimension() const = 0;

        /**********************************************************************
         *
         * Binary tensor operations (multiplication)
         *
         *********************************************************************/
        virtual void mult(const Scalar& alpha, bool conja, const TensorImpl<INDEXABLE>& A, const std::string& idxA,
                                               bool conjb, const TensorImpl<INDEXABLE>& B, const std::string& idxB,
                          const Scalar& beta,                                              const std::string& idxC) = 0;


        /**********************************************************************
         *
         * Unary tensor operations (summation)
         *
         *********************************************************************/
        virtual void sum(const Scalar& alpha, bool conja, const TensorImpl<INDEXABLE>& A, const std::string& idxA,
                         const Scalar& beta,                                              const std::string& idxB) = 0;


        /**********************************************************************
         *
         * Scalar operations
         *
         *********************************************************************/
        virtual void scale(const Scalar& alpha, const std::string& idxA) = 0;

        virtual Scalar dot(bool conja, const TensorImpl<INDEXABLE>& A, const std::string& idxA,
                           bool conjb,                                 const std::string& idxB) const = 0;
};

template <> class TensorWrapper<INDEXABLE>
{
    protected:
        Tensor<>& base;

    public:
        TensorWrapper(Tensor<>& base) : base(base) {}

        int getDimension() const
        {
            return base.impl<INDEXABLE>().getDimension();
        }

        /**********************************************************************
         *
         * Explicit indexing operations
         *
         *********************************************************************/

        IndexedTensor operator[](const std::string& idx);

        ConstIndexedTensor operator[](const std::string& idx) const;

        /**********************************************************************
         *
         * Implicitly indexed binary operations (inner product, trace, and weighting)
         *
         *********************************************************************/

        Tensor<>& operator=(const IndexedTensorMult& other);

        Tensor<>& operator+=(const IndexedTensorMult& other);

        Tensor<>& operator-=(const IndexedTensorMult& other);

        /**********************************************************************
         *
         * Implicitly indexed unary operations (assignment and summation)
         *
         *********************************************************************/

        Tensor<>& operator=(const ConstIndexedTensor& other);

        Tensor<>& operator+=(const ConstIndexedTensor& other);

        Tensor<>& operator-=(const ConstIndexedTensor& other);
};

class ConstIndexedTensor
{
    private:
        const ConstIndexedTensor& operator=(const ConstIndexedTensor& other);

    public:
        TensorImplementation<>& tensor;
        std::string idx;
        Scalar factor;
        bool conj;

        ConstIndexedTensor(const TensorImplementation<>& tensor,
                      const std::string& idx,
                      const Scalar& factor,
                      bool conj=false)
        : tensor(const_cast<TensorImplementation<>&>(tensor)),
          idx(idx), factor(factor), conj(conj)
        {
            if (idx.size() != tensor.as<INDEXABLE>().getDimension()) throw InvalidNdimError();
        }

        /**********************************************************************
         *
         * Unary negation, conjugation
         *
         *********************************************************************/
        IndexedTensor operator-() const;

        friend IndexedTensor conj(const IndexedTensor& other);

        /**********************************************************************
         *
         * Binary tensor operations (multiplication)
         *
         *********************************************************************/

        IndexedTensorMult operator*(const ConstIndexedTensor& other) const;

        IndexedTensorMult operator*(const ConstScaledTensor& other) const;

        IndexedTensorMult operator*(const TensorWrapper<INDEXABLE>& other) const;

        friend IndexedTensorMult operator*(const TensorWrapper<INDEXABLE>& t1,
                                           const ConstIndexedTensor& t2);

        friend IndexedTensorMult operator*(const ConstScaledTensor& t1,
                                           const ConstIndexedTensor& t2);

        /**********************************************************************
         *
         * Operations with scalars
         *
         *********************************************************************/
        IndexedTensor operator*(const Scalar& factor) const;

        friend IndexedTensor operator*(const Scalar& factor, const IndexedTensor& other);
};

class IndexedTensor : public ConstIndexedTensor
{
    public:
        IndexedTensor(TensorImplementation<>& tensor,
                      const std::string& idx,
                      const Scalar& factor,
                      bool conj=false)
        : ConstIndexedTensor(tensor, idx, factor, conj) {}

        /**********************************************************************
         *
         * Unary tensor operations (summation)
         *
         *********************************************************************/
        IndexedTensor& operator=(const IndexedTensor& other);

        IndexedTensor& operator=(const IndexedTensor& other);

        IndexedTensor& operator+=(const IndexedTensor& other);

        IndexedTensor& operator-=(const IndexedTensor& other);

        /**********************************************************************
         *
         * Binary tensor operations (multiplication)
         *
         *********************************************************************/
        IndexedTensor& operator=(const IndexedTensorMult& other);

        IndexedTensor& operator+=(const IndexedTensorMult& other);

        IndexedTensor& operator-=(const IndexedTensorMult& other);

        /**********************************************************************
         *
         * Operations with scalars
         *
         *********************************************************************/

        IndexedTensor& operator*=(const Scalar& factor);

        IndexedTensor& operator/=(const Scalar& factor);

        /*
        IndexedTensor& operator=(const Scalar& val)
        {
            Derived tensor(tensor, val);
            *this = tensor[""];
            return *this;
        }

        IndexedTensor& operator+=(const T val)
        {
            Derived tensor(tensor, val);
            *this += tensor[""];
            return *this;
        }

        IndexedTensor& operator-=(const T val)
        {
            Derived tensor(tensor, val);
            *this -= tensor[""];
            return *this;
        }
        */
};

class IndexedTensorMult
{
    private:
        const IndexedTensorMult& operator=(const IndexedTensorMult& other);

    public:
        TensorImplementation<>& A;
        TensorImplementation<>& B;
        std::string idxa;
        std::string idxb;
        Scalar factor;
        bool conja;
        bool conjb;

        IndexedTensorMult(const ConstIndexedTensor& A,
                          const ConstIndexedTensor& B)
        : A(A.tensor), B(B.tensor), idxa(A.idx), idxb(B.idx),
          factor(A.factor*B.factor), conja(A.conj), conjb(B.conj) {}

        /**********************************************************************
         *
         * Unary negation, conjugation
         *
         *********************************************************************/
        IndexedTensorMult operator-() const;

        friend IndexedTensorMult conj(const IndexedTensorMult& other);

        /**********************************************************************
         *
         * Operations with scalars
         *
         *********************************************************************/
        IndexedTensorMult operator*(const Scalar& factor) const;

        IndexedTensorMult operator/(const Scalar& factor) const;

        friend IndexedTensorMult operator*(const Scalar& factor, const IndexedTensorMult& other);
};

}

/**************************************************************************
 *
 * Tensor to scalar operations
 *
 *************************************************************************/
Scalar scalar(const tensor::IndexedTensorMult& itm);

}

#endif
