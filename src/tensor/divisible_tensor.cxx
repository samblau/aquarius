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

#include "divisible_tensor.hpp"

using namespace std;
using namespace aquarius;
using namespace aquarius::tensor;

/**********************************************************************
 *
 * Binary operations (multiplication and division)
 *
 *********************************************************************/

Tensor<>& TensorWrapper<DIVISIBLE>::operator=(const TensorDiv& other)
{
    assert(base.C == other.A.base.C);
    assert(base.C == other.B.base.C);
    base.impl<DIVISIBLE>().div(other.factor, other.conja, other.A.as<DIVISIBLE>(),
                                             other.conjb, other.B.as<DIVISIBLE>(), 0);
    return base;
}

Tensor<>& TensorWrapper<DIVISIBLE>::operator+=(const TensorDiv& other)
{
    base.impl<DIVISIBLE>().div(other.factor, other.conja, other.A.as<DIVISIBLE>(),
                                             other.conjb, other.B.as<DIVISIBLE>(), 1);
    return base;
}

Tensor<>& TensorWrapper<DIVISIBLE>::operator-=(const TensorDiv& other)
{
    base.impl<DIVISIBLE>().div(-other.factor, other.conja, other.A.as<DIVISIBLE>(),
                                              other.conjb, other.B.as<DIVISIBLE>(), 1);
    return base;
}

/**********************************************************************
 *
 * Unary operations (assignment, summation, scaling, and inversion)
 *
 *********************************************************************/

Tensor<>& TensorWrapper<DIVISIBLE>::operator/=(const TensorWrapper<DIVISIBLE>& other)
{
    base.impl<DIVISIBLE>().div(1, false,       base.impl<DIVISIBLE>(),
                                  false, other.base.impl<DIVISIBLE>(), 0);
    return base;
}

Tensor<>& TensorWrapper<DIVISIBLE>::operator=(const InvertedTensor& other)
{
    base.impl<DIVISIBLE>().invert(other.factor, other.conj, other.tensor.as<DIVISIBLE>(), 0);
    return base;
}

Tensor<>& TensorWrapper<DIVISIBLE>::operator+=(const InvertedTensor& other)
{
    base.impl<DIVISIBLE>().invert(other.factor, other.conj, other.tensor.as<DIVISIBLE>(), 1);
    return base;
}

Tensor<>& TensorWrapper<DIVISIBLE>::operator-=(const InvertedTensor& other)
{
    base.impl<DIVISIBLE>().invert(-other.factor, other.conj, other.tensor.as<DIVISIBLE>(), 0);
    return base;
}

Tensor<>& TensorWrapper<DIVISIBLE>::operator*=(const InvertedTensor& other)
{
    base.impl<DIVISIBLE>().div(other.factor,      false,       base.impl<DIVISIBLE>(),
                                             other.conj, other.tensor.as<DIVISIBLE>(), 0);
    return base;
}

Tensor<>& TensorWrapper<DIVISIBLE>::operator/=(const InvertedTensor& other)
{
    base.impl().mult(1/other.factor,      false,  base.impl(),
                                     other.conj, other.tensor, 0);
    return base;
}

/**********************************************************************
 *
 * Intermediate operations
 *
 *********************************************************************/

InvertedTensor operator/(const Scalar& factor, const TensorWrapper<DIVISIBLE>& other)
{
    return InvertedTensor(other, factor);
}

TensorDiv TensorWrapper<DIVISIBLE>::operator/(const TensorWrapper<DIVISIBLE>& other) const
{
    return TensorDiv(ConstScaledTensor(      base, 1),
                     ConstScaledTensor(other.base, 1));
}

/**********************************************************************
 *
 * Unary negation, conjugation
 *
 *********************************************************************/

InvertedTensor InvertedTensor::operator-() const
{
    InvertedTensor ret(*this);
    ret.factor = -ret.factor;
    return *this;
}

InvertedTensor conj(const InvertedTensor& tm)
{
    InvertedTensor ret(tm);
    ret.conj = !ret.conj;
    return ret;
}

/**********************************************************************
 *
 * Operations with scalars
 *
 *********************************************************************/

InvertedTensor InvertedTensor::operator*(const Scalar& factor) const
{
    InvertedTensor ret(*this);
    ret.factor *= factor;
    return ret;
}

InvertedTensor InvertedTensor::operator/(const Scalar& factor) const
{
    InvertedTensor ret(*this);
    ret.factor /= factor;
    return ret;
}

InvertedTensor operator*(const Scalar& factor, const InvertedTensor& other)
{
    return other*factor;
}

/**********************************************************************
 *
 * Unary negation, conjugation
 *
 *********************************************************************/

TensorDiv TensorDiv::operator-() const
{
    TensorDiv ret(*this);
    ret.factor = -ret.factor;
    return ret;
}

TensorDiv conj(const TensorDiv& tm)
{
    TensorDiv ret(tm);
    ret.conja = !ret.conja;
    ret.conjb = !ret.conjb;
    return ret;
}

/**********************************************************************
 *
 * Operations with scalars
 *
 *********************************************************************/

TensorDiv TensorDiv::operator*(const Scalar& factor) const
{
    TensorDiv ret(*this);
    ret.factor *= factor;
    return ret;
}

TensorDiv TensorDiv::operator/(const Scalar& factor) const
{
    TensorDiv ret(*this);
    ret.factor /= factor;
    return ret;
}

TensorDiv operator*(const Scalar& factor, const TensorDiv& other)
{
    return other*factor;
}
