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

#include "tensor.hpp"

using namespace std;
using namespace aquarius;
using namespace aquarius::tensor;

Tensor<>& Tensor<>::operator*=(const Scalar& val)
{
    impl().scale(val);
    return *this;
}

Tensor<>& Tensor<>::operator/=(const Scalar& val)
{
    impl().scale(1/val);
    return *this;
}

/**********************************************************************
 *
 * Binary operations (multiplication and division)
 *
 *********************************************************************/

Tensor<>& Tensor<>::operator=(const TensorMult& other)
{
    impl().mult(other.factor_, other.conja_, other.A_, other.conjb_, other.B_, 0);
    return *this;
}

Tensor<>& Tensor<>::operator+=(const TensorMult& other)
{
    impl().mult(other.factor_, other.conja_, other.A_, other.conjb_, other.B_, 1);
    return *this;
}

Tensor<>& Tensor<>::operator-=(const TensorMult& other)
{
    impl().mult(-other.factor_, other.conja_, other.A_, other.conjb_, other.B_, 1);
    return *this;
}

/**********************************************************************
 *
 * Unary operations (assignment, summation, scaling, and inversion)
 *
 *********************************************************************/

Tensor<>& Tensor<>::operator=(const Tensor<>& other)
{
    impl().sum(1, false, other.impl(), 0);
    return *this;
}

Tensor<>& Tensor<>::operator+=(const Tensor<>& other)
{
    impl().sum(1, false, other.impl(), 1);
    return *this;
}

Tensor<>& Tensor<>::operator-=(const Tensor<>& other)
{
    impl().sum((-1), false, other.impl(), 1);
    return *this;
}

Tensor<>& Tensor<>::operator*=(const Tensor<>& other)
{
    impl().mult(1, false, impl(), false, other.impl(), 0);
    return *this;
}

Tensor<>& Tensor<>::operator=(const ConstScaledTensor& other)
{
    impl().sum(other.factor_, other.conj_, other.tensor_, 0);
    return *this;
}

Tensor<>& Tensor<>::operator+=(const ConstScaledTensor& other)
{
    impl().sum(other.factor_, other.conj_, other.tensor_, 1);
    return *this;
}

Tensor<>& Tensor<>::operator-=(const ConstScaledTensor& other)
{
    impl().sum(-other.factor_, other.conj_, other.tensor_, 1);
    return *this;
}

Tensor<>& Tensor<>::operator*=(const ConstScaledTensor& other)
{
    impl().mult(other.factor_, false, impl(), other.conj_, other.tensor_, 0);
    return *this;
}

/**********************************************************************
 *
 * Intermediate operations
 *
 *********************************************************************/

ScaledTensor operator*(const Scalar& factor, Tensor<>& other)
{
    return ScaledTensor(other.impl(), factor);
}

ConstScaledTensor operator*(const Scalar& factor, const Tensor<>& other)
{
    return ConstScaledTensor(other.impl(), factor);
}

ScaledTensor Tensor<>::operator*(const Scalar& factor)
{
    return ScaledTensor(impl(), factor);
}

ConstScaledTensor Tensor<>::operator*(const Scalar& factor) const
{
    return ConstScaledTensor(impl(), factor);
}

ScaledTensor Tensor<>::operator/(const Scalar& factor)
{
    return ScaledTensor(impl(), 1/factor);
}

ConstScaledTensor Tensor<>::operator/(const Scalar& factor) const
{
    return ConstScaledTensor(impl(), 1/factor);
}

ScaledTensor Tensor<>::operator-()
{
    return ScaledTensor(impl(), -1);
}

ConstScaledTensor Tensor<>::operator-() const
{
    return ConstScaledTensor(impl(), -1);
}

ConstScaledTensor conj(const Tensor<>& t)
{
    return ConstScaledTensor(t.impl(), 1, true);
}

TensorMult Tensor<>::operator*(const Tensor<>& other) const
{
    return TensorMult(ConstScaledTensor(      impl(), 1),
                      ConstScaledTensor(other.impl(), 1));
}

/**********************************************************************
 *
 * Unary negation, conjugation
 *
 *********************************************************************/

ConstScaledTensor ConstScaledTensor::operator-() const
{
    ConstScaledTensor ret(*this);
    ret.factor_ = -ret.factor_;
    return ret;
}

ConstScaledTensor conj(const ConstScaledTensor& st)
{
    ConstScaledTensor ret(st);
    ret.conj_ = !ret.conj_;
    return ret;
}

/**********************************************************************
 *
 * Binary tensor operations
 *
 *********************************************************************/

TensorMult ConstScaledTensor::operator*(const ConstScaledTensor& other) const
{
    return TensorMult(*this, other);
}

TensorMult ConstScaledTensor::operator*(const Tensor<>& other) const
{
    return TensorMult(*this, ConstScaledTensor(other, 1));
}

TensorMult operator*(const Tensor<>& t, const ConstScaledTensor& other)
{
    return TensorMult(ConstScaledTensor(t, 1), other);
}

/**********************************************************************
 *
 * Operations with scalars
 *
 *********************************************************************/

ConstScaledTensor ConstScaledTensor::operator*(const Scalar& factor) const
{
    ConstScaledTensor it(*this);
    it.factor_ *= factor;
    return it;
}

ConstScaledTensor operator*(const Scalar& factor, const ConstScaledTensor& other)
{
    return other*factor;
}

ConstScaledTensor ConstScaledTensor::operator/(const Scalar& factor) const
{
    ConstScaledTensor it(*this);
    it.factor_ /= factor;
    return it;
}

/**********************************************************************
 *
 * Unary negation, conjugation
 *
 *********************************************************************/

ScaledTensor ScaledTensor::operator-() const
{
    ScaledTensor ret(*this);
    ret.factor_ = -ret.factor_;
    return ret;
}

/**********************************************************************
 *
 * Unary tensor operations
 *
 *********************************************************************/

ScaledTensor& ScaledTensor::operator+=(const Tensor<>& other)
{
    tensor_.sum(1, false, other.impl(), factor_);
    return *this;
}

ScaledTensor& ScaledTensor::operator-=(const Tensor<>& other)
{
    tensor_.sum(-1, false, other.impl(), factor_);
    return *this;
}

ScaledTensor& ScaledTensor::operator*=(const Tensor<>& other)
{
    tensor_.mult(factor_, false, tensor_, false, other.impl(), 0);
    return *this;
}

ScaledTensor& ScaledTensor::operator=(const ScaledTensor& other)
{
    tensor_.sum(other.factor_, other.conj_, other.tensor_, 0);
    return *this;
}

ScaledTensor& ScaledTensor::operator=(const ConstScaledTensor& other)
{
    tensor_.sum(other.factor_, other.conj_, other.tensor_, 0);
    return *this;
}

ScaledTensor& ScaledTensor::operator+=(const ConstScaledTensor& other)
{
    tensor_.sum(other.factor_, other.conj_, other.tensor_, factor_);
    return *this;
}

ScaledTensor& ScaledTensor::operator-=(const ConstScaledTensor& other)
{
    tensor_.sum(-other.factor_, other.conj_, other.tensor_, factor_);
    return *this;
}

ScaledTensor& ScaledTensor::operator*=(const ConstScaledTensor& other)
{
    tensor_.mult(factor_*other.factor_, false, tensor_, other.conj_, other.tensor_, 0);
    return *this;
}

/**********************************************************************
 *
 * Binary tensor operations
 *
 *********************************************************************/

ScaledTensor& ScaledTensor::operator=(const TensorMult& other)
{
    tensor_.mult(other.factor_, other.conja_, other.A_, other.conjb_, other.B_, 0);
    return *this;
}

ScaledTensor& ScaledTensor::operator+=(const TensorMult& other)
{
    tensor_.mult(other.factor_, other.conja_, other.A_, other.conjb_, other.B_, factor_);
    return *this;
}

ScaledTensor& ScaledTensor::operator-=(const TensorMult& other)
{
    tensor_.mult(-other.factor_, other.conja_, other.A_, other.conjb_, other.B_, factor_);
    return *this;
}

/**********************************************************************
 *
 * Operations with scalars
 *
 *********************************************************************/

ScaledTensor ScaledTensor::operator*(const Scalar& factor) const
{
    ScaledTensor it(*this);
    it.factor_ *= factor;
    return it;
}

ScaledTensor operator*(const Scalar& factor, const ScaledTensor& other)
{
    return other*factor;
}

ScaledTensor ScaledTensor::operator/(const Scalar& factor) const
{
    ScaledTensor it(*this);
    it.factor_ /= factor;
    return it;
}

ScaledTensor& ScaledTensor::operator*=(const Scalar& val)
{
    tensor_.scale(val*factor_);
    return *this;
}

ScaledTensor& ScaledTensor::operator/=(const Scalar& val)
{
    tensor_.scale(factor_/val);
    return *this;
}

/**********************************************************************
 *
 * Unary negation, conjugation
 *
 *********************************************************************/

TensorMult TensorMult::operator-() const
{
    TensorMult ret(*this);
    ret.factor_ = -ret.factor_;
    return ret;
}

TensorMult conj(const TensorMult& tm)
{
    TensorMult ret(tm);
    ret.conja_ = !ret.conja_;
    ret.conjb_ = !ret.conjb_;
    return ret;
}

/**********************************************************************
 *
 * Operations with scalars
 *
 *********************************************************************/

TensorMult TensorMult::operator*(const Scalar& factor) const
{
    TensorMult ret(*this);
    ret.factor_ *= factor;
    return ret;
}

TensorMult TensorMult::operator/(const Scalar& factor) const
{
    TensorMult ret(*this);
    ret.factor_ /= factor;
    return ret;
}

TensorMult operator*(const Scalar& factor, const TensorMult& other)
{
    return other*factor;
}
