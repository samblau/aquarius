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

#include "indexable_tensor.hpp"

IndexedTensor TensorWrapper<INDEXABLE>::operator[](const std::string& idx)
{
    return IndexedTensor(*this, idx);
}

ConstIndexedTensor TensorWrapper<INDEXABLE>::operator[](const std::string& idx) const
{
    return ConstIndexedTensor(*this, idx);
}

Tensor<>& TensorWrapper<INDEXABLE>::operator=(const IndexedTensorMult& other)
{
    (*this)[implicit(getDimension())] = other;
    return base;
}

Tensor<>& TensorWrapper<INDEXABLE>::operator+=(const IndexedTensorMult& other)
{
    (*this)[implicit(getDimension())] += other;
    return base;
}

Tensor<>& TensorWrapper<INDEXABLE>::operator-=(const IndexedTensorMult& other)
{
    (*this)[implicit(getDimension())] -= other;
    return base;
}

Tensor<>& TensorWrapper<INDEXABLE>::operator=(const ConstIndexedTensor& other)
{
    (*this)[implicit(getDimension())] = other;
    return base;
}

Tensor<>& TensorWrapper<INDEXABLE>::operator+=(const ConstIndexedTensor& other)
{
    (*this)[implicit(getDimension())] += other;
    return base;
}

Tensor<>& TensorWrapper<INDEXABLE>::operator-=(const ConstIndexedTensor& other)
{
    (*this)[implicit(getDimension())] -= other;
    return base;
}

IndexedTensor ConstIndexedTensor::operator-() const
{
    IndexedTensor ret(*this);
    ret.factor = -ret.factor;
    return ret;
}

IndexedTensor conj(const IndexedTensor& other)
{
    IndexedTensor ret(other);
    ret.conj = !ret.conj;
    return ret;
}

IndexedTensorMult ConstIndexedTensor::operator*(const ConstIndexedTensor& other) const
{
    return IndexedTensorMult(*this, other);
}

IndexedTensorMult ConstIndexedTensor::operator*(const ConstScaledTensor& other) const
{
    return IndexedTensorMult(*this,
        ConstIndexedTensor(other.tensor,
                           implicit(other.tensor.as<INDEXABLE>().getDimension()),
                           other.factor,
                           other.conj));
}

IndexedTensorMult ConstIndexedTensor::operator*(const TensorWrapper<INDEXABLE>& other) const
{
    return IndexedTensorMult(*this, other[implicit(other.getDimension())]);
}

IndexedTensorMult operator*(const TensorWrapper<INDEXABLE>& t1,
                            const ConstIndexedTensor& t2)
{
    return IndexedTensorMult(t1[implicit(t1.getDimension())], t2);
}

IndexedTensorMult operator*(const ConstScaledTensor& t1,
                            const ConstIndexedTensor& t2)
{
    return IndexedTensorMult(
        ConstIndexedTensor(t1.tensor,
                           implicit(t1.base.as<INDEXABLE>().getDimension()),
                           t1.factor,
                           t1.conj), t2);
}

IndexedTensor ConstIndexedTensor::operator*(const Scalar& factor) const
{
    IndexedTensor it(*this);
    it.factor *= factor;
    return it;
}

IndexedTensor operator*(const Scalar& factor, const IndexedTensor& other)
{
    return other*factor;
}

IndexedTensor& IndexedTensor::operator=(const IndexedTensor& other)
{
    tensor.sum(other.factor, other.conj, other.tensor, other.idx, 0, idx);
    return *this;
}

IndexedTensor& IndexedTensor::operator=(const IndexedTensor& other)
{
    tensor.sum(other.factor, other.conj, other.tensor, other.idx, 0, idx);
    return base;
}

IndexedTensor& IndexedTensor::operator+=(const IndexedTensor& other)
{
    tensor.sum(other.factor, other.conj, other.tensor, other.idx, factor, idx);
    return *this;
}

IndexedTensor& IndexedTensor::operator-=(const IndexedTensor& other)
{
    tensor.sum(-other.factor, other.conj, other.tensor, other.idx, factor, idx);
    return *this;
}

IndexedTensor& IndexedTensor::operator=(const IndexedTensorMult& other)
{
    tensor.mult(other.factor, other.A.conj, other.A.tensor.as<INDEXABLE>(), other.A.idx,
                              other.B.conj, other.B.tensor.as<INDEXABLE>(), other.B.idx,
                           0,                                                       idx);
    return *this;
}

IndexedTensor& IndexedTensor::operator+=(const IndexedTensorMult& other)
{
    tensor.mult(other.factor, other.A.conj, other.A.tensor.as<INDEXABLE>(), other.A.idx,
                              other.B.conj, other.B.tensor.as<INDEXABLE>(), other.B.idx,
                      factor,                                                       idx);
    return *this;
}

IndexedTensor& IndexedTensor::operator-=(const IndexedTensorMult& other)
{
    tensor.mult(-other.factor, other.A.conj, other.A.tensor.as<INDEXABLE>(), other.A.idx,
                               other.B.conj, other.B.tensor.as<INDEXABLE>(), other.B.idx,
                       factor,                                                       idx);
    return *this;
}

IndexedTensor& IndexedTensor::operator*=(const Scalar& factor)
{
    tensor.as<INDEXABLE>().scale(factor, idx);
    return *this;
}

IndexedTensor& IndexedTensor::operator/=(const Scalar& factor)
{
    tensor.as<INDEXABLE>().scale(1/factor, idx);
    return *this;
}

IndexedTensorMult IndexedTensorMult::operator-() const
{
    IndexedTensorMult ret(*this);
    ret.factor = -ret.factor;
    return ret;
}

IndexedTensorMult conj(const IndexedTensorMult& other)
{
    IndexedTensorMult ret(other);
    ret.A.conj = !ret.A.conj;
    ret.B.conj = !ret.B.conj;
    return ret;
}

IndexedTensorMult IndexedTensorMult::operator*(const Scalar& factor) const
{
    IndexedTensorMult ret(*this);
    ret.factor *= factor;
    return ret;
}

IndexedTensorMult IndexedTensorMult::operator/(const Scalar& factor) const
{
    IndexedTensorMult ret(*this);
    ret.factor /= factor;
    return ret;
}

IndexedTensorMult operator*(const Scalar& factor, const IndexedTensorMult& other)
{
    return other*factor;
}

Scalar scalar(const tensor::IndexedTensorMult& itm)
{
    return itm.factor*itm.B.as<INDEXABLE>().dot(itm.conja, itm.A.as<INDEXABLE>(), itm.idxa,
                                                itm.conjb,                        itm.idxb);
}
