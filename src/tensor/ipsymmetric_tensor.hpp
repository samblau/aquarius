/* Copyright (c) 2014, Devin Matthews
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

#ifndef _AQUARIUS_TENSOR_IPSYMMETRIC_TENSOR_HPP_
#define _AQUARIUS_TENSOR_IPSYMMETRIC_TENSOR_HPP_

#include "util/stl_ext.hpp"

#include "tensor.hpp"

namespace aquarius
{
namespace tensor
{

template <> class TensorInitializer<IPSYMMETRIC_>
{
    public:
        std::vector<int> sym;

        TensorInitializer(const std::vector<int>& sym) : sym(sym) {}
}

template <> class TensorImpl<IPSYMMETRIC_>
{
    public:
        virtual ~TensorImpl() {}

        virtual const std::vector<int>& getSymmetry() const = 0;
};

template <> class TensorWrapper<IPSYMMETRIC_>
{
    protected:
        Tensor<>& base;

    public:
        TensorWrapper(Tensor<>& base) : base(base) {}

        const std::vector<int>& getSymmetry() const
        {
            return base.impl<IPSYMMETRIC_>().getSymmetry();
        }
};

}
}

#endif
