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

#ifndef _AQUARIUS_PGSYMMETRIC_TENSOR_HPP_
#define _AQUARIUS_PGSYMMETRIC_TENSOR_HPP_

#include "util/stl_ext.hpp"

#include "tensor.hpp"

namespace aquarius
{
namespace tensor
{

template <> class TensorInitializer<PGSYMMETRIC_>
{
    public:
        symmetry::Representation rep;
        std::vector<std::vector<int> > len;

        TensorInitializer(const symmetry::Representation& rep,
                          const std::vector<std::vector<int> >& len)
        : rep(rep), len(len) {}
}

template <> class TensorInitializer<PGSYMMETRIC> : TensorInitializerList<PGSYMMETRIC>
{
    protected:
        static std::vector<int> totalLengths(const std::vector<std::vector<int> >& len)
        {
            std::vector<int> tot_len(len.size());

            for (int i = 0;i < len.size();i++)
            {
                for (int j = 0;j < len[i].size;j++)
                {
                    tot_len[i] += len[i][j];
                }
            }
        }

    public:
        TensorInitializer(const symmetry::Representation& rep,
                          const std::vector<std::vector<int> >& len)
        : TensorInitializerList<PGSYMMETRIC>(
                TensorInitializer<BOUNDED>(totalLengths(len)) <<
                TensorInitializer<PGSYMMETRIC_>(rep, len))
        {
            for (int i = 0;i < len.size();i++)
            {
                assert(len[i].size() == rep.getPointGroup().getNumIrreps());
            }
        }
}

template <> class TensorImpl<PGSYMMETRIC_>
{
    public:
        virtual ~TensorImpl() {}

        virtual const std::vector<std::vector<int> >& getIrrepLengths() const = 0;

        virtual const symmetry::Representation& getRepresentation() const = 0;
};

template <> class TensorWrapper<PGSYMMETRIC_>
{
    protected:
        Tensor<>& base;

    public:
        TensorWrapper(Tensor<>& base) : base(base) {}

        const std::vector<std::vector<int> >& getIrrepLengths() const
        {
            return base.impl<PGSYMMETRIC_>().getIrrepLengths();
        }

        const symmetry::PointGroup& getPointGroup() const
        {
            return base.impl<PGSYMMETRIC_>().getRepresentation().getPointGroup();
        }

        const symmetry::Representation& getRepresentation() const
        {
            return base.impl<PGSYMMETRIC_>().getRepresentation();
        }
};

}
}

#endif
