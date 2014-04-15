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

#ifndef _AQUARIUS_TENSOR_HPP_
#define _AQUARIUS_TENSOR_HPP_

#include <stdexcept>
#include <iostream>
#include <string>
#include <string.h>
#include <map>

#include "util/stl_ext.hpp"
#include "util/util.h"
#include "util/blas.h"

#include "ring.hpp"

#define ARE_DISTINCT(a,b) (((a)&(b))==0)
#define IS_SUPERSET_OF(a,b) (((a)&(b))==(a))

namespace aquarius
{
namespace tensor
{

typedef int64_t capability_type;

enum Capability {DIVISIBLE      = 0x0000000000000001ll,
                 INDEXABLE      = 0x0000000000000002ll,
                 BOUNDED_       = 0x0000000000000004ll,
                 BOUNDED        = BOUNDED_|INDEXABLE,
                 DISTRIBUTED    = 0x0000000000000008ll,
                 IPSYMMETRIC_   = 0x0000000000000010ll,
                 IPSYMMETRIC    = IPSYMMETRIC_|INDEXABLE,
                 PGSYMMETRIC_   = 0x0000000000000020ll,
                 PGSYMMETRIC    = PGSYMMETRIC_|BOUNDED,
                 SPINORBITAL_   = 0x0000000000000040ll,
                 SPINORBITAL    = SPINORBITAL_|IPSYMMETRIC|BOUNDED
};

#define DONT_NEED_INITIALIZATION (DIVISIBLE)
#define INITIALIZER_TYPE(C) InitializerList<((C)&(~DONT_NEED_INITIALIZATION))>

class ConstScaledTensor;
class ScaledTensor;
class TensorMult;

class TensorError;
class OutOfBoundsError;
class LengthMismatchError;
class IndexMismatchError;
class InvalidNdimError;
class InvalidLengthError;
class InvalidLdError;
class LdTooSmallError;
class SymmetryMismatchError;
class InvalidSymmetryError;
class InvalidStartError;

/**************************************************************************************************
 *
 * Tensor Initialization
 *
 *************************************************************************************************/

/*
 * Data structures to store initialization data specific to each capability
 */
template <capability_type C=0> class TensorInitializer;

/*
 * Generic initializer type which stores any number of specialized initializers.
 * This type is constructed from TensorInitializer by concatenation with the << operator.
 */
template <capability_type C=0> class TensorInitializerList
        : public std::map<capability_type, std::shared_ptr<std::Destructible> >
{
    template <capability_type C_> friend class TensorInitializerList;

    protected:
        /*
         * Allow initialization of a new initializer list from another which has a
         * subset of capabilities.
         */
        template <capability_type C_>
        TensorInitializerList(const TensorInitializerList<C_>& other,
                              typename std::enable_if<IS_SUPERSET_OF(C,C_),void*>::type = 0)
        : std::map<capability_type, std::shared_ptr<std::Destructible> >(other) {}

        TensorInitializerList() {}

    public:
        /*
         * Additional initializers may be added to the list by the << operator,
         * although the new initializer may not already be present.
         *
         * Checking for IS_POWER_OF_TWO(C_) ensures that the new initializar represents
         * only one capability, and that this specialization is not selected for something
         * like class TensorInitializer<A|B> : TensorInitializerList<A|B>.
         */
        template <capability_type C_>
        typename std::enable_if<ARE_DISTINCT(C,C_) && IS_POWER_OF_TWO(C_),
                                TensorInitializerList<C|C_> >::type
        operator<<(const TensorInitializer<C_>& init) const
        {
            TensorInitializerList<C|C_> ret(*this);
            /*
             * Can't use ret.swap(*this) since *this is const
             * (a non-member function taking rvalue-refs would be ideal).
             */
            ret[C_] = std::shared_ptr<std::Destructible>(new TensorInitializer<C_>(init));
            return ret;
        }

        /*
         * Initializer lists may also be concatenated providing they
         * provide distinct capabilities.
         */
        template <capability_type C_>
        typename std::enable_if<ARE_DISTINCT(C,C_),TensorInitializerList<C|C_> >::type
        operator<<(const TensorInitializerList<C_>& ilist) const
        {
            TensorInitializerList<C|C_> ret(*this);
            ret.insert(ilist.begin(), ilist.end());
            return ret;
        }
};

/*
 * Class to hold initialization data for the base Tensor class.
 */
class TensorInitializer_ : public std::Destructible
{
    public:
        const std::string name;
        const Field F;
        const Ring R;

        TensorInitializer_(const std::string& name, Field F, Ring R)
        : name(name), F(F), R(R) {}
};

/*
 * Thin wrapper which just creates a TensorInitializer_. Since this derives from
 * TensorInitializerList<> with a public constructor, this is the only way to
 * generator an initializer list, ensuring this information is present (alas, it
 * does not guarantee that it is not provided multiple times, in which case the behavior
 * is undefined).
 */
template <> class TensorInitializer<> : public TensorInitializerList<>
{
    public:
        TensorInitializer(const std::string& name, Field F) : TensorInitializerList<>()
        {
            (*this)[0] = std::shared_ptr<std::Destructible>(new TensorInitializer_(name, F, F));
        }

        TensorInitializer(const std::string& name, Field F, Ring R) : TensorInitializerList<>()
        {
            (*this)[0] = std::shared_ptr<std::Destructible>(new TensorInitializer_(name, F, R));
        }
};

/*
 * Convenience function since << should be commutative for this purpose.
 */
template <capability_type C1, capability_type C2>
typename std::enable_if<ARE_DISTINCT(C1,C1),TensorInitializerList<C1|C2> >::type
operator<<(const TensorInitializer<C1>& init, const TensorInitializerList<C2>& ilist)
{
    return ilist << init;
}

/**************************************************************************************************
 *
 * Tensor Capability Implementations
 *
 *************************************************************************************************/

/*
 * The real work-horse type which handles generic construction of base
 * capability implementations. TensorImplementation<> is a special case
 * which forms the base class (incl. type-erasure)
 * for the TensorImpl chain (although it is only a base once, not
 * for each wrapper derived from).
 */
template <capability_type C = 0, typename dummy=void> class TensorImplementation;

/*
 * Implementations of tensor capabilities, but only so far as providing basic
 * data structures and initialization.
 */
template <capability_type C, typename dummy=void> class TensorImpl;
template <capability_type C> class TensorImpl<C, typename std::enable_if<(C<0)>::type>
{
    public:
        TensorImpl(const std::Destructible& init) {}
};

/*
 * Automatically derives from TensorImpl for each defined capability.
 */
template <capability_type C,
          capability_type C_=(1ll<<(sizeof(capability_type)*8-2)),
          typename dummy=void>
class TensorImplBase;

/*
 * For capabilities C_ present in C, save a (appropriately typed and then
 * type-erased) pointer to the implementation for C_.
 */
template <capability_type C, capability_type C_>
class TensorImplBase<C, C_, typename std::enable_if<(C_!=0 && !ARE_DISTINCT(C,C_))>::type>
    : public TensorImplBase<C, (C_>>1)>,
      public TensorImpl<C_>
{
    public:
        TensorImplBase(const std::map<capability_type,
                       std::shared_ptr<std::Destructible> >& ilist)
        : TensorImplBase<C, (C_>>1)>(ilist),
          TensorImpl<Test>(
              static_cast<const TensorInitializer<C_>&>(*ilist.find(C_)->second))
        {
            this->ptr[C_] = static_cast<void*>(static_cast<TensorImpl<C_>*>(this));
        }
};

/*
 * For capabilities C_ not present in C, do nothing.
 */
template <capability_type C, int C_>
class TensorImplBase<C, C_, typename std::enable_if<(C_!=0 && ARE_DISTINCT(C,C_))>::type>
    : public TensorImplBase<C, (C_>>1)>
{
    public:
        TensorImplBase(const std::map<capability_type, std::shared_ptr<std::Destructible> >& ilist)
        : TensorImplBase<C, (C_>>1)>(ilist) {}
};

/*
 * Dummy specialization to terminate inheritance chain and derive
 * from the base TensorImplementation<> class.
 *
 * A note about TensorImplementation<(C&(~C))>: since the explicit
 * specialization of TensorImplementation<0> = TensorImplementation<>
 * hasn't been given yet, we need to trick the compiler into leaving the
 * template argument unevaluated but still ensure that it will be 0.
 */
template <capability_type C> class TensorImplBase<C,0> : public TensorImplementation<(C&(~C))>
{
    public:
        TensorImplBase(const std::map<capability_type, std::shared_ptr<std::Destructible> >& ilist)
        : TensorImplementation<>(C, static_cast<const TensorInitializer_&>(*ilist.find(0)->second)) {}
};

/*
template <capability_type C> typename std::enable_if<IS_POWER_OF_TWO(C),TensorImpl<C>&>::type
AsTensorImpl(TensorImplementation<>& t);

template <capability_type C> typename std::enable_if<IS_POWER_OF_TWO(C),const TensorImpl<C>&>::type
AsTensorImpl(const TensorImplementation<>& t);
*/

/**************************************************************************************************
 *
 * Tensor Capability Wrappers
 *
 *************************************************************************************************/

/*
 * The real work-horse type which handles initialization of wrappers
 * around an implementation and updating/downdating presented capabilities.
 * Tensor<> is a special case which forms the base class (incl. type-erasure)
 * for the TensorWrapper chain (although it is only a base once, not
 * for each wrapper derived from).
 */
template <capability_type C = 0, typename dummy=void> class Tensor;

/*
 * Implementations of AbstractTensor which simply delegate to
 * another implementation, but also provide more elaborate interface
 * options.
 */
template <capability_type C> class TensorWrapper;

/*
 * Automatically derives from TensorWrapper for each defined capability.
 */
template <capability_type C,
          capability_type C_=(1ll<<(sizeof(capability_type)*8-2)),
          typename dummy=void>
class TensorWrapperBase;

template <capability_type C, capability_type C_>
class TensorWrapperBase<C, C_, typename std::enable_if<(C_!=0 && !ARE_DISTINCT(C,C_))>::type>
    : public TensorWrapperBase<C, (C_>>1)>,
      public TensorWrapper<C_>
{
    public:
        TensorWrapperBase(TensorImplementation<>& t)
        : TensorWrapperBase<C, (C_>>1)>(t),
          TensorWrapper<C_>(static_cast<Tensor<>&>(*this)) {}
};

template <capability_type C, capability_type C_>
class TensorWrapperBase<C, C_, typename std::enable_if<(C_!=0 && ARE_DISTINCT(C,C_))>::type>
    : public TensorWrapperBase<C, (C_>>1)>
{
    public:
        TensorWrapperBase(TensorImplementation<>& t)
        : TensorWrapperBase<C, (C_>>1)>(t) {}
};

/*
 * Dummy specialization to terminate inheritance chain and derive
 * from the base Tensor<> class.
 *
 * A note about Tensor<(C&(~C))>: see TensorImplBase<C,0> above.
 */
template <capability_type C> class TensorWrapperBase<C,0> : public Tensor<(C&(~C))>
{
    public:
        TensorWrapperBase(TensorImplementation<>& t) : Tensor<>(C, t) {}
};

/**************************************************************************************************
 *
 * Tensor Implementation Base Specialization
 *
 *************************************************************************************************/

/*
 * No-capability specialization; this provides the operations universal to all
 * tensor types
 *
 * This also serves as a type-erased base class for return values and
 * capability-agnostic storage
 */
template <>
class TensorImplementation<> : public std::Destructible
{
    friend class Tensor<>;

    protected:
        std::map<capability_type,void*> ptr;

    public:
        const capability_type C;
        const Ring R;
        const Field F;
        const std::string name;

        TensorImplementation(capability_type C, const TensorInitializer_& ti)
        : C(C), R(ti.R), F(ti.F), name(ti.name) {}

        template <capability_type C>
        typename std::enable_if<IS_POWER_OF_TWO(C),TensorImpl<C>&>::type as()
        {
            std::map<capability_type,void*>::iterator i = ptr.find(C);
            assert(i != ptr.end());
            return *static_cast<TensorImpl<C>*>(i->second);
        }

        template <capability_type C>
        typename std::enable_if<IS_POWER_OF_TWO(C),const TensorImpl<C>&>::type as() const
        {
            std::map<capability_type,void*>::const_iterator i = ptr.find(C);
            assert(i != ptr.end());
            return *static_cast<const TensorImpl<C>*>(i->second);
        }

        /*
         * return a scalar unique to this process
         */
        virtual TensorImplementation<>& scalar() const = 0;

        /*
         * this = a
         */
        template <typename ring> TensorImplementation<>& operator=(const ring& a);

        /*
         * this = beta*this + alpha*A*B
         */
        virtual void mult(const Scalar& alpha, bool conja, const TensorImplementation<>& A,
                                               bool conjb, const TensorImplementation<>& B, const Scalar& beta) = 0;

        /*
         * this = beta*this + alpha*A
         */
        virtual void sum(const Scalar& alpha, bool conja, const TensorImplementation<>& A, const Scalar& beta) = 0;

        /*
         * this = alpha*this
         */
        virtual void scale(const Scalar& alpha) = 0;

        /*
         * scalar = A*this
         */
        virtual Scalar dot(bool conja, const TensorImplementation<>& A, bool conjb) const = 0;
};

/*
template <capability_type C>
typename std::enable_if<IS_POWER_OF_TWO(C),TensorImpl<C>&>::type
AsTensorImpl(TensorImplementation<>& t)
{
    return t.as<C>();
}

template <capability_type C>
typename std::enable_if<IS_POWER_OF_TWO(C),const TensorImpl<C>&>::type
AsTensorImpl(const TensorImplementation<>& t)
{
    return t.as<C>();
}
*/

/**************************************************************************************************
 *
 * Generic Tensor Implementation
 *
 *************************************************************************************************/

/*
 * Base class for all final implementations; provides automatic derivation from each of
 * the needed TensorImpl classes, safe destruction for type-erasure within Tensor,
 * and automatic initialization of TensorImpl base classes from an initialization list.
 */
template <capability_type C>
class TensorImplementation<C, typename std::enable_if<(C>0)>::type>
    : public TensorImplBase<C>
{
    public:
        /*
         * Constructor need only initiate automatic initialization chain
         */
        TensorImplementation(const INITIALIZER_TYPE(C)& ilist)
        : TensorImplBase<C>(ilist) {}
};

/**************************************************************************************************
 *
 * Tensor Wrapper Base Specialization
 *
 *************************************************************************************************/

/*
 * No-capability specialization; this provides the operations universal to all
 * tensor types
 *
 * This also serves as a type-erased base class for return values and
 * capability-agnostic storage
 */
template <>
class Tensor<>
{
    friend class ConstScaledTensor;
    friend class ScaledTensor;
    friend class TensorMult;
    template <capability_type C> friend class TensorWrapper;

    protected:
        /*
         * Use shared_ptr to ensure clean-up
         */
        std::shared_ptr<TensorImplementation<> > ptr;

    public:
        const capability_type C;

        Tensor(capability_type C, TensorImplementation<>& impl) : C(C), ptr(&impl) {}

        template <capability_type C>
        typename std::enable_if<IS_POWER_OF_TWO(C),TensorImpl<C>&>::type
        impl() { return ptr->as<C>(); }

        template <capability_type C>
        typename std::enable_if<IS_POWER_OF_TWO(C),const TensorImpl<C>&>::type
        impl() const { return ptr->as<C>(); }

        TensorImplementation<>& impl() { return *ptr; }

        const TensorImplementation<>& impl() const { return *ptr; }

        /**********************************************************************
         *
         * Operators with scalars
         *
         *********************************************************************/

        Tensor& operator*=(const Scalar& val);

        Tensor& operator/=(const Scalar& val);

        /**********************************************************************
         *
         * Binary operations (multiplication and division)
         *
         *********************************************************************/

        Tensor& operator=(const TensorMult& other);

        Tensor& operator+=(const TensorMult& other);

        Tensor& operator-=(const TensorMult& other);

        /**********************************************************************
         *
         * Unary operations (assignment, summation, scaling, and inversion)
         *
         *********************************************************************/

        Tensor& operator=(const Tensor& other);

        Tensor& operator+=(const Tensor& other);

        Tensor& operator-=(const Tensor& other);

        Tensor& operator*=(const Tensor& other);

        Tensor& operator=(const ConstScaledTensor& other);

        Tensor& operator+=(const ConstScaledTensor& other);

        Tensor& operator-=(const ConstScaledTensor& other);

        Tensor& operator*=(const ConstScaledTensor& other);

        /**********************************************************************
         *
         * Intermediate operations
         *
         *********************************************************************/

        friend ScaledTensor operator*(const Scalar& factor, Tensor& other);

        friend ConstScaledTensor operator*(const Scalar& factor, const Tensor& other);

        ScaledTensor operator*(const Scalar& factor);

        ConstScaledTensor operator*(const Scalar& factor) const;

        ScaledTensor operator/(const Scalar& factor);

        ConstScaledTensor operator/(const Scalar& factor) const;

        ScaledTensor operator-();

        ConstScaledTensor operator-() const;

        friend ConstScaledTensor conj(const Tensor& t);

        TensorMult operator*(const Tensor& other) const;
};

/**************************************************************************************************
 *
 * Generic Tensor Wrapper
 *
 *************************************************************************************************/

/*
 * Workhorse wrapper class for arbitrary capabilities; automatically exposes
 * interfaces for defined capabilities and delegates to user-defined implementation
 */
template <capability_type C>
class Tensor<C, typename std::enable_if<(C>0)>::type>
    : public TensorWrapperBase<C>
{
    public:
        /*
         * Initialize from an actual implementation. It will be destroyed when
         * the last wrapper referencing it dies.
         */
        Tensor(TensorImplementation<>* t)
        : TensorWrapperBase<C>(*t)
        {
            assert(IS_SUPERSET_OF(C,t->C));
        }

        /*
         * Re-wrap another Tensor wrapper. Capabilities are checked at
         * run-time to enable "remembering" lost capabilties.
         */
        Tensor(const Tensor<>& t)
        : TensorWrapperBase<C>(const_cast<Tensor<>&>(t).ptr)
        {
            assert(IS_SUPERSET_OF(C,t.C));
        }

        /*
         * Copy ctor. Wrap the same implementation with the
         * same capabilities.
         */
        Tensor(const Tensor& t)
        : TensorWrapperBase<C>(const_cast<Tensor<>&>(t).ptr) {}

        Tensor<>& operator=(const Tensor<>& other)
        {
            return static_cast<Tensor<>&>(*this).operator=(other);
        }

        Tensor<>& operator=(const Tensor& other)
        {
            return static_cast<Tensor<>&>(*this).operator=(other);
        }
};

/**************************************************************************************************
 *
 * Intermediate Wrappers
 *
 *************************************************************************************************/

class ConstScaledTensor
{
    public:
        TensorImplementation<>& tensor;
        Scalar factor;
        bool conj;

        ConstScaledTensor(const TensorImplementation<>& tensor, const Scalar& factor, bool conj=false)
        : tensor(const_cast<TensorImplementation<>&>(tensor), factor(factor), conj(conj) {}

        /**********************************************************************
         *
         * Unary negation, conjugation
         *
         *********************************************************************/

        ConstScaledTensor operator-() const;

        friend ConstScaledTensor conj(const ConstScaledTensor& st);

        /**********************************************************************
         *
         * Binary tensor operations
         *
         *********************************************************************/

        TensorMult operator*(const ConstScaledTensor& other) const;

        TensorMult operator*(const Tensor<>& other) const;

        friend TensorMult operator*(const Tensor<>& t, const ConstScaledTensor& other);

        /**********************************************************************
         *
         * Operations with scalars
         *
         *********************************************************************/

        ConstScaledTensor operator*(const Scalar& factor) const;

        friend ConstScaledTensor operator*(const Scalar& factor, const ConstScaledTensor& other);

        ConstScaledTensor operator/(const Scalar& factor) const;
};

class ScaledTensor : public ConstScaledTensor
{
    public:
        ScaledTensor(const ConstScaledTensor& other)
        : ConstScaledTensor(other) {}

        ScaledTensor(const ScaledTensor& other)
        : ConstScaledTensor(other) {}

        ScaledTensor(Tensor<>& tensor, const Scalar& factor, bool conj=false)
        : ConstScaledTensor(tensor, factor, conj) {}

        /**********************************************************************
         *
         * Unary negation, conjugation
         *
         *********************************************************************/

        ScaledTensor operator-() const;

        /**********************************************************************
         *
         * Unary tensor operations
         *
         *********************************************************************/

        ScaledTensor& operator+=(const Tensor<>& other);

        ScaledTensor& operator-=(const Tensor<>& other);

        ScaledTensor& operator*=(const Tensor<>& other);

        ScaledTensor& operator=(const ScaledTensor& other);

        ScaledTensor& operator=(const ConstScaledTensor& other);

        ScaledTensor& operator+=(const ConstScaledTensor& other);

        ScaledTensor& operator-=(const ConstScaledTensor& other);

        ScaledTensor& operator*=(const ConstScaledTensor& other);

        /**********************************************************************
         *
         * Binary tensor operations
         *
         *********************************************************************/

        ScaledTensor& operator=(const TensorMult& other);

        ScaledTensor& operator+=(const TensorMult& other);

        ScaledTensor& operator-=(const TensorMult& other);

        /**********************************************************************
         *
         * Operations with scalars
         *
         *********************************************************************/

        ScaledTensor operator*(const Scalar& factor) const;

        friend ScaledTensor operator*(const Scalar& factor, const ScaledTensor& other);

        ScaledTensor operator/(const Scalar& factor) const;

        ScaledTensor& operator*=(const Scalar& val);

        ScaledTensor& operator/=(const Scalar& val);
};

class TensorMult
{
    private:
        const TensorMult& operator=(const TensorMult& other);

    public:
        TensorImplementation<>& A;
        TensorImplementation<>& B;
        bool conja;
        bool conjb;
        Scalar factor;

        TensorMult(const ConstScaledTensor& A, const ConstScaledTensor& B)
        : A(A.tensor), conja(A.conj), B(B.tensor), conjb(B.conj), factor(A.factor*B.factor) {}

        /**********************************************************************
         *
         * Unary negation, conjugation
         *
         *********************************************************************/

        TensorMult operator-() const;

        friend TensorMult conj(const TensorMult& tm);

        /**********************************************************************
         *
         * Operations with scalars
         *
         *********************************************************************/

        TensorMult operator*(const Scalar& factor) const;

        TensorMult operator/(const Scalar& factor) const;

        friend TensorMult operator*(const Scalar& factor, const TensorMult& other);
};

/**************************************************************************************************
 *
 * Exceptions
 *
 *************************************************************************************************/

class TensorError : public std::exception
{
    public:
        virtual const char* what() const throw() = 0;
};

class OutOfBoundsError : public TensorError
{
    public:
        virtual const char* what() const throw() { return "out-of-bounds read or write"; }
};

class LengthMismatchError : public TensorError
{
    public:
        virtual const char* what() const throw() { return "length mismatch error"; }
};

class IndexMismatchError : public TensorError
{
    public:
        virtual const char* what() const throw() { return "index mismatch error"; }
};

class InvalidNdimError : public TensorError
{
    public:
        virtual const char* what() const throw() { return "invalid number of dimensions"; }
};

class InvalidLengthError : public TensorError
{
    public:
        virtual const char* what() const throw() { return "invalid length"; }
};

class InvalidLdError : public TensorError
{
    public:
        virtual const char* what() const throw() { return "invalid leading dimension"; }
};

class LdTooSmallError : public TensorError
{
    public:
        virtual const char* what() const throw() { return "leading dimension is too small"; }
};

class SymmetryMismatchError : public TensorError
{
    public:
        virtual const char* what() const throw() { return "symmetry mismatch error"; }
};

class InvalidSymmetryError : public TensorError
{
    public:
        virtual const char* what() const throw() { return "invalid symmetry value"; }
};

class InvalidStartError : public TensorError
{
    public:
        virtual const char* what() const throw() { return "invalid start value"; }
};

}

inline Scalar scalar(const tensor::TensorMult& tm)
{
    return tm.factor*tm.B.dot(tm.conja, tm.A, tm.conjb);
}

}

#endif
