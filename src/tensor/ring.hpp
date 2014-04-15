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

#ifndef _AQUARIUS_TENSOR_RING_HPP_
#define _AQUARIUS_TENSOR_RING_HPP_

#include <stdexcept>
#include <iostream>
#include <string>
#include <string.h>
#include <map>

#include "util/stl_ext.hpp"
#include "util/util.h"
#include "util/blas.h"

namespace aquarius
{

struct Ring
{
    const int type;

    Ring(int type) : type(type) {}

    bool operator==(const Ring& other) { return type == other.type; }
};

struct Field : Ring
{
    enum field {SINGLE, DOUBLE, LDOUBLE, SCOMPLEX, DCOMPLEX, LDCOMPLEX};

    Field(field F) : Ring(F) {}

    Field(const              float& val)        : Ring(SINGLE) {}
    Field(const              double& val)       : Ring(DOUBLE) {}
    Field(const              long double& val)  : Ring(LDOUBLE) {}
    Field(const std::complex<float>& val)       : Ring(SCOMPLEX) {}
    Field(const std::complex<double>& val)      : Ring(DCOMPLEX) {}
    Field(const std::complex<long double>& val) : Ring(LDCOMPLEX) {}
};

class Scalar
{
    friend void swap(Scalar& s1, Scalar& s2)
    {
        Scalar tmp;
        memcpy(&tmp, & s1, sizeof(Scalar));
        memcpy(& s1, & s2, sizeof(Scalar));
        memcpy(& s2, &tmp, sizeof(Scalar));
    }

    protected:
        Field F;
        char data[sizeof(std::complex<long double>)];

        Scalar() : F(Field::DOUBLE) {}

        template <typename T>
        Scalar(Field::field type, T val) : F(type)
        {
            *this = val;
        }

        template <typename T>
        Field::field resultType(T other) const
        {
            /*
             * If the other argument is from a real or complex field, up-convert to
             * complex if either argument is and set the floating-point type to the
             * widest of the operands. Otherwise, keep the field's type.
             */
            return Field::field(is_field<T>::value
                        ? (F.type >= Field::SCOMPLEX || std::is_complex<T>::value ? Field::SCOMPLEX : 0)+
                           std::max(F.type >= Field::SCOMPLEX ? F.type-Field::SCOMPLEX : F.type,
                                    Field((Field::field)typename std::real_type<T>::type()).type)
                        : F.type);
        }

                           float &   fval() { return *(                   float *)(&data[0]); }
                          double &   dval() { return *(                  double *)(&data[0]); }
                     long double &  ldval() { return *(             long double *)(&data[0]); }
        std::complex<      float>&  fcval() { return *(std::complex<      float>*)(&data[0]); }
        std::complex<     double>&  dcval() { return *(std::complex<     double>*)(&data[0]); }
        std::complex<long double>& ldcval() { return *(std::complex<long double>*)(&data[0]); }

        const                    float &   fval() const { return *(                   float *)(&data[0]); }
        const                   double &   dval() const { return *(                  double *)(&data[0]); }
        const              long double &  ldval() const { return *(             long double *)(&data[0]); }
        const std::complex<      float>&  fcval() const { return *(std::complex<      float>*)(&data[0]); }
        const std::complex<     double>&  dcval() const { return *(std::complex<     double>*)(&data[0]); }
        const std::complex<long double>& ldcval() const { return *(std::complex<long double>*)(&data[0]); }

        template <typename T>
        struct is_field : std::is_floating_point<typename std::real_type<T>::type> {};

    public:
        template <typename T>
        Scalar(T val) : F(is_field<T>::value ? val : double())
        {
            STATIC_ASSERT(std::is_arithmetic<T>::value);
            *this = val;
        }

        template <typename T> operator T() const
        {
            STATIC_ASSERT(is_field<T>::value);
            return to<T>();
        }

        template <typename T> typename std::enable_if<std::is_complex<T>::value,T>::type to() const
        {
            switch (F.type)
            {
                case Field::SINGLE:    return T(  fval()); break;
                case Field::DOUBLE:    return T(  dval()); break;
                case Field::LDOUBLE:   return T( ldval()); break;
                case Field::SCOMPLEX:  return T( fcval()); break;
                case Field::DCOMPLEX:  return T( dcval()); break;
                case Field::LDCOMPLEX: return T(ldcval()); break;
            }
        }

        template <typename T> typename std::enable_if<!std::is_complex<T>::value,T>::type to() const
        {
            switch (F.type)
            {
                case Field::SINGLE:    return T(           fval() ); break;
                case Field::DOUBLE:    return T(           dval() ); break;
                case Field::LDOUBLE:   return T(          ldval() ); break;
                case Field::SCOMPLEX:  return T(std::abs( fcval())); break;
                case Field::DCOMPLEX:  return T(std::abs( dcval())); break;
                case Field::LDCOMPLEX: return T(std::abs(ldcval())); break;
            }
        }

        template <typename T>
        typename std::enable_if<std::is_complex<T>::value,Scalar&>::type operator=(T other)
        {
            STATIC_ASSERT(std::is_arithmetic<typename std::real_type<T>::type>::value);

            switch (F.type)
            {
                case Field::SINGLE:      fval() = std::abs(other); break;
                case Field::DOUBLE:      dval() = std::abs(other); break;
                case Field::LDOUBLE:    ldval() = std::abs(other); break;
                case Field::SCOMPLEX:   fcval() =          other ; break;
                case Field::DCOMPLEX:   dcval() =          other ; break;
                case Field::LDCOMPLEX: ldcval() =          other ; break;
            }

            return *this;
        }

        template <typename T>
        typename std::enable_if<!std::is_complex<T>::value,Scalar&>::type operator=(T other)
        {
            STATIC_ASSERT(std::is_arithmetic<typename std::real_type<T>::type>::value);

            switch (F.type)
            {
                case Field::SINGLE:      fval() = other; break;
                case Field::DOUBLE:      dval() = other; break;
                case Field::LDOUBLE:    ldval() = other; break;
                case Field::SCOMPLEX:   fcval() = other; break;
                case Field::DCOMPLEX:   dcval() = other; break;
                case Field::LDCOMPLEX: ldcval() = other; break;
            }

            return *this;
        }

        Scalar& operator=(Scalar other)
        {
            switch (other.F.type)
            {
                case Field::SINGLE:    *this =   other.fval(); break;
                case Field::DOUBLE:    *this =   other.dval(); break;
                case Field::LDOUBLE:   *this =  other.ldval(); break;
                case Field::SCOMPLEX:  *this =  other.fcval(); break;
                case Field::DCOMPLEX:  *this =  other.dcval(); break;
                case Field::LDCOMPLEX: *this = other.ldcval(); break;
            }

            return *this;
        }

        template <typename T>
        typename std::enable_if<std::is_complex<T>::value,Scalar&>::type operator+=(T other)
        {
            STATIC_ASSERT(std::is_arithmetic<typename std::real_type<T>::type>::value);

            switch (F.type)
            {
                case Field::SINGLE:      fval() += std::abs(other); break;
                case Field::DOUBLE:      dval() += std::abs(other); break;
                case Field::LDOUBLE:    ldval() += std::abs(other); break;
                case Field::SCOMPLEX:   fcval() +=          other ; break;
                case Field::DCOMPLEX:   dcval() +=          other ; break;
                case Field::LDCOMPLEX: ldcval() +=          other ; break;
            }

            return *this;
        }

        template <typename T>
        typename std::enable_if<!std::is_complex<T>::value,Scalar&>::type operator+=(T other)
        {
            STATIC_ASSERT(std::is_arithmetic<typename std::real_type<T>::type>::value);

            switch (F.type)
            {
                case Field::SINGLE:      fval() += other; break;
                case Field::DOUBLE:      dval() += other; break;
                case Field::LDOUBLE:    ldval() += other; break;
                case Field::SCOMPLEX:   fcval() += other; break;
                case Field::DCOMPLEX:   dcval() += other; break;
                case Field::LDCOMPLEX: ldcval() += other; break;
            }

            return *this;
        }

        Scalar& operator+=(Scalar other)
        {
            switch (other.F.type)
            {
                case Field::SINGLE:    *this +=   other.fval(); break;
                case Field::DOUBLE:    *this +=   other.dval(); break;
                case Field::LDOUBLE:   *this +=  other.ldval(); break;
                case Field::SCOMPLEX:  *this +=  other.fcval(); break;
                case Field::DCOMPLEX:  *this +=  other.dcval(); break;
                case Field::LDCOMPLEX: *this += other.ldcval(); break;
            }

            return *this;
        }

        template <typename T>
        typename std::enable_if<std::is_complex<T>::value,Scalar&>::type operator-=(T other)
        {
            STATIC_ASSERT(std::is_arithmetic<typename std::real_type<T>::type>::value);

            switch (F.type)
            {
                case Field::SINGLE:      fval() -= std::abs(other); break;
                case Field::DOUBLE:      dval() -= std::abs(other); break;
                case Field::LDOUBLE:    ldval() -= std::abs(other); break;
                case Field::SCOMPLEX:   fcval() -=          other ; break;
                case Field::DCOMPLEX:   dcval() -=          other ; break;
                case Field::LDCOMPLEX: ldcval() -=          other ; break;
            }

            return *this;
        }

        template <typename T>
        typename std::enable_if<!std::is_complex<T>::value,Scalar&>::type operator-=(T other)
        {
            STATIC_ASSERT(std::is_arithmetic<typename std::real_type<T>::type>::value);

            switch (F.type)
            {
                case Field::SINGLE:      fval() -= other; break;
                case Field::DOUBLE:      dval() -= other; break;
                case Field::LDOUBLE:    ldval() -= other; break;
                case Field::SCOMPLEX:   fcval() -= other; break;
                case Field::DCOMPLEX:   dcval() -= other; break;
                case Field::LDCOMPLEX: ldcval() -= other; break;
            }

            return *this;
        }

        Scalar& operator-=(Scalar other)
        {
            switch (other.F.type)
            {
                case Field::SINGLE:    *this -=   other.fval(); break;
                case Field::DOUBLE:    *this -=   other.dval(); break;
                case Field::LDOUBLE:   *this -=  other.ldval(); break;
                case Field::SCOMPLEX:  *this -=  other.fcval(); break;
                case Field::DCOMPLEX:  *this -=  other.dcval(); break;
                case Field::LDCOMPLEX: *this -= other.ldcval(); break;
            }

            return *this;
        }

        template <typename T>
        typename std::enable_if<std::is_complex<T>::value,Scalar&>::type operator*=(T other)
        {
            STATIC_ASSERT(std::is_arithmetic<typename std::real_type<T>::type>::value);

            switch (F.type)
            {
                case Field::SINGLE:      fval() *= std::abs(other); break;
                case Field::DOUBLE:      dval() *= std::abs(other); break;
                case Field::LDOUBLE:    ldval() *= std::abs(other); break;
                case Field::SCOMPLEX:   fcval() *=          other ; break;
                case Field::DCOMPLEX:   dcval() *=          other ; break;
                case Field::LDCOMPLEX: ldcval() *=          other ; break;
            }

            return *this;
        }

        template <typename T>
        typename std::enable_if<!std::is_complex<T>::value,Scalar&>::type operator*=(T other)
        {
            STATIC_ASSERT(std::is_arithmetic<typename std::real_type<T>::type>::value);

            switch (F.type)
            {
                case Field::SINGLE:      fval() *= other; break;
                case Field::DOUBLE:      dval() *= other; break;
                case Field::LDOUBLE:    ldval() *= other; break;
                case Field::SCOMPLEX:   fcval() *= other; break;
                case Field::DCOMPLEX:   dcval() *= other; break;
                case Field::LDCOMPLEX: ldcval() *= other; break;
            }

            return *this;
        }

        Scalar& operator*=(Scalar other)
        {
            switch (other.F.type)
            {
                case Field::SINGLE:    *this *=   other.fval(); break;
                case Field::DOUBLE:    *this *=   other.dval(); break;
                case Field::LDOUBLE:   *this *=  other.ldval(); break;
                case Field::SCOMPLEX:  *this *=  other.fcval(); break;
                case Field::DCOMPLEX:  *this *=  other.dcval(); break;
                case Field::LDCOMPLEX: *this *= other.ldcval(); break;
            }

            return *this;
        }

        template <typename T>
        typename std::enable_if<std::is_complex<T>::value,Scalar&>::type operator/=(T other)
        {
            STATIC_ASSERT(std::is_arithmetic<typename std::real_type<T>::type>::value);

            switch (F.type)
            {
                case Field::SINGLE:      fval() /= std::abs(other); break;
                case Field::DOUBLE:      dval() /= std::abs(other); break;
                case Field::LDOUBLE:    ldval() /= std::abs(other); break;
                case Field::SCOMPLEX:   fcval() /=          other ; break;
                case Field::DCOMPLEX:   dcval() /=          other ; break;
                case Field::LDCOMPLEX: ldcval() /=          other ; break;
            }

            return *this;
        }

        template <typename T>
        typename std::enable_if<!std::is_complex<T>::value,Scalar&>::type operator/=(T other)
        {
            STATIC_ASSERT(std::is_arithmetic<typename std::real_type<T>::type>::value);

            switch (F.type)
            {
                case Field::SINGLE:      fval() /= other; break;
                case Field::DOUBLE:      dval() /= other; break;
                case Field::LDOUBLE:    ldval() /= other; break;
                case Field::SCOMPLEX:   fcval() /= other; break;
                case Field::DCOMPLEX:   dcval() /= other; break;
                case Field::LDCOMPLEX: ldcval() /= other; break;
            }

            return *this;
        }

        Scalar& operator/=(Scalar other)
        {
            switch (other.F.type)
            {
                case Field::SINGLE:    *this /=   other.fval(); break;
                case Field::DOUBLE:    *this /=   other.dval(); break;
                case Field::LDOUBLE:   *this /=  other.ldval(); break;
                case Field::SCOMPLEX:  *this /=  other.fcval(); break;
                case Field::DCOMPLEX:  *this /=  other.dcval(); break;
                case Field::LDCOMPLEX: *this /= other.ldcval(); break;
            }

            return *this;
        }

        Scalar operator-() const
        {
            Scalar n(*this);

            switch (n.F.type)
            {
                case Field::SINGLE:      n.fval() =   -n.fval(); break;
                case Field::DOUBLE:      n.dval() =   -n.dval(); break;
                case Field::LDOUBLE:    n.ldval() =  -n.ldval(); break;
                case Field::SCOMPLEX:   n.fcval() =  -n.fcval(); break;
                case Field::DCOMPLEX:   n.dcval() =  -n.dcval(); break;
                case Field::LDCOMPLEX: n.ldcval() = -n.ldcval(); break;
            }

            return n;
        }

        template <typename T>
        Scalar operator+(T other) const
        {
            return other+(*this);
        }

        template <typename T> friend
        Scalar operator+(T other, const Scalar& s)
        {
            STATIC_ASSERT(std::is_arithmetic<typename std::real_type<T>::type>::value);

            Field::field new_type = s.resultType(other);

            switch (s.F.type)
            {
                case Field::SINGLE:    return Scalar(new_type, other+  s.fval()); break;
                case Field::DOUBLE:    return Scalar(new_type, other+  s.dval()); break;
                case Field::LDOUBLE:   return Scalar(new_type, other+ s.ldval()); break;
                case Field::SCOMPLEX:  return Scalar(new_type, other+ s.fcval()); break;
                case Field::DCOMPLEX:  return Scalar(new_type, other+ s.dcval()); break;
                case Field::LDCOMPLEX: return Scalar(new_type, other+s.ldcval()); break;
            }

            return Scalar(0.0);
        }

        Scalar operator+(const Scalar& other) const
        {
            switch (other.F.type)
            {
                case Field::SINGLE:    return *this +   other.fval(); break;
                case Field::DOUBLE:    return *this +   other.dval(); break;
                case Field::LDOUBLE:   return *this +  other.ldval(); break;
                case Field::SCOMPLEX:  return *this +  other.fcval(); break;
                case Field::DCOMPLEX:  return *this +  other.dcval(); break;
                case Field::LDCOMPLEX: return *this + other.ldcval(); break;
            }

            return Scalar(0.0);
        }

        template <typename T>
        Scalar operator-(T other) const
        {
            STATIC_ASSERT(std::is_arithmetic<typename std::real_type<T>::type>::value);

            Field::field new_type = resultType(other);

            switch (F.type)
            {
                case Field::SINGLE:    return Scalar(new_type,   fval()-other); break;
                case Field::DOUBLE:    return Scalar(new_type,   dval()-other); break;
                case Field::LDOUBLE:   return Scalar(new_type,  ldval()-other); break;
                case Field::SCOMPLEX:  return Scalar(new_type,  fcval()-other); break;
                case Field::DCOMPLEX:  return Scalar(new_type,  dcval()-other); break;
                case Field::LDCOMPLEX: return Scalar(new_type, ldcval()-other); break;
            }

            return Scalar(0.0);
        }

        template <typename T> friend
        Scalar operator-(T other, const Scalar& s)
        {
            STATIC_ASSERT(std::is_arithmetic<typename std::real_type<T>::type>::value);

            Field::field new_type = s.resultType(other);

            switch (s.F.type)
            {
                case Field::SINGLE:    return Scalar(new_type, other-  s.fval()); break;
                case Field::DOUBLE:    return Scalar(new_type, other-  s.dval()); break;
                case Field::LDOUBLE:   return Scalar(new_type, other- s.ldval()); break;
                case Field::SCOMPLEX:  return Scalar(new_type, other- s.fcval()); break;
                case Field::DCOMPLEX:  return Scalar(new_type, other- s.dcval()); break;
                case Field::LDCOMPLEX: return Scalar(new_type, other-s.ldcval()); break;
            }

            return Scalar(0.0);
        }

        Scalar operator-(const Scalar& other) const
        {
            switch (other.F.type)
            {
                case Field::SINGLE:    return *this -   other.fval(); break;
                case Field::DOUBLE:    return *this -   other.dval(); break;
                case Field::LDOUBLE:   return *this -  other.ldval(); break;
                case Field::SCOMPLEX:  return *this -  other.fcval(); break;
                case Field::DCOMPLEX:  return *this -  other.dcval(); break;
                case Field::LDCOMPLEX: return *this - other.ldcval(); break;
            }

            return Scalar(0.0);
        }

        template <typename T>
        Scalar operator*(T other) const
        {
            return other*(*this);
        }

        template <typename T> friend
        Scalar operator*(T other, const Scalar& s)
        {
            STATIC_ASSERT(std::is_arithmetic<typename std::real_type<T>::type>::value);

            Field::field new_type = s.resultType(other);

            switch (s.F.type)
            {
                case Field::SINGLE:    return Scalar(new_type, other*  s.fval()); break;
                case Field::DOUBLE:    return Scalar(new_type, other*  s.dval()); break;
                case Field::LDOUBLE:   return Scalar(new_type, other* s.ldval()); break;
                case Field::SCOMPLEX:  return Scalar(new_type, other* s.fcval()); break;
                case Field::DCOMPLEX:  return Scalar(new_type, other* s.dcval()); break;
                case Field::LDCOMPLEX: return Scalar(new_type, other*s.ldcval()); break;
            }

            return Scalar(0.0);
        }

        Scalar operator*(const Scalar& other) const
        {
            switch (other.F.type)
            {
                case Field::SINGLE:    return *this *   other.fval(); break;
                case Field::DOUBLE:    return *this *   other.dval(); break;
                case Field::LDOUBLE:   return *this *  other.ldval(); break;
                case Field::SCOMPLEX:  return *this *  other.fcval(); break;
                case Field::DCOMPLEX:  return *this *  other.dcval(); break;
                case Field::LDCOMPLEX: return *this * other.ldcval(); break;
            }

            return Scalar(0.0);
        }

        template <typename T>
        Scalar operator/(T other) const
        {
            STATIC_ASSERT(std::is_arithmetic<typename std::real_type<T>::type>::value);

            Field::field new_type = resultType(other);

            switch (F.type)
            {
                case Field::SINGLE:    return Scalar(new_type,   fval()/other); break;
                case Field::DOUBLE:    return Scalar(new_type,   dval()/other); break;
                case Field::LDOUBLE:   return Scalar(new_type,  ldval()/other); break;
                case Field::SCOMPLEX:  return Scalar(new_type,  fcval()/other); break;
                case Field::DCOMPLEX:  return Scalar(new_type,  dcval()/other); break;
                case Field::LDCOMPLEX: return Scalar(new_type, ldcval()/other); break;
            }

            return Scalar(0.0);
        }

        template <typename T> friend
        Scalar operator/(T other, const Scalar& s)
        {
            STATIC_ASSERT(std::is_arithmetic<typename std::real_type<T>::type>::value);

            Field::field new_type = s.resultType(other);

            switch (s.F.type)
            {
                case Field::SINGLE:    return Scalar(new_type, other/  s.fval()); break;
                case Field::DOUBLE:    return Scalar(new_type, other/  s.dval()); break;
                case Field::LDOUBLE:   return Scalar(new_type, other/ s.ldval()); break;
                case Field::SCOMPLEX:  return Scalar(new_type, other/ s.fcval()); break;
                case Field::DCOMPLEX:  return Scalar(new_type, other/ s.dcval()); break;
                case Field::LDCOMPLEX: return Scalar(new_type, other/s.ldcval()); break;
            }

            return Scalar(0.0);
        }

        Scalar operator/(const Scalar& other) const
        {
            switch (other.F.type)
            {
                case Field::SINGLE:    return *this /   other.fval(); break;
                case Field::DOUBLE:    return *this /   other.dval(); break;
                case Field::LDOUBLE:   return *this /  other.ldval(); break;
                case Field::SCOMPLEX:  return *this /  other.fcval(); break;
                case Field::DCOMPLEX:  return *this /  other.dcval(); break;
                case Field::LDCOMPLEX: return *this / other.ldcval(); break;
            }

            return Scalar(0.0);
        }
};

}

#endif
