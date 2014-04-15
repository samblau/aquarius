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

#ifndef _AQUARIUS_UTIL_ARRAY_HPP_
#define _AQUARIUS_UTIL_ARRAY_HPP_

#include <vector>
#include <cstdlib>
#include <cassert>
#include <ostream>
#include <algorithm>

#include "stl_ext.hpp"

template <class T, int ndim>
class myarray;

template <class T, int ndim, int dim>
class myarray_ref
{
    friend class myarray_ref<T,ndim,dim-1>;
    friend class myarray<T,ndim>;

    private:
        myarray_ref& operator=(const myarray_ref& other);

    protected:
        myarray<T, ndim>& array;
        size_t idx;

        myarray_ref(const myarray_ref& other)
        : array(other.array), idx(other.idx) {}

        myarray_ref(myarray<T, ndim>& array, size_t idx, int i);

    public:
        myarray_ref<T,ndim,dim+1> operator[](int i);

        const myarray_ref<T,ndim,dim+1> operator[](int i) const;
};

template <class T, int ndim>
class myarray_ref<T, ndim, ndim>
{
    friend class myarray_ref<T,ndim,ndim-1>;
    friend class myarray<T,ndim>;

    private:
        myarray_ref& operator=(const myarray_ref& other);

    protected:
        myarray<T, ndim>& array;
        size_t idx;

        myarray_ref(const myarray_ref& other)
        : array(other.array), idx(other.idx) {}

        myarray_ref(myarray<T, ndim>& array, size_t idx, int i);

    public:
        T& operator[](int i);

        const T& operator[](int i) const;
};

template <class T, int ndim>
class myarray
{
    template<class T_, int ndim_, int dim_> friend class myarray_ref;

    friend void swap(myarray& a, myarray& b)
    {
        using std::swap;
        swap(a.data_, b.data_);
        swap(a.len, b.len);
        swap(a.stride, b.stride);
    }

    protected:
        T* data_;
        std::vector<int> len;
        std::vector<size_t> stride;

    public:
        enum Layout {COLUMN_MAJOR, ROW_MAJOR};

        explicit myarray(const myarray& other)
        : len(other.len), stride(other.stride)
        {
            size_t num = 1;
            for (int i = 0;i < ndim;i++) num *= len[i];
            data_ = new T[num];
            std::copy(other.data_, other.data_+num, data_);
        }

        explicit myarray(const std::vector<int>& len, Layout layout = ROW_MAJOR)
        : len(len), stride(ndim)
        {
            assert(len.size() == ndim);

            if (layout == ROW_MAJOR)
            {
                stride[ndim-1] = 1;
                for (int i = ndim-2;i >= 0;i--)
                {
                    stride[i] = stride[i+1]*len[i+1];
                }
                data_ = new T[stride[0]*len[0]]();
            }
            else
            {
                stride[0] = 1;
                for (int i = 1;i < ndim;i++)
                {
                    stride[i] = stride[i-1]*len[i-1];
                }
                data_ = new T[stride[ndim-1]*len[ndim-1]]();
            }
        }

        ~myarray()
        {
            delete[] data_;
        }

        myarray& operator=(myarray other)
        {
            swap(*this, other);
            return *this;
        }

        myarray_ref<T,ndim,2> operator[](int i)
        {
            return myarray_ref<T,ndim,2>(*this, (size_t)0, i);
        }

        const myarray_ref<T,ndim,2> operator[](int i) const
        {
            return myarray_ref<T,ndim,2>(const_cast<myarray&>(*this), (size_t)0, i);
        }

        T* data() { return data_; }

        const T* data() const { return data_; }

		const int* length() const { return len.data(); }
};

template <class T>
class myarray<T,0>
{
    protected:
        T data_;

    public:
        myarray(T val = 0) : data_(val) {}

        operator T&() { return data_; }

        operator const T&() const { return data_; }

        T* data() { return &data_; }

        const T* data() const { return &data_; }
};

template <class T>
class myarray<T,1>
{
    friend void swap(myarray& a, myarray& b)
    {
        using std::swap;
        swap(a.data_, b.data_);
        swap(a.len, b.len);
    }

    protected:
        T* data_;
        int len;

    public:
        myarray(const myarray& other)
        : len(other.len)
        {
            data_ = new T[len];
            std::copy(other.data_, other.data_+len, data_);
        }

        explicit myarray(int n)
        : len(n)
        {
            data_ = new T[len]();
        }

        ~myarray()
        {
            delete[] data_;
        }

        myarray& operator=(myarray other)
        {
            swap(*this, other);
            return *this;
        }

        T& operator[](int i)
        {
            return data_[i];
        }

        const T& operator[](int i) const
        {
            return data_[i];
        }

        T* data() { return data_; }

        const T* data() const { return data_; }
};

template <class T, int ndim, int dim>
myarray_ref<T,ndim,dim>::myarray_ref(myarray<T, ndim>& array, size_t idx, int i)
: array(array), idx(idx+i*array.stride[dim-2]) {}

template <class T, int ndim, int dim>
myarray_ref<T,ndim,dim+1> myarray_ref<T,ndim,dim>::operator[](int i)
{
    return myarray_ref<T,ndim,dim+1>(array, idx, i);
}

template <class T, int ndim, int dim>
const myarray_ref<T,ndim,dim+1> myarray_ref<T,ndim,dim>::operator[](int i) const
{
    return myarray_ref<T,ndim,dim+1>(array, idx, i);
}

template <class T, int ndim>
myarray_ref<T,ndim,ndim>::myarray_ref(myarray<T, ndim>& array, size_t idx, int i)
: array(array), idx(idx+i*array.stride[ndim-2]) {}

template <class T, int ndim>
T& myarray_ref<T,ndim,ndim>::operator[](int i)
{
    return array.data_[idx+i*array.stride[ndim-1]];
}

template <class T, int ndim>
const T& myarray_ref<T,ndim,ndim>::operator[](int i) const
{
    return array.data_[idx+i*array.stride[ndim-1]];
}

template <class T>
class scalar : public myarray<T,0>
{
    public:
        scalar(T val = 0) : myarray<T,0>(val) {}
};

template <class T>
class matrix : public myarray<T,2>
{
    public:
        matrix(int m, int n, typename myarray<T,2>::Layout layout = myarray<T,2>::ROW_MAJOR)
        : myarray<T,2>(std::vec(m,n), layout) {}
};

#define array myarray

#endif
