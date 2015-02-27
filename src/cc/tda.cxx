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

#include "tda.hpp"
#include "util/lapack.h"

#ifdef ELEMENTAL
using namespace El;
#endif

using namespace std;
using namespace aquarius;
using namespace aquarius::op;
using namespace aquarius::cc;
using namespace aquarius::input;
using namespace aquarius::tensor;
using namespace aquarius::task;
using namespace aquarius::symmetry;

template <typename U>
TDA<U>::TDA(const std::string& name, const Config& config)
: Task("tda", name)
{
    vector<Requirement> reqs;
    reqs.push_back(Requirement("molecule", "molecule"));
    reqs.push_back(Requirement("moints", "H"));
    addProduct(Product("tda.TDAevals", "TDAevals", reqs));
    addProduct(Product("tda.TDAevecs", "TDAevecs", reqs));
}

template <typename U>
void TDA<U>::run(TaskDAG& dag, const Arena& arena)
{
    const Molecule& molecule = get<Molecule>("molecule");
    const PointGroup& group = molecule.getGroup();
    int nirrep = group.getNumIrreps();

    TwoElectronOperator<U>& W = get<TwoElectronOperator<U> >("H");
    const Space& occ = W.occ;
    const Space& vrt = W.vrt;

    // // Code for checking sparcity:  
    // const SpinorbitalTensor<U>& WABCD = W.getABCD();
    // const SymmetryBlockedTensor<U>& ABAB = WABCD({1,0},{1,0});

    // for (int R = 0;R < nirrep;R++)
    // {
    //     int num_total = 0;
    //     int num_zero = 0;
    //     const Representation& irr_R = group.getIrrep(R);
    //     for (int a = 0;a < nirrep;a++)
    //     {
    //         const Representation& irr_a = group.getIrrep(a);
    //         for (int b = 0;b < nirrep;b++)
    //         {
    //             const Representation& irr_b = group.getIrrep(b);
    //             if (!(irr_a*irr_b*irr_R).isTotallySymmetric()) continue;

    //             for (int c = 0;c < nirrep;c++)
    //             {
    //                 const Representation& irr_c = group.getIrrep(c);
    //                 for (int d = 0;d < nirrep;d++)
    //                 {
    //                     const Representation& irr_d = group.getIrrep(d);
    //                     if (!(irr_c*irr_d*irr_R).isTotallySymmetric()) continue;

    //                     CTFTensor<U> this_tensor = ABAB({a,b,c,d});
    //                     vector<U> temp;
    //                     this_tensor.getAllData(temp);
    //                     for (int i = 0; i < temp.size(); i++)
    //                     {
    //                         num_total++;
    //                         if (abs(temp[i]) < 1e-10)
    //                             num_zero++;
    //                     }
    //                 }
    //             }
    //         }
    //     }
    //     cout << "num_total = " << num_total << endl;
    //     cout << "num_zero = " << num_zero << endl;
    //     cout << "num_zero / num_total = " << float(num_zero) / float(num_total) << endl;
    // }

    SpinorbitalTensor<U> Hguess("Hguess", arena, group, {vrt,occ}, {1,1}, {1,1});
    Hguess = 0;

    const SpinorbitalTensor<U>& FAB = W.getAB();
    const SpinorbitalTensor<U>& FIJ = W.getIJ();
    const SpinorbitalTensor<U>& WAIBJ = W.getAIBJ();

    Hguess["aibi"]  = FAB["ab"];
    Hguess["aibj"] -= WAIBJ["aibj"];
    Hguess["aiaj"] -= FIJ["ij"];

    auto& TDAevecs = put("TDAevecs", new vector<unique_vector<SpinorbitalTensor<U>>>(nirrep));
    auto& TDAevals = put("TDAevals", new vector<vector<U>>(nirrep));

    for (int R = 0;R < nirrep;R++)
    {
        const Representation& irr_R = group.getIrrep(R);

        int ntot = 0;
        for (int i = 0, count = 0;i < nirrep;i++)
        {
            const Representation& irr_i = group.getIrrep(i);
            for (int a = 0;a < nirrep;a++)
            {
                const Representation& irr_a = group.getIrrep(a);
                if (!(irr_a*irr_i*irr_R).isTotallySymmetric()) continue;
                ntot += vrt.nalpha[a]*occ.nalpha[i];
                ntot += vrt.nbeta [a]*occ.nbeta [i];
                assert(count++ < nirrep);
            }
            assert(i < nirrep-1 || count == nirrep);
        }

        #ifdef ELEMENTAL

        DistMatrix<U> H_elem(ntot, ntot);
        DistMatrix<U> C_elem(ntot, ntot);
        DistMatrix<U,VC,STAR> C_local;
        DistMatrix<U> E_elem;
        DistMatrix<U,STAR,STAR> E_local;

        int offbj = 0;
        for (int spin_bj : {1,0})
        {
            for (int j = 0;j < nirrep;j++)
            {
                const Representation& irr_j = group.getIrrep(j);
                for (int b = 0;b < nirrep;b++)
                {
                    const Representation& irr_b = group.getIrrep(b);
                    if (!(irr_b*irr_j*irr_R).isTotallySymmetric()) continue;

                    int nb = (spin_bj == 1 ? vrt.nalpha[b] : vrt.nbeta[b]);
                    int nj = (spin_bj == 1 ? occ.nalpha[j] : occ.nbeta[j]);
                    int nbj = nb*nj;

                    int offai = 0;
                    for (int spin_ai : {1,0})
                    {
                        for (int i = 0;i < nirrep;i++)
                        {
                            const Representation& irr_i = group.getIrrep(i);
                            for (int a = 0;a < nirrep;a++)
                            {
                                const Representation& irr_a = group.getIrrep(a);
                                if (!(irr_a*irr_i*irr_R).isTotallySymmetric()) continue;

                                int na = (spin_ai == 1 ? vrt.nalpha[a] : vrt.nbeta[a]);
                                int ni = (spin_ai == 1 ? occ.nalpha[i] : occ.nbeta[i]);
                                int nai = na*ni;

                                int cshift = H_elem.ColShift();
                                int rshift = H_elem.RowShift();
                                int cstride = H_elem.ColStride();
                                int rstride = H_elem.RowStride();

                                int ishift0 = offai%cstride;
                                int iloc0 = offai/cstride;
                                if (ishift0 > cshift) iloc0++;

                                int ishift1 = (offai+nai)%cstride;
                                int iloc1 = (offai+nai)/cstride;
                                if (ishift1 > cshift) iloc1++;

                                int jshift0 = offbj%rstride;
                                int jloc0 = offbj/rstride;
                                if (jshift0 > rshift) jloc0++;

                                int jshift1 = (offbj+nai)%rstride;
                                int jloc1 = (offbj+nai)/rstride;
                                if (jshift1 > rshift) jloc1++;

                                vector<tkv_pair<U>> pairs;

                                for (int iloc = iloc0;iloc < iloc1;iloc++)
                                {
                                    key aidx = (iloc-offai)%na;
                                    key iidx = (iloc-offai)/na;
                                    for (int jloc = jloc0;jloc < jloc1;jloc++)
                                    {
                                        key bidx = (jloc-offbj)%nb;
                                        key jidx = (jloc-offbj)/nb;

                                        assert(aidx >= 0 && aidx < na);
                                        assert(bidx >= 0 && bidx < nb);
                                        assert(iidx >= 0 && iidx < ni);
                                        assert(jidx >= 0 && jidx < nj);
                                        key k = ((iidx*nb+bidx)*nj+jidx)*na+aidx;
                                        pairs.emplace_back(k, 0);
                                    }
                                }

                                Hguess({spin_ai,spin_bj},{spin_bj,spin_ai})({a,j,b,i}).getRemoteData(pairs);

                                for (auto p : pairs)
                                {
                                    key k = p.k;
                                    int aidx = k%na;
                                    k /= na;
                                    int jidx = k%nj;
                                    k /= nj;
                                    int bidx = k%nb;
                                    k /= nb;
                                    int iidx = k;

                                    int iloc = aidx+iidx*na+offai;
                                    int jloc = bidx+jidx*nb+offbj;

                                    assert(aidx >= 0 && aidx < na);
                                    assert(bidx >= 0 && bidx < nb);
                                    assert(iidx >= 0 && iidx < ni);
                                    assert(jidx >= 0 && jidx < nj);
                                    H_elem.SetLocal(iloc, jloc, p.d);
                                }

                                offai += nai;
                            }
                        }
                    }
                    offbj += nbj;
                }
            }
        }

        HermitianEig(UPPER, H_elem, E_elem, C_elem);

        E_local = E_elem;
        C_local = C_elem;

        for (int root = 0;root < ntot;root++)
        {
            TDAevals[R].push_back(E_local.GetLocal(root, 0));
            TDAevecs[R].emplace_back("R", arena, occ.group, irr_R, vec(vrt, occ), vec(1,0), vec(0,1));
            SpinorbitalTensor<U>& evec = TDAevecs[R][root];

            vector<tkv_pair<U>> pairs;
            pairs.reserve(ntot);

            int offai = 0;
            for (int spin_ai : {1,0})
            {
                for (int i = 0;i < nirrep;i++)
                {
                    const Representation& irr_i = group.getIrrep(i);
                    for (int a = 0;a < nirrep;a++)
                    {
                        const Representation& irr_a = group.getIrrep(a);
                        if (!(irr_a*irr_i*irr_R).isTotallySymmetric()) continue;

                        int nai = (spin_ai == 1 ? vrt.nalpha[a] : vrt.nbeta[a])*
                                  (spin_ai == 1 ? occ.nalpha[i] : occ.nbeta[i]);

                        int cshift = C_local.ColShift();
                        int cstride = C_local.ColStride();

                        int shift0 = offai%cstride;
                        int loc0 = offai/cstride;
                        if (shift0 > cshift) loc0++;

                        int shift1 = (offai+nai)%cstride;
                        int loc1 = (offai+nai)/cstride;
                        if (shift1 > cshift) loc1++;

                        vector<tkv_pair<U>> pairs;

                        for (int loc = loc0;loc < loc1;loc++)
                        {
                            int gloc = loc*cstride+cshift-offai;
                            pairs.emplace_back(gloc, C_local.GetLocal(loc, root));
                        }

                        evec({spin_ai,0},{0,spin_ai})({a,i}).writeRemoteData(pairs);

                        offai += nai;
                    }
                }
            }
        }

        #else

        vector<U> data(ntot*ntot);

        int offbj = 0;
        for (int spin_bj : {1,0})
        {
            for (int j = 0;j < nirrep;j++)
            {
                const Representation& irr_j = group.getIrrep(j);
                for (int b = 0;b < nirrep;b++)
                {
                    const Representation& irr_b = group.getIrrep(b);
                    if (!(irr_b*irr_j*irr_R).isTotallySymmetric()) continue;

                    int nbj = (spin_bj == 1 ? vrt.nalpha[b] : vrt.nbeta[b])*
                              (spin_bj == 1 ? occ.nalpha[j] : occ.nbeta[j]);

                    int offai = 0;
                    for (int spin_ai : {1,0})
                    {
                        for (int i = 0;i < nirrep;i++)
                        {
                            const Representation& irr_i = group.getIrrep(i);
                            for (int a = 0;a < nirrep;a++)
                            {
                                const Representation& irr_a = group.getIrrep(a);
                                if (!(irr_a*irr_i*irr_R).isTotallySymmetric()) continue;

                                int nai = (spin_ai == 1 ? vrt.nalpha[a] : vrt.nbeta[a])*
                                          (spin_ai == 1 ? occ.nalpha[i] : occ.nbeta[i]);

                                CTFTensor<U>& this_tensor = Hguess({spin_ai,spin_bj},{spin_bj,spin_ai})({a,j,b,i});
                                CTFTensor<U> trans_tensor("trans_tensor", arena, 4, {vrt.nalpha[a],occ.nalpha[i],vrt.nbeta[b],occ.nbeta[j]}, {NS,NS,NS,NS}, true);
                                trans_tensor["ajbi"] = this_tensor["aibj"];
                                vector<U> tempdata;
                                trans_tensor.getAllData(tempdata);
                                assert(tempdata.size() == nai*nbj);
                                for (int bj = 0, aibj = 0;bj < nbj;bj++)
                                {
                                    for (int ai = 0;ai < nai;ai++, aibj++)
                                    {
                                        data[offai+ai+(offbj+bj)*ntot] = tempdata[aibj];
                                    }
                                }
                                offai += nai;
                            }
                        }
                    }
                    offbj += nbj;
                }
            }
        }

        TDAevals[R].resize(ntot);
        heev('V','U',ntot,data.data(),ntot,TDAevals[R].data());

        // cout << fixed << setprecision(5);
        // if (arena.rank == 0)
        // {
        //     cout << TDAevals[R] << endl;
        //     cout << "I'm rank 0. " << TDAevals[R][0] << endl;
        //     arena.Barrier();
        // }
        // else
        // {
        //     arena.Barrier();
        //     cout << TDAevals[R] << endl;
        //     cout << "I'm not rank 0. " << TDAevals[R][0] << endl;
        // }
        arena.Barrier();

        for (int root = 0;root < ntot;root++)
        {
            TDAevecs[R].emplace_back("R", arena, occ.group, irr_R, vec(vrt, occ), vec(1,0), vec(0,1));
            SpinorbitalTensor<U>& evec = TDAevecs[R][root];

            int offai = 0;
            for (int spin_ai = 1;spin_ai >= 0;spin_ai--)
            {
                for (int i = 0;i < nirrep;i++)
                {
                    const Representation& irr_i = group.getIrrep(i);
                    for (int a = 0;a < nirrep;a++)
                    {
                        const Representation& irr_a = group.getIrrep(a);
                        if (!(irr_a*irr_i*irr_R).isTotallySymmetric()) continue;

                        int nai = (spin_ai == 1 ? vrt.nalpha[a] : vrt.nbeta[a])*
                                  (spin_ai == 1 ? occ.nalpha[i] : occ.nbeta[i]);

                        vector<tkv_pair<U> > pairs(nai);
                        for (int ai = 0;ai < nai;ai++)
                        {
                            pairs[ai].k = ai;
                            pairs[ai].d = data[offai+ai+root*ntot];
                        }

                        if (arena.rank == 0)
                            evec({spin_ai,0},{0,spin_ai})({a,i}).writeRemoteData(pairs);
                        else
                            evec({spin_ai,0},{0,spin_ai})({a,i}).writeRemoteData();

                        offai += nai;
                    }
                }
            }
        }

        #endif

        cosort(TDAevals[R].begin() , TDAevals[R].end(),
               TDAevecs[R].pbegin(), TDAevecs[R].pend());
    }
}

INSTANTIATE_SPECIALIZATIONS(TDA);
REGISTER_TASK(TDA<double>, "tda");
