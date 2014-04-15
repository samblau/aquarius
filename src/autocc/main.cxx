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

#include <cstdlib>

#include "autocc.hpp"

using namespace std;
using namespace aquarius;
using namespace aquarius::autocc;

int main(int argc, char **argv)
{
	vector<string> s;
    s.push_back("P(Av0|Av1) P(Av3|Av4) A(bo0 Av1 av2,bv0 Av4 ao0) B(Av0 bv0,Av3 bo0)");
    s.push_back("P(Av0|Av1) P(Av3|Av4) A(Bo0 Av1 av2,Bv0 Av4 ao0) B(Av0 Bv0,Av3 Bo0)");
    s.push_back("- P(Av3|Av4) A(bo0 Av1 Av0,Bv0 Av4 ao0) B(av2 Bv0,Av3 bo0)");
	Diagram d(Diagram::UHF, s);
	d.convert(Diagram::SKELETON);
	cout << d << endl;

    return 0;
}

