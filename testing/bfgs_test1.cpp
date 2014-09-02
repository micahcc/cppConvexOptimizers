/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file bfgs_test1.cpp Tests the bfgs algorithm with various functions
 * 
 *****************************************************************************/

#include "opt.h"
#include "bfgs.h"
#include <iostream>
#include <iomanip>

using namespace std;
using namespace npl;

bool DEBUG = true;

int callback(const Vector& x, double v, const Vector& g, size_t iter)
{
    cout << "Iter: " << iter << " (" << v << ")\n";

    cout << "Position:\n[";
    for(size_t ii=0; ii<x.rows(); ii++)
        cout << std::setw(10) << setprecision(3) << x[ii];
    cout << "]" << endl;

    cout << "Gradient:\n[";
    for(size_t ii=0; ii<x.rows(); ii++)
        cout << std::setw(10) << setprecision(3) << g[ii];
    cout << "]" << endl << endl;

    return 0;
}

int generalized_rosenbrock_test(size_t n)
{
    const double tol = 0.0001;
    double error = 0;
    Vector x = Vector::Random(n);
    x.normalize();
    
    if(testgrad(error, x, 0.00000001, tol, gRosenbrock_V, gRosenbrock_G) != 0) {
        cerr << "Error "<<error<<" larger than tolerance ("<<tol<<")"<<endl;
        return -1;
    }

    BFGSOpt optimizer(n , gRosenbrock_V, gRosenbrock_G, callback);
    optimizer.state_x = x;
    optimizer.stop_Its = 10000;
    optimizer.stop_X = 0;
    optimizer.stop_G = 0.0000000001;
    
    StopReason stopr = optimizer.optimize();
    cerr << Optimizer::explainStop(stopr) << endl;

    return 0;
}

int main()
{
    for(size_t ii=39; ii < 40; ii++) {
        cerr << "N=" << ii << endl;
        int iters = generalized_rosenbrock_test(ii);
        if(iters < 0)
            return -1;
        cerr << "Iters=" << iters << endl << endl;

        size_t vcalls = 0, gcalls = 0;
        gRosenbrock_callCounts(vcalls, gcalls);
        cerr << "Value Calls: " << vcalls << endl;
        cerr << "Grad  Calls: " << gcalls << endl;
        cerr << "Total Calls: " << (vcalls+gcalls) << endl;
    }
}

