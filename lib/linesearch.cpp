/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * 	http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file linesearch.cpp Implemntation for linesearch algorithms to final minima
 * along a particular direction.
 *
 *****************************************************************************/

#include "linesearch.h"
#include <cmath>
#include <cassert>
#include <iostream>

using namespace std;
using Eigen::VectorXd;

namespace npl {

Wolfe::Wolfe(const ValFunc& valFunc, const GradFunc& gradFunc) 
{
    opt_s = 1;
    opt_beta = .5; // slowish
    opt_c1 = 1e-5;
    opt_c2 = 0.9;
    opt_maxIt = 20;

    compVal = valFunc;
    compGrad = gradFunc;
}
    
double Wolfe::search(double init_val, const VectorXd& init_x, const
        VectorXd& init_g, const VectorXd& direction)
{
//#ifdef DEBUG
//        fprintf(stderr, "Linesearch\n");
//#endif 
    VectorXd x = init_x;
    double gradDotDir = init_g.dot(direction); 
    double alpha = 0;
    double v = 0;
    VectorXd g(x.rows());

    if(opt_s <= 0)
        throw std::invalid_argument("opt_s must be > 0");

    for(size_t m = 0; m < opt_maxIt; m++) {
        alpha = pow(opt_beta, m)*opt_s;
        x = init_x + alpha*direction;
        compVal(x, v);

//#ifdef DEBUG
//        fprintf(stderr, "Alpha: %f, Init Val: %f, Val: %f, Sigma: %f, gd: %f",
//                alpha, init_val, v, opt_sigma, gradDotDir);
//#endif 
        if(init_val - v >= -opt_c1*alpha*gradDotDir) {
            compGrad(x, g);
            if(g.dot(direction) >= opt_c2*gradDotDir)
                return alpha;
        }
    }

    return 0;
};

Armijo::Armijo(const ValFunc& valFunc) 
{
    opt_s = 1;
    opt_beta = .5; // slowish
    opt_sigma = 1e-5;
    opt_maxIt = 20;

    compVal = valFunc;
}
    
double Armijo::search(double init_val, const VectorXd& init_x, const
        VectorXd& init_g, const VectorXd& direction)
{
//#ifdef DEBUG
//        fprintf(stderr, "Linesearch\n");
//#endif 
    VectorXd x = init_x;
    double gradDotDir = init_g.dot(direction); 
    double alpha = 0;
    double v = 0;

    if(opt_s <= 0)
        throw std::invalid_argument("opt_s must be > 0");

    for(size_t m = 0; m < opt_maxIt; m++) {
        alpha = pow(opt_beta, m)*opt_s;
        x = init_x + alpha*direction;
        compVal(x, v);

//#ifdef DEBUG
//        fprintf(stderr, "Alpha: %f, Init Val: %f, Val: %f, Sigma: %f, gd: %f",
//                alpha, init_val, v, opt_sigma, gradDotDir);
//#endif 
        if(init_val - v >= -opt_sigma*alpha*gradDotDir)
            return alpha;
    }

    return 0;
};

}
