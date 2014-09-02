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
 * @file lbfgs.cpp Declaration of the LBFGSOpt class which implements 
 * a LBFGS optimization (energy minimization) algorithm.
 * 
 *****************************************************************************/

#ifndef LBFGS_H
#define LBFGS_H

#include <list>
#include <tuple>
#include <iostream>
#include <cmath>
#include <Eigen/Dense>

#include "opt.h"
#include "linesearch.h"

namespace npl {  

class LBFGSOpt : virtual public Optimizer
{
public:
    

    LBFGSOpt(size_t dim, const ValFunc& valfunc, 
            const GradFunc& gradfunc, 
            const CallBackFunc& callback = noopCallback);

    LBFGSOpt(size_t dim, const ValFunc& valfunc, 
            const GradFunc& gradfunc, 
            const ValGradFunc& gradAndValFunc, 
            const CallBackFunc& callback = noopCallback);

    /**
     * @brief Armijo line search class, note that it has several options that
     * may need to be set
     */
//    Wolfe m_lsearch;
    Armijo m_lsearch;

    /**
     * @brief Perform LBFGS optimization
     *
     * @return 
     */
    StopReason optimize();

    /**
     * @brief Number of updates to store for the purposes of estimating the 
     * hessian matrix
     */
    int opt_histsize;
    
//    /**
//     * @brief During linesearch, beta_1 determines the minimum drop in function
//     * value needed to accept an alpha (stepping scale)
//     */
//    double opt_beta1;
//    
//    /**
//     * @brief During linesearch, beta_2 determines the necesary drop in 
//     * gradient.direction to accept an alpha (stepping scale)
//     */
//    double opt_beta2;
//
    /**
     * @brief Default (initial) value for inverse hessian matrix
     */
    Vector opt_H0inv;

private:
    /**
     * @brief Stores the approximate value of the inverse hessian.
     * 
     * Elements are rho_k,qk,pk
     */
    std::list<std::tuple<double,Vector,Vector>> m_hist;

    /**
     * @brief Function for computing the hessian recursively
     *
     * @param gamma Scale of initial (H0)
     * @param d Direction from right multiplication so far
     * @param it Position in history list
     *
     * @return Direction (d) after right multiplying d by H_k, the hessian
     * estimate for position it, 
     */
    Vector hessFunc(double gamma, const Vector& d, 
        std::list<std::tuple<double,Vector,Vector>>::const_iterator it);

    Vector hessFuncTwoLoop(double gamma, const Vector& g);
};

}

#endif // LBFGS_H
