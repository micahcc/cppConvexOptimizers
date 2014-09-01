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
 * @file bfgs.cpp Declaration of the BFGSOpt class which implements 
 * a BFGS optimization (energy minimization) algorithm.
 * 
 *****************************************************************************/

#ifndef BFGS_H
#define BFGS_H

#include <iostream>
#include <cmath>
#include <Eigen/Dense>

#include "opt.h"

namespace npl {  

class BFGSOpt : virtual public Optimizer
{
private:
    /**
     * @brief Stores the approximate value of the inverse hessian
     */
    Matrix state_Hinv;

public:
    
    /**
     * @brief Implementation of Armijo approximate line search algorithm
     */
    class Armijo
    {
    public:
        Armijo(const ValFunc& valFunc);

        /**
         * @brief Maximum step, if this is <= 0, then a quadratic fit will be used 
         * to estimate a guess.
         */
        double opt_s;

        /**
         * @brief Power function base, values closer to 0 will decrease step size
         * faster than ones close to 1.
         */
        double opt_beta;

        /**
         * @brief Theshold for stopping
         */
        double opt_sigma;

        /**
         * @brief Maximum number of iterations
         */
        int opt_maxIt; 

        /**
         * @brief Performs a line search to find the alpha (step size) that
         * satifies the armijo rule.
         *
         * @param init_val Initial energy/function value
         * @param init_x Initial state 
         * @param init_g Initial gradient
         * @param direction Direction to search
         *
         * @return 
         */
        double search(double init_val, const Vector& init_x, const Vector& init_g,
                const Vector& direction);

    private:
        ValFunc compVal;
    };


    BFGSOpt(size_t dim, const ValFunc& valfunc, 
            const GradFunc& gradfunc, 
            const CallBackFunc& callback = noopCallback);

    BFGSOpt(size_t dim, const ValFunc& valfunc, 
            const GradFunc& gradfunc, 
            const ValGradFunc& gradAndValFunc, 
            const CallBackFunc& callback = noopCallback);

    /**
     * @brief Armijo line search class, note that it has several options that
     * may need to be set
     */
    Armijo m_lsearch;

    int optimize();
};

}

#endif // BFGS_H
