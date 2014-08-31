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
 * @file linesearch.h Definitions for classes which of several line search
 * algorithms.
 *
 *****************************************************************************/

#ifndef LINESEARCH_H
#define LINESEARCH_H

#include "opt.h"

class Armijo
{
public:
    Armijo(const ComputeValFunc& valFunc) 
    {
        compVal = valFunc;
    }

    double opt_s;
    double opt_beta;
    double opt_sigma;
    size_t opt_maxIt;

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
            const Vector& direction)
    {
        double alpha = 0;
        Vector x = init_x;

        for(size_t m = 0; m < opt_maxIt; m++) {
            alpha = pow(opt_beta, m)*opt_s;
            x = init_x + alpha*direction;
            double val = compVal(x);
            if(init_val - val >= -opt_sigma*alpha*init_g.dot(direction))
                return alpha;
        }

        return alpha;
    };

private:
    ComputeValFunc compVal;
};

#endif 

