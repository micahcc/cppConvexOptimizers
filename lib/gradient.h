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
 * @file gradient.h Definition for the GradientOpt class which implements 
 * a gradient descent energy minimization (optimization) algorithm.
 * 
 *****************************************************************************/

#ifndef GRADIENT_H
#define GRADIENT_H

#include <iostream>
#include <cmath>
#include <Eigen/Dense>

#include "opt.h"

namespace npl {  

/** \addtogroup Optimizers
 * @{
 */

class GradientOpt : virtual public Optimizer
{
public:
    /**
     * @brief Maximum step size, step will be rescaled to this length if it 
     * exceeds it after other scaling is complete.
     */
    double opt_maxstep;
    
    /**
     * @brief Initial scale to use during optimization, actual scale may differ
     * due to other options
     */
    double opt_init_scale;

    /**
     * @brief Multiply scale by this value after each iteration ( 0 < v < 1 ).
     * Values <= 0 will be considered unused. 
     */
    double opt_rdec_scale; 


    /**
     * @brief Constructor for optimizer function.
     *
     * @param dim       Dimensionality of state vector
     * @param valfunc   Function which computes the energy of the underlying
     *                  mathematical function
     * @param gradfunc  Function which computes the gradient of energy in the
     *                  underlying mathematical function
     * @param valgradfunc 
     *                  Function which computes the both the energy and
     *                  gradient in the underlying mathematical function
     * @param callback  Function which should be called at the end of each
     *                  iteration (for instance, to debug)
     */
    GradientOpt(size_t dim, const ValFunc& valfunc, 
                const GradFunc& gradfunc, 
                const ValGradFunc& valgradfunc, 
                const CallBackFunc& callback = noopCallback);
    
public:
    StopReason optimize();
};

/** @} */

}

#endif // GRADIENT_H
