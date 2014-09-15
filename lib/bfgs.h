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
#include "linesearch.h"

namespace npl {  

/** \addtogroup Optimizers Optimization Algorithms
 * @{
 */

class BFGSOpt : virtual public Optimizer
{
private:
    /**
     * @brief Stores the approximate value of the inverse hessian
     */
    MatrixXd state_Hinv;

public:
    
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

    StopReason optimize();
};

/** @} */

}

#endif // BFGS_H
