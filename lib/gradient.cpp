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
 * @file gradient.cpp Implemenation of the GradientOpt class which implements 
 * a gradient descent energy minimization (optimization) algorithm.
 * 
 *****************************************************************************/

#include "gradient.h"

namespace npl
{

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
GradientOpt::GradientOpt(size_t dim, const ValFunc& valfunc, 
            const GradFunc& gradfunc, const ValGradFunc& valgradfunc, 
            const CallBackFunc& callback) 
            : Optimizer(dim, valfunc, gradfunc, valgradfunc, callback)
{
    opt_maxstep = 1;
    opt_init_scale = 1;
    opt_rdec_scale = 0.999;
};

/**
 * @brief Optimize Based on a combined value and gradient function
 * Note that during line search, we don't always use the gradient,
 * so if there is additional overhead of the gradient, you can avoid it by
 * using optimize(ComputeValFunc, ComputeGradFunc). 
 *
 * @param update    Function which returns the function value at a position 
 *                  and places gradient in the grad argument
 * @param callback  Called at the end of each iteration (not during line
 *                  search though)
 *
 * @return          StopReason
 */
StopReason GradientOpt::optimize()
{
    VectorXd grad(state_x.rows());
    VectorXd prevx(state_x.rows());

    double cur = 0;
    double prev = 0;
    double stepsize = opt_init_scale;
    for(int iter  = 0 ; (stop_Its <= 0 || iter < stop_Its); iter++) {
        prev = cur;

        if(m_compFG(state_x, cur, grad) != 0)
            return ENDFAIL;

        if(grad.norm() <= stop_G)
            return ENDGRAD;
        if(fabs(prev-cur) <= stop_F)
            return ENDVALUE;

        // compute step, (in grad variable)
        grad = -grad*stepsize;

        double gn = grad.norm();
        if(gn <= stop_X)
            return ENDVALUE;

        if(gn > opt_maxstep)
            grad = opt_maxstep*grad/gn;

        // perform step
        state_x += grad;

        // rescale step
        stepsize *= opt_rdec_scale;

        // call back at end of iter
        m_callback(state_x, cur, grad, iter);
    }

    return ENDITERS;
}

}
