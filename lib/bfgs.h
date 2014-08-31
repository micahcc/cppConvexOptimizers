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

#ifndef BFGS_H
#define BFGS_H

#include <iostream>
#include <cmath>
#include <Eigen/Dense>

#include "opt.h"

namespace npl {  

class BFGSOpt : public Optimizer
{
private:
    /**
     * @brief Stores the approximate value of the hessian
     */
    Matrix state_Hinv;
    
public:
    /**
     * @brief Constructor of Gradient Optimizer
     *
     * @param start_x Initial state
     */
    BFGSOpt(const Vector& start_x);

    /**
     * @brief Optimize Based on a value function and gradient function
     * separately. Since we do a line search, we don't always use the gradient,
     * so if there is additional overhead of the gradient, you can avoid it by
     * using this function.
     *
     * @param valfunc   Function which returns the function value at a position
     * @param gradfunc  Function which returns the gradient of the function at
     *                  a position
     * @param callback  Called at the end of each iteration (not during line
     *                  search though)
     *
     * @return          StopReason
     */
    virtual 
    int optimize(const ComputeValFunc& valfunc, 
                const ComputeGradFunc& gradfunc, 
                const CallBackFunc& callback = noopCallback);

    /**
     * @brief Optimize Based on a value function and gradient function
     * separately. When both gradient and value are needed it will call update,
     * when it needs just the gradient it will call gradFunc, and when it just 
     * needs the value it will cal valFunc. This is always the most efficient,
     * assuming there is additional cost of computing the gradient or value, but 
     * its obviously more complicated. 
     *
     * @param update    Function which returns the function value at a
     *                  position, and the gradient at the same position
     *                  (through the grad argument)
     * @param valfunc   Function which returns the function value at a position
     * @param gradfunc  Function which returns the gradient of the function at
     *                  a position
     * @param callback  Called at the end of each iteration (not during line
     *                  search though)
     *
     * @return          StopReason
     */
    virtual
    int optimize(const ComputeFunc& update, 
                const ComputeValFunc& valfunc, 
                const ComputeGradFunc& gradfunc, 
                const CallBackFunc& callback = noopCallback);

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
    virtual
    int optimize(const ComputeFunc& update, 
                 const CallBackFunc& callback = noopCallback);
};

}

#endif // BFGS_H
