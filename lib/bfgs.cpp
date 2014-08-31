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

#include "bfgs.h"

namespace npl {

BFGSOpt::BFGSOpt(const Vector& start_x) : Optimizer(start_x) 
{
    state_Hinv = Matrix::Identity(start_x.rows(), start_x.rows());
};

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
int BFGSOpt::optimize(const ComputeValFunc& valfunc, 
        const ComputeGradFunc& gradfunc, 
        const CallBackFunc& callback)
{
    return ENDFAIL;
};

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
int BFGSOpt::optimize(const ComputeFunc& update, 
        const ComputeValFunc& valfunc, 
        const ComputeGradFunc& gradfunc, 
        const CallBackFunc& callback)
{
    Matrix& Dk = state_Hinv;
    Vector xkp1 = state_x;
    Vector xk(sate_x.rows());
    
    Vector gkp1(sate_x.rows());
    Vector gk(sate_x.rows());

    Vector pk(sate_x.rows());
    Vector qk(sate_x.rows());
    Vector vk(sate_x.rows());

    double tauk = 0;

    //D(k+1) += p(k)p(k)'   - D(k)q(k)q(k)'D(k) + Z(k)T(k)v(k)v(k)'
    //          ----------    ----------------- 
    //         (p(k)'q(k))      q(k)'D(k)q(k)
    for() {
        // step, using line search
        xk = xkp1;
        gk = gkp1;
        xkp1 = linesearch();
        if(gradfunc(xkp1, gk) != 0)
            return ENDFAIL;

        // update information 
        pk = xkp1 - xk;
        qk = gkp1 - gk;
        tauk = qk.dot(Dk*qk);

        if(tauk < 1E-20) 
            vk.setZero();
        else
            vk = pk/pk.dot(qk) - Dk*qk/tauk;

        Dk += pk*pk.transpose()/pk.dot(qk) - Dk*qk*qk.transpose()*Dk/
                    (qk.dot(Dk*qk)) + opt_zeta*tauk*vk*vk.transpose();
    }
                   
    return ENDFAIL;
}
    
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
int BFGSOpt::optimize(const ComputeFunc& update, const CallBackFunc& callback)
{
    return ENDFAIL;   
};

}
