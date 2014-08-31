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
 * @file opt.h Contains base class for all optimizers, defines Function types,
 * and StopReason
 * 
 *****************************************************************************/

#ifndef OPT_H
#define OPT_H

namespace npl
{

using std::function;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;

typedef function<int(const Vector& x, double& value, Vector& grad)> ComputeFunc;
typedef function<int(const Vector& x, Vector& grad)> ComputeGradFunc;
typedef function<int(const Vector& x, double& value)> ComputeValFunc;
typedef function<int(const Vector& x, double value, const Vector& grad)> 
    CallBackFunc;

int noopCallback(const Vector& x, double value, const Vector& grad)
{
    (void)(x);
    (void)(value);
    (void)(grad);
    return 0;
}

enum StopReason
{
    ENDGRAD,    // end due to gradient below threshold
    ENDVALUE,   // end due to change in value below threshold
    ENDITERS,   // end due to number iterations
    ENDFAIL     // end tue to some error
};

using std::max;
using std::abs;


class Optimizer
{
    
public:
    Vector state_x; 
    
    double stop_G;
    double stop_X;
    double stop_F;
    int stop_Its;
    
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
     * @brief Constructor
     *
     * @param start_x Initial state
     */
    Optimizer(const Vector& start_x)
    {
        stop_G = 0.00001;
        stop_X = 0;
        stop_F = 0;
        stop_Its = -1;

        opt_init_scale = 1;
        opt_rdec_scale = .99;

        // don't allow scales > 1, that would lead to infinitely 
        // increasing of scale

        if(opt_rdec_scale > 1 || opt_rdec_scale < 0) 
            throw std::invalid_argument("0 < opt_rdec_scale <= 1");

        state_x = start_x;
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
    int optimize(const ComputeFunc& update, const ComputeValFunc& valfunc, 
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

} // npl

#endif //OPT_H
