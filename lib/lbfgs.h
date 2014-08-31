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

#include <iostream>
#include <cmath>
#include <Eigen/Dense>

namespace npl {  

using std::function;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;

typedef function<int(const Vector& x, double& value, Vector& grad)> ComputeFunc;
typedef function<void(const Vector& x, double value, const Vector& grad)> 
    CallBackFunc;

enum StopReason
{
    ENDGRAD = 1,    // end due to gradient below threshold
    ENDVALUE = 2,   // end due to change in value below threshold
    ENDITERS = 3,   // end due to number iterations
    ENDFAIL = -1    // end tue to some error
};

using std::max;
using std::abs;

/**
 * @brief Implements the Quasi-Newton optimization method 
 * Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm by default (when
 * opt_zeta=1). When zeta=0 it is the Davidson-Fletcher-Powell (DFP) method.
 *
 */
class BFGSOpt
{
    
public:
    Vector state_x; 
    Matrix state_H; 
    
    double stop_G;
    double stop_X;
    double stop_F;
    int stop_Its;
    
    /**
     * @brief Maximum step size, step will be rescaled to this length if it 
     * exceeds it after other scaling is complete.
     */
    double opt_max_step;
    
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
     * @brief Weighting of 
     */
    double opt_zeta;


    /**
     * @brief Constructor of Gradient Optimizer
     *
     * @param start_x Initial state
     */
    BFGSOpt(const Vector& start_x);

    /**
     * @brief Returns the current state, use to get the final state after
     * optimize returns
     *
     * @return Vector with state variable
     */
    const Vector& getState();

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
    int optimize(ComputeValFunc valfunc, ComputeGradFunc gradfunc, 
                CallBackFunc callback = []{return 0;});

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
    int optimize(ComputeFunc update, ComputeValFunc valfunc, 
            ComputeGradFunc gradfunc, CallBackFunc callback = []{return 0;});

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
    int optimize(ComputeFunc update, CallBackFunc callback = []{return 0;});
};

}

