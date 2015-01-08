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

#include <functional>
#include <Eigen/Dense>

namespace npl
{

/** \addtogroup Optimizers Optimization Algorithms
 * @{
 */

using std::function;
using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * @brief Value and Gradient Computation Function
 */
typedef function<int(const VectorXd& x, double& v, VectorXd& g)> ValGradFunc;

/**
 * @brief Gradient Only Computation Function
 */
typedef function<int(const VectorXd& x, VectorXd& g)> GradFunc;

/**
 * @brief Value Only Computation Function
 */
typedef function<int(const VectorXd& x, double& v)> ValFunc;

/**
 * @brief Callback function
 */
typedef function<int(const VectorXd& x, double v, const VectorXd& g, size_t iter)> CallBackFunc;

/**
 * @brief Tests a gradient function using the value function.
 *
 * @param error     Error between analytical and numeric gradient
 * @param x         Position to test
 * @param stepsize  Step to take when testing gradient (will be taken in each
 *                  dimension successively)
 * @param tol       Tolerance, error below the tolerance will cause the
 *                  function to return 0, higher error will cause the function to return -1
 * @param valfunc   Function values compute
 * @param valgradfunc  Function gradient compute
 *
 * @return
 */
int testgrad(double& error, const VectorXd& x, double stepsize, double tol,
        const ValFunc& valfunc, const ValGradFunc& valgradfunc);

/**
 * @brief Tests a gradient function using the value function. 
 *
 * @param error     Error between analytical and numeric gradient
 * @param x         Position to test
 * @param stepsize  Step to take when testing gradient (will be taken in each
 *                  dimension successively)
 * @param tol       Tolerance, error below the tolerance will cause the
 *                  function to return 0, higher error will cause the function to return -1
 * @param valfunc   Function values compute
 * @param gradfunc  Function gradient compute
 *
 * @return 
 */
int testgrad(double& error, const VectorXd& x, double stepsize, double tol, 
        const ValFunc& valfunc, const GradFunc& gradfunc);

/**
 * @brief Implements generized rosenbrock gradient
 *
 * @param x Position vector
 * @param gradient Gradient at the position
 *
 * @return 
 */
int gRosenbrock_G(const VectorXd& x, VectorXd& gradient);

   /**
 * @brief Implements generized rosenbrock value
 *
 * @param x Position vector
 * @param v values
 *
 * @return 
 */
int gRosenbrock_V(const VectorXd& x, double& v);

/**
 * @brief Returns the number of times the Value and Gradient functions for the
 * Generalized Rosenbrock Function were called.
 *
 * @param vcalls Value calls
 * @param gcalls Gradient calls
 */
void gRosenbrock_callCounts(size_t& vcalls, size_t& gcalls);

/**
 * @brief Callback that does nothing.
 *
 * @param x
 * @param value
 * @param grad
 * @param iter
 *
 * @return 
 */
inline
int noopCallback(const VectorXd& x, double value, const VectorXd& grad, size_t iter)
{
    (void)(x);
    (void)(value);
    (void)(grad);
    (void)(iter);
    return 0;
}

enum StopReason
{
    ENDGRAD,    // end due to gradient below threshold
    ENDSTEP,    // end due to step size below threshold
    ENDVALUE,   // end due to change in value below threshold
	ENDABSVALUE, // end due to surpassing value threshold
    ENDITERS,   // end due to number iterations
    ENDFAIL     // end tue to some error
};

using std::max;
using std::abs;


class Optimizer
{
    
public:
    /**
     * @brief State variable, set to initialize
     */
    VectorXd state_x;
    
    /**
     * @brief Stop when graient magnitde falls below this value
     */
    double stop_G;

    /**
     * @brief Stop when step size drops below this value
     */
    double stop_X;

    /**
     * @brief Stop when change in function value drops below this value
     */
    double stop_F;

    /**
     * @brief Stop immediately when value goes above this
     */
    double stop_F_over;
    
	/**
     * @brief Stop immediately when value goes below this
     */
    double stop_F_under;

    /**
     * @brief Stop after this many iterations (does not include linesearch)
     */
    int stop_Its;
    
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
    Optimizer(size_t dim, const ValFunc& valfunc, const GradFunc& gradfunc, 
                const ValGradFunc& valgradfunc, 
                const CallBackFunc& callback = noopCallback);
    
    /**
     * @brief Constructor for optimizer function.
     *
     * @param dim       Dimensionality of state vector
     * @param valfunc   Function which computes the energy of the underlying
     *                  mathematical function
     * @param gradfunc  Function which computes the gradient of energy in the
     *                  underlying mathematical function
     * @param callback  Function which should be called at the end of each
     *                  iteration (for instance, to debug)
     */
    Optimizer(size_t dim, const ValFunc& valfunc, const GradFunc& gradfunc, 
                const CallBackFunc& callback = noopCallback);
    
    /**
     * @brief Perform optimization
     *
     * @return StopReason
     */
    virtual 
    StopReason optimize() { return ENDFAIL; };

    /**
     * @brief Provides a string that describes the stop reason
     *
     * @param r
     */
    static std::string explainStop(StopReason r);
protected:

    ValGradFunc m_compFG;
    GradFunc m_compG;
    ValFunc m_compF;
    CallBackFunc m_callback;
};

/**  @} */

} // npl

#endif //OPT_H
