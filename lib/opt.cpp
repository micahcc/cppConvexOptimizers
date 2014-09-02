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
 * @file opt.cpp Contains implementation for base optimizer
 * 
 *****************************************************************************/

#include "opt.h"

#include <iostream>
#include <iomanip>

namespace npl {

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
Optimizer::Optimizer(size_t dim, const ValFunc& valfunc, const GradFunc& gradfunc, 
        const ValGradFunc& valgradfunc, const CallBackFunc& callback) 
        : state_x(dim)
{
    stop_G = 0.00001;
    stop_X = 0;
    stop_F = 0;
    stop_Its = -1;

    m_compF = valfunc;
    m_compG = gradfunc;
    m_compFG = valgradfunc;

    m_callback = callback;
};
    
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
Optimizer::Optimizer(size_t dim, const ValFunc& valfunc, const GradFunc&
            gradfunc, const CallBackFunc& callback) 
            : state_x(dim)
{
    stop_G = 0.00001;
    stop_X = 0;
    stop_F = 0;
    stop_Its = -1;

    m_compF = valfunc;
    m_compG = gradfunc;
    m_compFG = [&](const Vector& x, double& value, Vector& grad) -> int
    {
        return !(valfunc(x, value)==0 && gradfunc(x, grad)==0);
    };

    m_callback = callback;
};

std::string Optimizer::explainStop(StopReason r)
{
    switch(r) {
        case ENDGRAD:
            return "Optimizer stopped due to gradient below threshold.";
        case ENDSTEP: 
            return "Optimizer stopped due to step size below threshold.";
        case ENDVALUE:
            return "Optimizer stopped due to change in value below threshold.";
        case ENDITERS: 
            return "Optimizer stopped due to number iterations.";
        case ENDFAIL:
            return "Optimizer due to failure of callback functions.";
    }
    return "Unknown stop condition!";
}


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
int testgrad(double& error, const Vector& x, double stepsize, double tol, 
        const ValFunc& valfunc, const GradFunc& gradfunc)
{
//#ifndef NDEBUG
    size_t wid = 10;
    std::cerr << "Testing Gradient" << std::endl;
    std::cerr << std::setw(wid) << "Dim" << std::setw(wid) << "Analytic" <<
        std::setw(wid) << "Numeric" << std::endl;
//#endif //NDEBUG
    Vector g(x.rows());
    if(gradfunc(x, g) != 0) 
        return -1;

    double v = 0;
    double center = 0;
    if(valfunc(x, center) != 0)
        return -1;

    Vector step = Vector::Zero(x.rows());
    Vector gbrute(x.rows());
    for(size_t dd=0; dd<x.rows(); dd++) {
        step[dd] = stepsize;
        if(valfunc(x+step, v) != 0)
            return -1;
        step[dd] = 0;

        gbrute[dd] = (v-center)/stepsize;

//#ifndef NDEBUG
    std::cerr << std::setw(wid) << dd << std::setw(wid) << g[dd] <<
        std::setw(wid) << gbrute[dd] << std::endl;
//#endif //NDEBUG
    }

    error = (gbrute - g).norm();
    if(error > tol)
        return -2;

    return 0;
}

/**
 * @brief Implements generized rosenbrock value
 *
 * @param x Position vector
 * @param v values
 *
 * @return 
 */
int gRosenbrock_V(const Vector& x, double& v)
{
    v = 0;
    for(size_t ii=0; ii<x.rows()-1; ii++)
        v += pow(x[ii]-1,2)+100*pow(x[ii+1]-x[ii]*x[ii], 2);

    // i=N-1
    size_t ii=x.rows()-1;
    v += pow(x[ii]-1,2)+100*pow(-x[ii]*x[ii], 2);
    return 0;
}

/**
 * @brief Implements generized rosenbrock gradient
 *
 * @param x Position vector
 * @param gradient Gradient at the position
 *
 * @return 
 */
int gRosenbrock_G(const Vector& x, Vector& gradient)
{
//    gradient.resize(x.rows());
    
    for(size_t ii=1; ii<x.rows()-1; ii++)
        gradient[ii] = 2*(x[ii]-1)-400*(x[ii+1]-x[ii]*x[ii])*x[ii]+
                    200*(x[ii]-x[ii-1]*x[ii-1]);

    // boundaries
    size_t ii=0;
    gradient[ii] = 2*(x[ii]-1)-400*(x[ii+1]-x[ii]*x[ii])*x[ii];
    
    ii=x.rows()-1;
    gradient[ii] = 2*(x[ii]-1)-400*(-x[ii]*x[ii])*x[ii]+
                200*(x[ii]-x[ii-1]*x[ii-1]);
    
    return 0;
}

}
