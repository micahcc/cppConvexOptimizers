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

/**
 * @brief Constructor for optimizer function.
 *
 * @param dim       Dimension of state variable
 * @param valfunc   Function which computes the energy of the underlying
 *                  mathematical function
 * @param gradfunc  Function which computes the gradient of energy in the
 *                  underlying mathematical function
 * @param callback  Function which should be called at the end of each
 *                  iteration (for instance, to debug)
 */
BFGSOpt::BFGSOpt(size_t dim, const ValFunc& valfunc, 
        const GradFunc& gradfunc, const CallBackFunc& callback) 
        : Optimizer(dim, valfunc, gradfunc, callback), m_lsearch(valfunc)
{
    state_Hinv = Matrix::Identity(dim, dim);
};

/**
 * @brief Constructor for optimizer function.
 *
 * @param dim       Dimension of state variable
 * @param valfunc   Function which computes the energy of the underlying
 *                  mathematical function
 * @param gradfunc  Function which computes the gradient of energy in the
 *                  underlying mathematical function
 * @param gradAndValFunc 
 *                  Function which computes the energy and gradient in the
 *                  underlying mathematical function
 * @param callback  Function which should be called at the end of each
 *                  iteration (for instance, to debug)
 */
BFGSOpt::BFGSOpt(size_t dim, const ValFunc& valfunc, const GradFunc& gradfunc, 
        const ValGradFunc& gradAndValFunc, const CallBackFunc& callback) 
        : Optimizer(dim, valfunc, gradfunc, gradAndValFunc, callback),
        m_lsearch(valfunc)
{
    state_Hinv = Matrix::Identity(dim, dim);
};


/**
 * @brief Optimize Based on a value function and gradient function
 * separately. When both gradient and value are needed it will call update,
 * when it needs just the gradient it will call gradFunc, and when it just 
 * needs the value it will cal valFunc. This is always the most efficient,
 * assuming there is additional cost of computing the gradient or value, but 
 * its obviously more complicated. 
 *
 * @param   x_init Starting value for optimization
 *
 * @return          StopReason
 */
int BFGSOpt::optimize()
{
    const double ZETA = 1;
    Matrix& Dk = state_Hinv;
    Vector gk(state_x.rows()); // gradient
    double f_xk; // value at current position

    Vector xkprev; 
    double tauk = 0;
    Vector pk, qk, dk, vk;

    //D(k+1) += p(k)p(k)'   - D(k)q(k)q(k)'D(k) + Z(k)T(k)v(k)v(k)'
    //          ----------    ----------------- 
    //         (p(k)'q(k))      q(k)'D(k)q(k)
    m_compFG(state_x, f_xk, gk);
    for(int iter = 0; iter < stop_Its; iter++) {

        // optain direction
        dk = -Dk*gk;

        // compute step size
        double alpha = m_lsearch.search(f_xk, state_x, gk, dk);
#ifdef DEBUG
        fprintf(stderr, "New Alpha: %f\n", alpha);
#endif 
        pk = alpha*dk;

        // step
        xkprev = state_x;
        state_x += pk;
        
        // update gradient, value
        qk = -gk;
        m_compFG(state_x, f_xk, gk);
        qk += gk;
#ifdef DEBUG
        fprintf(stderr, "Value: %f\n", f_xk);
#endif 

        // update information inverse hessian
        tauk = qk.dot(Dk*qk);
        if(tauk < 1E-20) 
            vk.setZero();
        else
            vk = pk/pk.dot(qk) - Dk*qk/tauk;

        Dk += pk*pk.transpose()/pk.dot(qk) - Dk*qk*qk.transpose()*Dk/
                    (qk.dot(Dk*qk)) + ZETA*tauk*vk*vk.transpose();
    }
                   
    return ENDFAIL;
}

BFGSOpt::Armijo::Armijo(const ValFunc& valFunc) 
{
    opt_s = 1;
    opt_beta = .5; // slowish
    opt_sigma = 1e-5;
    opt_maxIt = 10;

    compVal = valFunc;
}
    
double BFGSOpt::Armijo::search(double init_val, const Vector& init_x, const
        Vector& init_g, const Vector& direction)
{
#ifdef DEBUG
        fprintf(stderr, "Linesearch\n");
#endif 
    Vector x = init_x;
    double gradDotDir = init_g.dot(direction); 
    double alpha = 0;
    double v = 0;

    if(opt_s <= 0)
        throw std::invalid_argument("opt_s must be > 0");

    for(size_t m = 0; m < opt_maxIt; m++) {
        alpha = pow(opt_beta, m)*opt_s;
        x = init_x + alpha*direction;
        compVal(x, v);

#ifdef DEBUG
        fprintf(stderr, "Alpha: %f, Init Val: %f, Val: %f, Sigma: %f, gd: %f",
                alpha, init_val, v, opt_sigma, gradDotDir);
#endif 
        if(init_val - v >= -opt_sigma*alpha*gradDotDir)
            return alpha;
    }

    return 0;
};


}
