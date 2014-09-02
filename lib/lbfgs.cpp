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
 * @file lbfgs.cpp Implemenation of the LBFGSOpt class which implements 
 * a LBFGS optimization (energy minimization) algorithm.
 * 
 *****************************************************************************/

#include "lbfgs.h"

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
LBFGSOpt::LBFGSOpt(size_t dim, const ValFunc& valfunc, 
        const GradFunc& gradfunc, const CallBackFunc& callback) 
        : Optimizer(dim, valfunc, gradfunc, callback), m_lsearch(valfunc)
{
    m_hist.clear();
    opt_H0inv = Vector::Ones(dim);
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
LBFGSOpt::LBFGSOpt(size_t dim, const ValFunc& valfunc, const GradFunc& gradfunc, 
        const ValGradFunc& gradAndValFunc, const CallBackFunc& callback) 
        : Optimizer(dim, valfunc, gradfunc, gradAndValFunc, callback),
        m_lsearch(valfunc)
{
    m_hist.clear();
    opt_H0inv = Vector::Ones(dim);
};


/**
 * @brief Function for computing the hessian recursively
 *
 * @param gamma Scale of initial (H0)
 * @param d Direction from right multiplication so far
 * @param it Position in history list
 *
 * @return Direction (d) after right multiplying d by H_k, the hessian
 * estimate for position it, 
 */
Vector LBFGSOpt::hessFunc(double gamma, const Vector& d, 
        std::list<std::tuple<double,Vector,Vector>>::const_iterator it)
{
    double rho = std::get<0>(*it);
    const Vector& q = std::get<1>(*it); //y
    const Vector& p = std::get<2>(*it); //s 
    Vector tmp = d - rho*q*p.dot(d);

    auto it2 = it;
    it2++;
    if(it2 == m_hist.cend())
        tmp = tmp.cwiseProduct(opt_H0inv)*gamma;
    else
        tmp = hessFunc(gamma, tmp, it2);
    
    return tmp - p*rho*q.dot(tmp) + rho*p*p.dot(d);
}

/**
 * @brief Optimize Based on a value function and gradient function
 * separately. When both gradient and value are needed it will call update,
 * when it needs just the gradient it will call gradFunc, and when it just 
 * needs the value it will cal valFunc. This is always the most efficient,
 * assuming there is additional cost of computing the gradient or value, but 
 * its obviously more complicated. 
 *
 * Paper: On the limited memory BFGS method for large scale optimization
 * By: Liu, Dong C., Nocedal, Jorge
 *
 * @param   x_init Starting value for optimization
 *
 * @return          StopReason
 */
StopReason LBFGSOpt::optimize()
{
    double gradstop = this->stop_G >= 0 ? this->stop_G*this->stop_G : -1;
    double stepstop = this->stop_X >= 0 ? this->stop_X*this->stop_X : -1;
    double valstop = this->stop_F >= 0 ? this->stop_F : -1;

    Vector gk(state_x.rows()); // gradient
    double f_xk; // value at current position
    double f_xkm1; // value at previous position

    Vector pk, qk, dk, vk;
    Vector H0 = Vector::Ones(state_x.rows());
    double gamma = 1;

    //D(k+1) += p(k)p(k)'   - D(k)q(k)q(k)'D(k) + Z(k)T(k)v(k)v(k)'
    //          ----------    ----------------- 
    //         (p(k)'q(k))      q(k)'D(k)q(k)
    m_compFG(state_x, f_xk, gk);
    dk = -gk;
    for(int iter = 0; stop_Its <= 0 || iter < stop_Its; iter++) {
        // compute step size
        double alpha = m_lsearch.search(f_xk, state_x, gk, dk);
        pk = alpha*dk;
        
        if(alpha == 0 || pk.squaredNorm() < stepstop)
            return ENDSTEP;

        // step
        state_x += pk;
        
        // update gradient, value
        qk = -gk;
        f_xkm1 = f_xk;
        m_compFG(state_x, f_xk, gk);
        qk += gk;

        if(gk.squaredNorm() < gradstop)
            return ENDGRAD;
        
        if(abs(f_xk - f_xkm1) < valstop)
            return ENDVALUE;


        // update history
        m_hist.push_front(std::make_tuple(1./qk.dot(pk), qk, pk));
        if(m_hist.size() > opt_histsize)
            m_hist.pop_back();

        /*
         * update direction
         * qk - change in gradient (yk in original paper)
         * pk - change in x (sk in original paper)
         */
        gamma = qk.dot(pk)/qk.squaredNorm();
        dk = hessFunc(gamma, gk, m_hist.cbegin());
        m_callback(state_x, f_xk, gk, iter);
    }
                   
    return ENDFAIL;
}

LBFGSOpt::Armijo::Armijo(const ValFunc& valFunc) 
{
    opt_s = 1;
    opt_beta = .5; // slowish
    opt_sigma = 1e-5;
    opt_maxIt = 20;

    compVal = valFunc;
}
    
double LBFGSOpt::Armijo::search(double init_val, const Vector& init_x, const
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
