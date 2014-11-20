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

#ifdef DEBUG
#include <iterator>
#include <iostream>
#include <iomanip>
using std::cerr;
using std::endl;
using std::setw;
using std::setprecision;
#endif //DEBUG

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
        : Optimizer(dim, valfunc, gradfunc, callback), 
        m_lsearch(valfunc)
{
    m_hist.clear();
    opt_H0inv = VectorXd::Ones(dim);
    opt_histsize = 6;
	
	opt_ls_s = 1;
	opt_ls_beta = 0.5;
	opt_ls_sigma = 1e-5;
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
    opt_H0inv = VectorXd::Ones(dim);
    opt_histsize = 6;

	opt_ls_s = 1;
	opt_ls_beta = 0.5;
	opt_ls_sigma = 1e-5;
};

/**
 * @brief Function for computing the hessian recursively
 * Based on the algorithm from Numerical Optimization (Nocedal)
 *
 * @param gamma Scale of initial (H0)
 * @param g Direction from right multiplication so far
 * @param it Position in history list
 *
 * @return Direction (d) after right multiplying d by H_k, the hessian
 * estimate for position it, 
 */
VectorXd LBFGSOpt::hessFuncTwoLoop(double gamma, const VectorXd& g)
{
    VectorXd q = g;
    VectorXd alpha(m_hist.size());

    // iterate backward in time (forward in list)
    int ii = 0;
    for(auto it = m_hist.cbegin(); it != m_hist.cend(); ++it, ++ii) {
        double rho = std::get<0>(*it);
        const VectorXd& y = std::get<1>(*it); // or q 
        const VectorXd& s = std::get<2>(*it); // or p
        alpha[ii] = rho*s.dot(q);
        q -= alpha[ii]*y;
    }
    VectorXd r = opt_H0inv.cwiseProduct(q)*gamma;
    // oldest first
    ii = m_hist.size()-1;
    for(auto it = m_hist.crbegin(); it != m_hist.crend(); ++it, --ii) {
        double rho = std::get<0>(*it);
        const VectorXd& y = std::get<1>(*it); // or q 
        const VectorXd& s = std::get<2>(*it); // or p
        double beta = rho*y.dot(r);
        r += s*(alpha[ii]-beta);
    }

    return r;
}

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
 * @return          StopReason
 */
StopReason LBFGSOpt::optimize()
{
    double gradstop = this->stop_G >= 0 ? this->stop_G : 0;
    double stepstop = this->stop_X >= 0 ? this->stop_X : 0;
    double valstop = this->stop_F >= 0 ? this->stop_F : -1;

    // update linesearch with minimum step, an options
    m_lsearch.opt_s = opt_ls_s;
    m_lsearch.opt_minstep = stepstop;
    m_lsearch.opt_beta = opt_ls_beta;
    m_lsearch.opt_sigma = opt_ls_sigma;

    VectorXd gk(state_x.rows()); // gradient
    double f_xk; // value at current position
    double f_xkm1; // value at previous position

    VectorXd pk, qk, dk, vk;
    VectorXd H0 = VectorXd::Ones(state_x.rows());
    double gamma = 1;

    //D(k+1) += p(k)p(k)'   - D(k)q(k)q(k)'D(k) + Z(k)T(k)v(k)v(k)'
    //          ----------    ----------------- 
    //         (p(k)'q(k))      q(k)'D(k)q(k)
    m_compFG(state_x, f_xk, gk);
    dk = -gk;
    for(int iter = 0; stop_Its <= 0 || iter < stop_Its; iter++) {
		// reset history if grad . gk > 0 (ie they go the same direction)
		if(gk.dot(dk) >= 0) {
			dk = -gk;
			m_hist.clear();
#ifdef DEBUG
			cerr << "Clearing LBFGS History!" << endl;
#endif //DEBUG
		}

        // compute step size
        double alpha = m_lsearch.search(f_xk, state_x, gk, dk);
        pk = alpha*dk;
        
        if(alpha == 0 || pk.squaredNorm() < stepstop*stepstop) {
            return ENDSTEP;
        }

        // step
        state_x += pk;
        
        // update gradient, value
        qk = -gk;
        f_xkm1 = f_xk;
        m_compFG(state_x, f_xk, gk);
        qk += gk;

        if(gk.squaredNorm() < gradstop*gradstop) 
            return ENDGRAD;
        
        if(abs(f_xk - f_xkm1) < valstop) 
            return ENDVALUE;

        if(f_xk < this->stop_F_under || f_xk > this->stop_F_over)
            return ENDABSVALUE;

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
        dk = -hessFuncTwoLoop(gamma, gk);
        m_callback(dk, f_xk, gk, iter);
    }
                   
    return ENDFAIL;
}

}
