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
 * @file bfgs.cpp Implemenation of the BFGSOpt class which implements
 * a BFGS optimization (energy minimization) algorithm.
 *
 *****************************************************************************/

#include "bfgs.h"

#ifdef DEBUG
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
BFGSOpt::BFGSOpt(size_t dim, const ValFunc& valfunc,
        const GradFunc& gradfunc, const CallBackFunc& callback)
        : Optimizer(dim, valfunc, gradfunc, callback), m_lsearch(valfunc)
{
    state_Hinv = MatrixXd::Identity(dim, dim);
	
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
BFGSOpt::BFGSOpt(size_t dim, const ValFunc& valfunc, const GradFunc& gradfunc,
        const ValGradFunc& gradAndValFunc, const CallBackFunc& callback)
        : Optimizer(dim, valfunc, gradfunc, gradAndValFunc, callback),
        m_lsearch(valfunc)
{
    state_Hinv = MatrixXd::Identity(dim, dim);
	
	opt_ls_s = 1;
	opt_ls_beta = 0.5;
	opt_ls_sigma = 1e-5;
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
StopReason BFGSOpt::optimize()
{
    double gradstop = this->stop_G >= 0 ? this->stop_G : 0;
    double stepstop = this->stop_X >= 0 ? this->stop_X : 0;
    double valstop = this->stop_F >= 0 ? this->stop_F : 0;

    // update linesearch with minimum step, other options from opt_ls
    m_lsearch.opt_s = opt_ls_s;
    m_lsearch.opt_minstep = stepstop;
    m_lsearch.opt_beta = opt_ls_beta;
    m_lsearch.opt_sigma = opt_ls_sigma;

    const double ZETA = 1;
    MatrixXd& Dk = state_Hinv;
    VectorXd gk(state_x.rows()); // gradient
    double f_xk; // value at current position
    double f_xkm1; // value at previous position

    VectorXd xkprev;
    double tauk = 0;
    VectorXd pk, qk, dk, vk;

    //D(k+1) += p(k)p(k)'   - D(k)q(k)q(k)'D(k) + Z(k)T(k)v(k)v(k)'
    //          ----------    -----------------
    //         (p(k)'q(k))      q(k)'D(k)q(k)
    m_compFG(state_x, f_xk, gk);
    for(int iter = 0; stop_Its <= 0 || iter < stop_Its; iter++) {

        // optain direction
        dk = -Dk*gk;
        m_callback(dk, f_xk, gk, iter);

        // compute step size
        double alpha = m_lsearch.search(f_xk, state_x, gk, dk);
        pk = alpha*dk;

        if(alpha == 0 || pk.squaredNorm() < stepstop*stepstop)
            return ENDSTEP;

        // step
        xkprev = state_x;
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

        // update information inverse hessian
        tauk = qk.dot(Dk*qk);
        if(tauk < 1E-20)
            vk.setZero();
        else
            vk = pk/pk.dot(qk) - Dk*qk/tauk;

#ifdef DEBUG
		cerr << "iter: " << iter << endl;
		cerr << "pk: " << pk.transpose() << endl;
		cerr << "qk: " << qk.transpose() << endl;
		cerr << "vk: " << vk.transpose() << endl;
#endif
        Dk += pk*pk.transpose()/pk.dot(qk) - Dk*qk*qk.transpose()*Dk/
                    (qk.dot(Dk*qk)) + ZETA*tauk*vk*vk.transpose();

    }

    return ENDFAIL;
}


}
