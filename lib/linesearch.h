/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * 	http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file linesearch.h Definitions for classes which of several line search
 * algorithms.
 *
 *****************************************************************************/

#ifndef LINESEARCH_H
#define LINESEARCH_H

#include "opt.h"

namespace npl 
{

template <typename T>
inline 
void debug(T v)
{
#ifdef DEBUG
    cerr << v;
#endif //DEBUG
}

//class StrongWolfe
//{
//public:
//    StrongWolfe(const ComputeValFunc& valFunc const ComputeGradFunc& gradFunc) 
//    {
//        compVal = valFunc;
//        compGrad = gradFunc;
//    }
//
//    /**
//     * @brief Maximum step, if this is <= 0, then a quadratic fit will be used 
//     * to estimate a guess.
//     */
//    double opt_s;
//
//    /**
//     * @brief Power function base, values closer to 0 will decrease step size
//     * faster than ones close to 1.
//     */
//    double opt_beta;
//
//    /**
//     * @brief Theshold for stopping
//     */
//    double opt_sigma;
//
//    /**
//     * @brief Maximum number of iterations
//     */
//    int opt_maxIt; 
//
//    double search(double v_init, const Vector& x_init, const Vector& dir)
//    {
//        if(opt_alpha_step <= 0)
//            throw std::invalid_argument("opt_alpha_step must be > 0");
//
//        const double ALPHA_START = 1;
//        const double ALPHA_MAX = 1;
//
//        double dPhi_dAlpha_init = g_init.dot(dir);
//        double alpha_prev = 0;
//        double alpha = ALPHA_START;
//        double alpha_max = ALPHA_MAX;
//
//        Vector g; // gradient
//
//        double v_prev;
//        double v = v_init; // value
//        for(int iter = 0; iter < opt_maxIt; iter++) {
//            x = x_init + alpha*direction;
//            v_prev = v;
//            if(comp(x, v, g) != 0)
//                throw std::domain_error("Update function returned error");
//
//#ifdef DEBUG
//            fprintf(stderr, "Alpha: %f, Init Val: %f, Val: %f, C1: %f, phi'(0): %f\n",
//                    alpha, v_init, v, opt_c1, dPhi_dAlpha_init);
//#endif 
//            if(v > v_init + opt_c1*alpha*dPhi_dAlpha_init || 
//                        (iter>0 && v >= v_prev )) 
//                return zoom(v_prev, v);
// 
//            double dPhi_dAlpha = g.dot(dir);
//#ifdef DEBUG
//            fprintf(stderr, "phi'(alpha): %f, C: %f, phi'(0): %f\n",
//                    dPhi_dAlpha, opt_c2, dPhi_dAlpha_init);
//#endif 
//            if(fabs(dPhi_dAlpha) <= -opt_c2*dPhi_dAlpha_init)
//                return alpha;
//
//            if(dPhi_dAlpha >= 0)
//                return zoom(alpha, alpha_prev);
//
//            alpha_prev = alpha;
//            alpha *= opt_alpha_step;
//
//            if(alpha == alpha_prev)
//                throw std::runtime_error("Repeated alpha, will form inf loop");
//        }
//
//        return 0;
//    }
//
//    double zoom(double alpha_low, double alpha_hi) 
//    {
//#ifdef DEBUG
//            fprintf(stderr, "Zoom(%f, %f)\n", alpha_low, alpha_hi);
//#endif 
//
//            
//    }
//
//private:
//    ComputeValFunc compVal;
//};

}

#endif 

