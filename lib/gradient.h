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
 * @file gradient.h Definition for the GradientOpt class which implements 
 * a gradient descent energy minimization (optimization) algorithm.
 * 
 *****************************************************************************/

#include <iostream>
#include <cmath>

namespace npl {  

using alglib::real_1d_array;
using std::max;
using std::abs;

class GradOpt
{
	
public:
	double m_EpsG;
	double m_EpsX;
	double m_EpsF;
	int m_MaxIts;
	real_1d_array m_x; 
	double m_scale;
	double m_dscale;  // decrease in scale
	double m_rdscale; // relative decrease in scale
	double m_maxG; 


	/**
	 * @brief Constructor of Gradient Optimizer
	 *
	 * @param EpsG	Gradient-based stopping Condition
	 * @param EpsF	Energy-Based Stopping Condition
	 * @param EpsX	State-Based Stopping Condition
	 * @param MaxIts	Maximum iterations
	 * @param x			Initial state
	 * @param scale		Scale factor to multiply gradient by
	 * @param rdscale	Relative change in scale at each step, the scale will
	 * 					by multiplied by this at each step, recommend .99
	 */
	GradOpt(double EpsG, double EpsF, double EpsX, int MaxIts, 
			const real_1d_array& x, double scale, 
			double rdscale = 1) 
	{
		// don't allow scales > 1, that would lead to infinitely 
		// increasing of scale

		if(rdscale > 1) {
			m_rdscale = .99;
			std::cerr << std::endl << "WARNING! Changing m_dscale to .99!" 
				<< std::endl;
		} else
			m_rdscale = rdscale;

		m_EpsF = EpsF;
		m_EpsX = EpsX;
		m_EpsG = EpsG;
		m_MaxIts = MaxIts;
		m_scale = scale;
		m_maxG = 0;

		m_x.setlength(x.length());
		for(int ii = 0 ; ii < x.length(); ii++)
			m_x[ii] = x[ii];
	};

	void getresults(real_1d_array& x)
	{
		x.setlength(m_x.length());
		for(int ii = 0 ; ii < x.length(); ii++)
			x[ii] = m_x[ii];
	};

	int optimize(
		void(*update)(const real_1d_array &x, double &func, real_1d_array &grad,
					void *ptr),
		void(*callback)(const real_1d_array &x, double func,  void *ptr),
		void* ptr)
	{
		real_1d_array grad;
		grad.setlength(m_x.length());

		m_maxG = 0;
		double cur = 0;
		double prev = 0;
		double gnorm; 
		double delta = m_EpsF+1;
		for(int iter  = 0 ; (m_MaxIts <= 0 || iter < m_MaxIts); iter++) {
			prev = cur;

			update(m_x, cur, grad, ptr);

			gnorm = 0;
			for(int ii = 0 ; ii < grad.length(); ii++) 
				gnorm += grad[ii]*grad[ii];
			gnorm = sqrt(gnorm);

			for(int ii = 0 ; ii < m_x.length(); ii++) 
				m_x[ii] -= m_scale*grad[ii];

			if(callback) 
				callback(m_x, cur, ptr);

			m_maxG = gnorm > m_maxG ? gnorm : m_maxG;
			if(gnorm/m_maxG < m_EpsX) 
				return 2;
			
			if(gnorm < m_EpsG) 
				return 4;

			delta = (delta+fabs(cur-prev))/2.;
			if(delta < m_EpsF) 
				return 1;

			if(m_rdscale > 0 && m_rdscale < 1)
				m_scale *= m_rdscale;
		}

		return 5;
	};
};

}
