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

#ifndef LINESEARCHER_H
#define LINESEARCHER_H

#include <map>
#include <list>

class LineSearcher
{
	public:
		LineSearcher(double left, double right, double xres, double yres);

		void getBest(double* xbest, double* ybest);
		bool getNextX(double* nextx);
		void addResult(double x, double y);

	private:
		double m_minxrange;
		double m_minyrange;
		
		std::list<double> m_queue;
		std::map<double, double> m_done;

		bool m_xdone;
		bool m_ydone;
};

#endif 

