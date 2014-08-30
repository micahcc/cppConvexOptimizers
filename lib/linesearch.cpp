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
 * @file linesearch.cpp Implemntation for linesearch algorithms to final minima
 * along a particular direction.
 *
 *****************************************************************************/

#include "linesearcher.h"
#include <cmath>
#include <cassert>
#include <iostream>

using namespace std;

const double TAU = (3-std::sqrt(5))/2.;

//constructor
LineSearcher::LineSearcher(double left, double right, double xres, double yres) 
{
	m_minxrange = xres;
	m_minyrange = yres;
	m_queue.push_back(left);
	m_queue.push_back(right);
	m_xdone = false;
	m_ydone = false;
};

//returns the largest y and x values so far
void LineSearcher::getBest(double* xbest, double* ybest)
{
	auto maxit = m_done.begin(); //points to the point after the largest gap
	double maxval = -INFINITY; 
	for(auto it = m_done.begin(); it != m_done.end(); it++) {
		if(it->second > maxval) {
			maxval = it->second;
			maxit = it;
		}
	}
	*xbest = maxit->first;
	*ybest = maxit->second;
};

//gets the next search point, returns true if it is done, according to
//the xres and yres originally passed
bool LineSearcher::getNextX(double* nextx)
{
	//this is sort of overly complicated to create a grid of 
	//search points, but I may switch to alternating between y
	//and x searches, in which case this would be necesssary
	if(m_queue.empty() && !m_xdone) {
		//search in largest x range
		auto it = m_done.begin(); //just the iterator
		auto gap = m_done.begin(); //points to the point after the largest gap
		double prev = it->first; 
		double maxgap = 0; 
		it++;
		for(; it != m_done.end(); it++) {
			double x = it->first;
			if(x-prev > maxgap) {
				maxgap = x-prev;
				gap = it;
			}
			prev = x;
		}

		double b = gap->first;
		--gap;
		double a = gap->first;

		if(fabs(b-a) >= m_minxrange) 
			m_queue.push_back((b+a)/2.);
	} 

	//if queue is still empty, switch to searching around max
	if(m_queue.empty() && !m_ydone) {
		m_xdone = true;
		//add points on each side of the largest point (if applicable)
		auto maxit = m_done.begin(); //points to the point after the largest gap
		double maxval = -INFINITY; 
		for(auto it = m_done.begin(); it != m_done.end(); it++) {
			if(it->second > maxval) {
				maxval = it->second;
				maxit = it;
			}
		}

		//add the point after the max
		double a = maxit->first;
		double ay = maxit->second;
		maxit++;
		if(maxit != m_done.end()) {
			if(fabs(ay-maxit->second) > m_minyrange) 
				m_queue.push_back((maxit->first+a)/2.);

		}
		maxit--;

		if(maxit != m_done.begin()) {
			maxit--;

			if(fabs(ay-maxit->second) > m_minyrange) 
				m_queue.push_back((maxit->first+a)/2.);

			maxit++;
		}
	}

	if(m_queue.empty()) {
		m_ydone = true;
		return true;//all done
	}

	assert(m_queue.front() != *nextx);
	*nextx = m_queue.front();
	m_queue.pop_front();
	return false;
};


//add a result of an operation to the internal map
void LineSearcher::addResult(double x, double y) 
{
	m_done[x] = y;
};
