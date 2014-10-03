#ifndef __BP__
#define __BP__
#include <string>
#include <fstream>
#include <math.h>
using namespace std;

class BP{
private:
	int i_num;
	int h_num;
	int o_num;
	double *input;
	double *output;
	double **ih_weights;
	double **ho_weights;
	double lr;
    #define e  2.7182818
	inline double sigmoid(double x)
	{
		return  1.0 / (1 + pow(e, -1 * x));
	}
	inline double sigmoid_des(double x)
	{
		double y = sigmoid(x);
		return y*(1 - y);
	}
	int getInOut(char* line);
public:
	void init(int i, int h, int o);
	BP(int i, int h, int o) :i_num(i), h_num(h), o_num(o){
		init(i, h, o);
	}
	virtual ~BP()
	{
	}
	void train(string file, int iterTimes);
	int predict(string file, string results);
};
#endif
