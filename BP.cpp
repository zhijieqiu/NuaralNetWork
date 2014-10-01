#include "bp.h"
#include <vector>
using namespace std;

/*
method:init
param i:the number of input nodes
param h:the number of hidden nodes
param o:the number of output nodes
*/
void BP::init(int i, int h, int o)
{
	i_num = i;
	h_num = h;
	o_num = o;
	this->input = new double[i_num];
	this->output = new double[o_num];
	ih_weights = new double*[i_num];
	for (int j = 0; j < i_num; j++)
		ih_weights[j] = new double[h_num];
	ho_weights = new double*[h_num];
	for (int j = 0; j < h_num; j++)
		ho_weights[j] = new double[o_num];
}
/*
method:getInOut
line is read from the train file
for example : 1	2	3	4	5
when the number of numbers in the line is the same as i_num then return -1
else return the last number as the output of the train example
*/
int BP::getInOut(char* line)
{
	string tmp = "";
	int iIndex = 0;
	int i = 0;
	for (i = 0; i < strlen(line); i++)
	{
		if (line[i] == '\t')
		{
			input[iIndex] = atoi(tmp.c_str());
			tmp = "";
			iIndex++;
			if (iIndex == i_num)
				break;
		}
		else
			tmp += line[i];
	}
	i += 1;
	if (i >= strlen(line))
		return -1;
	while (i < strlen(line))
		tmp += line[i++];
	return atoi(tmp.c_str());
}
/*
method:train
param file:the train samples saved in the file 
param iterTimes:需要扫描几次该文件
*/
void BP::train(string file, int iterTimes)
{
	for (int k = 0; k < iterTimes; k++)
	{
		ifstream in(file);
		char line[2048];
		int getNum;
		while (in.getline(line, 2048))
		{
			int realLable = getInOut(line);
			double *tmpH = new double[h_num];
			for (int h = 0; h < h_num; h++)
			{
				tmpH[h] = 0.0;
				for (int i = 0; i < i_num; i++)
					tmpH[h] += this->input[i] * ih_weights[i][h];
			}
			double *tmpO = new double[o_num];
			for (int o = 0; o < o_num; o++)
			{
				tmpO[o] = 0.0;
				for (int h = 0; h < h_num; h++)
					tmpO[o] += sigmoid(tmpH[h]) * ho_weights[h][o];
				output[o] = sigmoid(tmpO[o]);
			}
			for (int o = 0; o < o_num; o++)
			{
				int lable = (o == realLable) ? 1 : 0;
				for (int h = 0; h < h_num; h++)
				{
					ho_weights[h][o] = ho_weights[h][o] - lr*(lable - output[o])*sigmoid_des(tmpO[o])*sigmoid(tmpH[h]);
				}
				for (int i = 0; i < h_num; i++)
				{
					for (int h = 0; h < h_num; h++)
					{
						double adder = lable - output[o] * sigmoid_des(tmpO[o]);
						double sum = 0.0;
						for (int j = 0; j < h_num; j++)
						{
							sum += ho_weights[j][o] * sigmoid_des(tmpH[j])*input[i];
						}
						ih_weights[i][h] -= lr*adder*sum;
					}
				}
			}
		}
		in.close();
	}
}
/*
method:predict
param file:the  samples need to predict is saved here and every is a sample
param results: your predict result will be saved here 
*/
int BP::predict(string file, string results)
{
	ifstream in(file);
	ofstream ofile(results);
	char line[2048];
	while (in.getline(line, 2048))
	{
		getInOut(line);
		double *tmpH = new double[h_num];
		for (int h = 0; h < h_num; h++)
		{
			tmpH[h] = 0.0;
			for (int i = 0; i < i_num; i++)
				tmpH[h] += this->input[i] * ih_weights[i][h];
		}
		double *tmpO = new double[o_num];
		for (int o = 0; o < o_num; o++)
		{
			tmpO[o] = 0.0;
			for (int h = 0; h < h_num; h++)
				tmpO[o] += sigmoid(tmpH[h]) * ho_weights[h][o];
			output[o] = sigmoid(tmpO[o]);
		}
		int maxIndex = 0;
		double maxValue = output[0];
		for (int i = 1; i < o_num; i++)
		{
			if (output[i]>maxValue)
			{
				maxIndex = i;
				maxValue = output[i];
			}
		}
		string tmp(line);
		tmp += "\t";
		_itoa_s(maxIndex, line, 10);
		tmp += string(line);
		ofile.write(tmp.c_str(), tmp.size());
	}
	in.close();
	ofile.close();
}
