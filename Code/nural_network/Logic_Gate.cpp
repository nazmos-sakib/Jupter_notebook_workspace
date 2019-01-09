#include <iostream>

using namespace std;

float and_fun(float x1, float x2)
{
	float result;
	result = x1*1 + x2*1 + 1*-1.5;
	return result;
}

float nand_fun(float x1, float x2)
{
	float result;
	result = -x1*1 - x2*1 + 1*1.5;
	return result;
}

float or_fun(float x1, float x2)
{
	float result;
	result = x1*1 + x2*1 + 1*-0.5;
	return result;
}

float nor_fun(float x1, float x2)
{
	float result;
	result = -x1*1 - x2*1 + 1*0.5;
	return result;
}

float not_fun(float x1)
{
	float result;
	result = -x1*1 + 1*0.5;
	return result;
}



int main()
{
	cout << "----------And Gate---------------"<<endl;
	cout <<"x1   x2    predected_value" <<endl;
	cout <<"0   0      " << and_fun(0,0) << endl;
	cout <<"0   1      " << and_fun(0,1) << endl;
	cout <<"1   0      " << and_fun(1,0) << endl;
	cout <<"1   1      " << and_fun(1,1) << endl<<endl<<endl;	

	cout << "----------Nand Gate---------------"<<endl;
	cout <<"x1   x2    predected_value" <<endl;
	cout <<"0   0      " << nand_fun(0,0) << endl;
	cout <<"0   1      " << nand_fun(0,1) << endl;
	cout <<"1   0      " << nand_fun(1,0) << endl;
	cout <<"1   1      " << nand_fun(1,1) << endl<<endl<<endl;


	cout << "----------Or Gate---------------"<<endl;
	cout <<"x1   x2    predected_value" <<endl;
	cout <<"0   0      " << or_fun(0,0) << endl;
	cout <<"0   1      " << or_fun(0,1) << endl;
	cout <<"1   0      " << or_fun(1,0) << endl;
	cout <<"1   1      " << or_fun(1,1) << endl<<endl<<endl;

	cout << "----------Nor Gate---------------"<<endl;
	cout <<"x1   x2    predected_value" <<endl;
	cout <<"0   0      " << nor_fun(0,0) << endl;
	cout <<"0   1      " << nor_fun(0,1) << endl;
	cout <<"1   0      " << nor_fun(1,0) << endl;
	cout <<"1   1      " << nor_fun(1,1) << endl<<endl<<endl;	


	cout << "----------Not Gate---------------"<<endl;
	cout <<"x1     predected_value" <<endl;
	cout <<"0      " << not_fun(0) << endl;
	cout <<"1      " << not_fun(1) << endl<<endl<<endl;


}
