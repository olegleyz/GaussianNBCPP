#include "classifier.h"
#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <map>

using namespace std;

vector<vector<double> > Load_State2(string file_name)
{
    ifstream in_state_(file_name.c_str(), ifstream::in);
    vector< vector<double >> state_out;
    string line;
    
    while (getline(in_state_, line)) 
    {
    	
    	istringstream iss(line);
    	vector<double> x_coord;
    	double state1;
	    double state2;
	    double state3;
	    double state4;
	    iss >> state1;
	    x_coord.push_back(state1);
	    iss >> state2;
	    x_coord.push_back(state2);
    	iss >> state3;
    	x_coord.push_back(state3);
	    iss >> state4;
	    x_coord.push_back(state4);
    
	    state_out.push_back(x_coord);
    }
    return state_out;
}

vector<vector<double> > Load_State(string file_name)
{
	ifstream in_state_(file_name.c_str(), ifstream::in);
	vector< vector<double >> state_out;
	string start;
	
	while (getline(in_state_, start))
	{

		vector<double> x_coord;

		istringstream ss(start);
		double a;
		ss >> a;
		x_coord.push_back(a);

		string value;

		while (getline(ss, value, ',')) {
			double b;
			ss >> b;
			x_coord.push_back(b);
		}
				
		state_out.push_back(x_coord);
	}
	return state_out;
}
vector<string> Load_Label(string file_name)
{
    ifstream in_label_(file_name.c_str(), ifstream::in);
    vector< string > label_out;
    string line;
    while (getline(in_label_, line)) 
    {
    	istringstream iss(line);
    	string label;
	    iss >> label;
    
	    label_out.push_back(label);
    }
    return label_out;
    
}

int main() {
    
    vector< vector<double> > X_train = Load_State("./../data/train_states.txt");
    vector< vector<double> > X_test  = Load_State("./../data/test_states.txt");
    vector< string > Y_train  = Load_Label("./../data/train_labels.txt");
    vector< string > Y_test   = Load_Label("./../data/test_labels.txt");
    
    cout << "X_train number of elements " << X_train.size() << endl;
	cout << "X_train element size " << X_train[0].size() << endl;
	cout << "Y_train number of elements " << Y_train.size() << endl << endl;

	GNB gnb = GNB();
	
	gnb.train(X_train, Y_train);

	cout << "X_test number of elements " << X_test.size() << endl;
	cout << "X_test element size " << X_test[0].size() << endl;
	cout << "Y_test number of elements " << Y_test.size() << endl << endl;
	
	int score = 0;
	for(int i = 0; i < X_test.size(); i++)
	{
		vector<double> coords = X_test[i];
		string predicted = gnb.predict(coords);
		if(predicted.compare(Y_test[i]) == 0)
		{
			score += 1;
		}
	}

	float fraction_correct = float(score) / Y_test.size();
	cout << "You got " << (100*fraction_correct) << " correct" << endl;

	return 0;
}

