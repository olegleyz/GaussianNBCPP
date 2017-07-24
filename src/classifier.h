
#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include <map>

using namespace std;

class GNB {
public:

    map <string, double> p_class_;
    map <string, vector<vector<double>>> f_stats_; // 0 - mean; 1 - var
    vector <string> labels_list_;
    int features_count_;
    /**
    * Constructor
    */
    GNB();

    /**
    * Destructor
    */
    virtual ~GNB();

    void train(vector<vector<double> > data, vector<string>  labels);

    string predict(vector<double> vec);

};

#endif



