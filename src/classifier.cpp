#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include <map>
#include <assert.h>
#include <algorithm>
#include "classifier.h"
using namespace std;
#define M_PI 3.14159265359
/**
 * Initializes GNB
 */
GNB::GNB() {

}

GNB::~GNB() {}

void GNB::train(vector<vector<double>> data, vector<string> labels)
{  
    /*
        Trains the classifier with N data points and labels.

        INPUTS
        data - array of N observations
          - Each observation is a tuple with 4 values: s, d, 
            s_dot and d_dot.
          - Example : [
                [3.5, 0.1, 5.9, -0.02],
                [8.0, -0.3, 3.0, 2.2],
                ...
            ]

        labels - array of N labels
          - Each label is one of "left", "keep", or "right".
    */
    map <string, vector<vector<double>>> lfm;
    map <string, int> class_count;
    int train_size = labels.size();
    
    // assign get unique labels to labels_list
    labels_list_ = labels;
    std::sort(labels_list_.begin(), labels_list_.end());
    std::vector<string>::iterator newEnd;
     // Override duplicate elements
    newEnd = std::unique(labels_list_.begin(), labels_list_.end());
    labels_list_.erase(newEnd, labels_list_.end());
   
    features_count_ = data[0].size();
 
    // init mean and var
    for (auto lab : labels_list_){
        class_count[lab] = 0;
        vector <double> temp (data[0].size(), 0.0);
        f_stats_[lab].push_back(temp);
        f_stats_[lab].push_back(temp);
        f_stats_[lab].push_back(temp);
    }
 
    //gathering data list per class; count classes; sum per class
    for (auto i = 0; i < train_size; i++) {
        lfm[labels[i]].push_back(data[i]); // x_train per class
        class_count[labels[i]] += 1; // class count
        for (auto j = 0; j < features_count_; j++){
            f_stats_[labels[i]][0][j] += data[i][j]; // sum per feature
        }
    }

    // transforming f_stats 0 sum into mean
    for (auto lab : labels_list_){
        for (auto j = 0; j < features_count_; j++){
            f_stats_[lab][0][j] /= class_count[lab];
        }
        p_class_[lab] = class_count[lab] * 1.0 / labels.size();
    }

    // calc var by classes
    for (auto lab : labels_list_){
        for (auto j = 0; j < features_count_; j++){
            for (auto i = 0; i < lfm[lab].size(); i++){
                f_stats_[lab][1][j] += pow(lfm[lab][i][j] - f_stats_[lab][0][j], 2);
            }
            f_stats_[lab][1][j] /= class_count[lab];
            f_stats_[lab][2][j] = 1.0 / sqrt(2 * M_PI * f_stats_[lab][1][j]); // 1/sqrt(2*PI*var)
        }
    }

}

string GNB::predict(vector <double> vec)
{
    /*
        Once trained, this method is called and expected to return 
        a predicted behavior for the given observation.

        INPUTS

        observation - a 4 tuple with s, d, s_dot, d_dot.
          - Example: [3.5, 0.1, 8.5, -0.2]

        OUTPUT

        A label representing the best guess of the classifier. Can
        be one of "left", "keep" or "right".
        """
        # TODO - complete this
    */
    
    assert (features_count_ == vec.size());
    
    map <string, double> p;
    double max = 0;
    string result;

    for (auto lab:labels_list_) {
        p[lab] = p_class_[lab];
        for (auto i = 0; i < features_count_; i++){
            p[lab] *= f_stats_[lab][2][i] * exp(-pow(vec[i] - f_stats_[lab][0][i], 2) / (2 * f_stats_[lab][1][i]));
        }

        if (max < p[lab]) {
            max = p[lab];
            result = lab;
        }
    }
    return result;
}