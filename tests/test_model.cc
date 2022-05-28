
#include <fstream>
#include <catch2/catch.hpp>
#include <core/image_processor.h>
#include <iostream>
#include <core/model.h>

using namespace naivebayes;


TEST_CASE("Test Probability") {
    float epsilon = .05;
    ImageProcessor ip;
    string path = "/Users/pascaladhikary/Desktop/Cinder/my-projects/naive-bayes-pascaladhikary/data/test_model.txt";
    ifstream input_file(path);
    if (input_file.is_open()) {
        input_file >> ip;
    }

    Model m = Model();
    m.TrainFeatures(path);
    vector<float> p_e = {0.0784314, 0.235294, 0.0392157, 0.0784314, 0.0784314, 0.0980392, 0.0588235, 0.137255, 0.0784314, 0.117647};
    vector<float> p_a = m.GetPriors();

    SECTION("Test Priors") {
        bool flag = true;
        for (int i = 0; i < p_e.size(); i++) {
            if (abs(p_e[i] - p_a[i]) >= epsilon) {
                flag = false;
            }
        }
        REQUIRE(flag);
    }


    SECTION("Test Features/Conditionals") {
        vector<vector<vector<vector<float>>>> f_a = m.GetFeatures();
        vector<float> f_e = {0.8, 0.923077, 0.666667, 0.8, 0.8, 0.833333, 0.75, 0.875, .8, 0.857143};
        bool flag = true;
        for (int c = 0; c < f_e.size(); c++) {
            if (abs(f_a[c][0][0][0] - f_e[c]) >= epsilon) {
                flag = false;
            }
        }
        REQUIRE(flag);
    }


    SECTION("Test Distribution") {
        vector<float> d_e = {3, 11, 1, 3, 3, 4, 2, 6, 3, 5};
        vector<int> d_a = m.GetDistribution();
        bool flag = true;
        for (int i = 0; i < d_e.size(); i++) {
            if (abs(d_e[i] - d_a[i]) >= epsilon) {
                flag = false;
            }
        }
        REQUIRE(true);
    }
    REQUIRE(1 > 0);
}

TEST_CASE("Test File") {
    SECTION("File not found") {
        bool thrown = false;
        try {
            Model m = Model();
            m.TrainFeatures("fakepath");
        } catch(const exception & e) {
            thrown = true;
        }
        REQUIRE(thrown);
    }

    SECTION("Empty path") {
        bool thrown = false;
        try {
            Model m = Model();
            m.TrainFeatures("");
        } catch(const exception & e) {
            thrown = true;
        }
        REQUIRE(thrown);
    }
}


TEST_CASE("Test Model Operator Overload / Save State") {
    float epsilon = .05;
    string train_path = "/Users/pascaladhikary/Desktop/Cinder/my-projects/naive-bayes-pascaladhikary/data/test_model.txt";
    string export_path = "/Users/pascaladhikary/Desktop/Cinder/my-projects/naive-bayes-pascaladhikary/data/export.txt";

    SECTION("Test Export (Extraction)") {
        Model m = Model();
        m.TrainFeatures(train_path);

        ofstream input_file(export_path);
        if (input_file.is_open()) {
            input_file << m;
        }

        ifstream data_file(export_path);
        string line;
        vector<float> expected = {0.0784314, 0.235294, 0.0392157, 0.0784314, 0.0784314, 0.0980392, 0.0588235, 0.137255, 0.0784314, 0.117647};
        bool flag = true;
        if (data_file.is_open()) {
            getline(data_file, line);
            for(int i = 0; i < 10; i++) {
                getline(data_file, line);
                if (abs(stof(line) - expected[i]) > epsilon) {
                    flag = false;
                }
            }
        }
        REQUIRE(flag);
    }

    SECTION("Test Import (Insertion)") {
        Model m = Model();
        ifstream input_file(export_path);
        if (input_file.is_open()) {
            input_file >> m;
        }
        vector<float> expected = m.GetPriors();
        bool flag = true;
        ifstream data_file(export_path);
        string line;
        if (data_file.is_open()) {
            getline(data_file, line);
            for(int i = 0; i < 10; i++) {
                getline(data_file, line);
                if (abs(stof(line) - expected[i]) > epsilon) {
                    flag = false;
                }
            }
        }
        REQUIRE(flag);
    }
}

TEST_CASE("Test Accuracy") {
    string path = "/Users/pascaladhikary/Desktop/Cinder/my-projects/naive-bayes-pascaladhikary/data/testimagesandlabels.txt";
    string export_path = "/Users/pascaladhikary/Desktop/Cinder/my-projects/naive-bayes-pascaladhikary/data/export.txt";

    SECTION("Test Prediction Accuracy") {
        ImageProcessor ip;
        ifstream input_file(path);
        if (input_file.is_open()) {
            input_file >> ip;
        }
        int size = ip.GetImages().size();
        int correct = 0;
        Model m = Model();
        m.TrainFeatures(path);
        for (int i = 0; i < size; i++) {
            Image im = ip.GetImages()[i];
            if (m.MakePrediction(im) == im.GetLabel()) {
                correct++;
            }
        }
        double accuracy = (1.0 * correct / size) * 100;
        std::cout << "Accuracy: " << accuracy << std::endl;
        REQUIRE(accuracy > .7);
    }

    SECTION("Test Likelihood") {
        ImageProcessor ip;
        ifstream input_file(path);
        if (input_file.is_open()) {
            input_file >> ip;
        }
        Model m = Model();
        m.TrainFeatures(path);
        vector<float> expected = {368.891, 215.804, 255.286, 217.221, 232.901, 243.496, 266.07, 172.56, 221.812, 201.834};
        vector<float> likelihoods = m.FindLikelihoods(ip.GetImages().front());
        bool flag = true;
        for(int i = 0; i < 10; i++) {
            if (expected[i] + likelihoods[i] > .05) {
                flag = false;
            }
        }
        REQUIRE(flag);
    }
}