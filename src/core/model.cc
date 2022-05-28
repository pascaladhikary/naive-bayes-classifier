#include "core/model.h"
#include <fstream>


namespace naivebayes {

    /**
     * Constructor intializing vectors
     */
    Model::Model() {
        vector<vector<vector<vector<float>>>> vec(kClasses, vector<vector<vector<float>>>
                (kSize, vector<vector<float>>
                        (kSize, vector<float>
                                (kBinary))));
        features_ = vec;
        vector<int> vec2(kClasses, 0);
        distribution_ = vec2;
        vector<float> vec3(kClasses, 0);
        priors_ = vec3;
    }

    /**
     * Returns priors
     * @return A 1d float vector indexed by class
     */
    vector<float> Model::GetPriors() const{
        return priors_;
    }

    /**
     * Returns distribution
     * @return An int vector of distributions
     */
    vector<int> Model::GetDistribution() const{
        return distribution_;
    }

    /**
     * Returns features
     * @return A 4d float vector of features, by coordinate, number, then binary indicator
     */
    std::vector<std::vector<std::vector<std::vector<float>>>> Model::GetFeatures() const{
        return features_;
    }

    /**
     * Trains the model against the dataset
     * @param path, a string for filepath
     */
    void Model::TrainFeatures(const string& path) {
        ifstream input_file(path);
        if (input_file.is_open()) {
            input_file >> image_processor_;
        } else {
            throw runtime_error("Could not open file");
        }

        vector<Image> list = image_processor_.GetImages();
        for(const Image& i: list) {
            distribution_[i.GetLabel()]++;
        }

        for(int i = 0; i < kClasses; i++) {
            priors_[i] = ((distribution_[i] + 1) / (kClasses + image_processor_.GetImages().size()*1.0));
        }

        for (int c = 0; c < kClasses; c++) {
            for (int i = 0; i < kSize; i++) {
                for (int j = 0; j < kSize; j++) {
                    for (int s = 0; s < kBinary; s++) {
                        double count = 0;
                        //loop through images, if label & shade matches -> increment count
                        for(const Image& im: list) {
                            if (im.GetLabel() == c && im.GetValue(i,j) == s) {
                                count++;
                            }
                        }
                        features_[c][i][j][s] = (count + 1) / (2 + distribution_[c]);
                    }
                }
            }
        }
    }

    vector<float> Model::FindLikelihoods(Image image) {
        /*
         * loop through 10 cases
         * for zero,
         * int shade = grid[x][y]
         * take log prior (zero) + loop through every pixel and and + log(feature[0][x][y][shade]
         */
        vector<float> likelihoods(kClasses, 0);
        vector<vector<int>> grid = image.GetGrid();
        for (int c = 0; c < kClasses; c++) {
            float likelihood = log(priors_[c]);
            for(int i = 0; i < kSize; i++) {
                for(int j = 0; j < kSize; j++) {
                    int shade = grid[i][j];
                    likelihood += log(features_[c][i][j][shade]);
                }
            }
            likelihoods[c] = likelihood;
        }
        return likelihoods;
    }

    /**
     * Makes the prediction for what number the ascii pattern is
     * @param image, the Image being predicted
     * @return an int for the predicted value
     */
    int Model::MakePrediction(Image image) {
        vector<float> likelihoods = FindLikelihoods(image);
        return std::max_element(likelihoods.begin(),likelihoods.end()) - likelihoods.begin();
    }

    /**
     * Overloads >> to import data to linear txt, top down priors then features numerically and by index
     */
    std::istream &operator>>(istream &is, Model &m) {
        string line;
        m = Model();
        while (getline(is, line)) {
            for (int i = 0; i < Model::kClasses; i ++) {
                getline(is, line);
                m.priors_[i] = stof(line);
            }
            for (int c = 0; c < Model::kClasses; c++) {
                for (int i = 0; i < Model::kSize; i++) {
                    for (int j = 0; j < Model::kSize; j++) {
                        for (int s = 0; s < Model::kBinary; s++) {
                            getline(is, line);
                            m.features_[c][i][j][s] = stof(line);
                        }
                    }
                }
            }
        }
        return is;
    }

    /**
     * Overloads << to export data to linear txt, top down priors then features numerically and by index
     */
    std::ostream &operator<<(ostream &os, Model &m) {
        os << endl;
        for (int i = 0; i < Model::kClasses; i++) {
            os << m.priors_[i] << endl;
        }
        for (int c = 0; c < Model::kClasses; c++) {
            for (int i = 0; i < Model::kSize; i++) {
                for (int j = 0; j < Model::kSize; j++) {
                    for (int s = 0; s < Model::kBinary; s++) {
                        os << m.features_[c][i][j][s] << endl;
                    }
                }
            }
        }
        return os;
    }

}  // namespace naivebayes