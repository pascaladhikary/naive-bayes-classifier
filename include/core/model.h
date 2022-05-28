//
// Created by Pascal Adhikary on 4/5/21.
//

#ifndef NAIVE_BAYES_MODEL_H
#define NAIVE_BAYES_MODEL_H

#endif //NAIVE_BAYES_MODEL_H

#include <vector>
#include <string>
#include "image_processor.h"

namespace naivebayes {

/**
 * The trained model, holding the feature and prior probabilities
 */
    class Model {
    public:
        Model();
        void TrainFeatures(const string& path);
        int MakePrediction(Image image);
        vector<float> FindLikelihoods(Image image);
        vector<float> GetPriors() const;
        vector<int> GetDistribution() const;
        std::vector<std::vector<std::vector<std::vector<float>>>> GetFeatures() const;
        friend std::istream& operator >>(std::istream& is, Model& m); // insertion operator
        friend std::ostream& operator <<(std::ostream& os, Model& m);

        const static int kClasses = 10;
        const static int kSize = 28;
        const static int kBinary = 2;

    private:
        vector<float> priors_;
        vector<vector<vector<vector<float>>>> features_;
        vector<int> distribution_;
        ImageProcessor image_processor_ = ImageProcessor();
    };

}  // namespace naivebayes
