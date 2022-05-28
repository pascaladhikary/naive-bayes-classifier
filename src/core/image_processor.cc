#include <iostream>
#include <core/model.h>
#include "core/image_processor.h"

namespace naivebayes {

    /**
     * Overloaded operator, takes data and converts into a list of Images
     * @param is, input stream
     * @param ip, Image Processor
     * @return
     */
    std::istream& operator>>(std::istream &is, ImageProcessor &ip) {
        string line;
        int count = 0;
        int line_count = 0;
        while (getline(is, line)) {
            line_count++;
            Image temp;
            temp.SetLabel(stoi(line));
            for (int i = 0; i < Model::kSize; i++) {
                line_count++;
                getline(is, line);
                for (int j = 0; j < line.size(); j++) {
                    int shaded;
                    line.at(j) == ' ' ? shaded = 0 : shaded = 1;
                    temp.SetValue(i, j, shaded);
                }
            }
            count++;
            ip.line_count_ = line_count;
            ip.AddImage(temp);
        }
        return is;
    }

    vector<Image> ImageProcessor::GetImages() const{
        return images_;
    }

    void ImageProcessor::AddImage(Image i) {
        images_.push_back(i);
    }

}  // namespace naivebayes