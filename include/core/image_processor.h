//
// Created by Pascal Adhikary on 4/5/21.
//

#pragma once

#include "core/image.h"

namespace naivebayes {

/**
 * The image processor, deserializing ascii grid to Image
 */
    class ImageProcessor {
    public:
        //friend std::ostream& operator <<(std::ostream& os, const WriteableVector& vector);
        friend std::istream& operator >>(std::istream& is, ImageProcessor& ip); // insertion operator
        vector<Image> GetImages() const;
        void AddImage(Image i);
        int line_count_ = 0;

    private:
        vector<Image> images_;
    };

}  // namespace naivebayes
