//
// Created by Pascal Adhikary on 4/5/21.
//

#pragma once

#include <vector>

using namespace std;

namespace naivebayes {

/**
 * The image, holding a 2d vector binary representation of un(shaded)
 */
    class Image {
    public:
        Image();
        Image(vector<vector<int>>);
        void SetValue(int i, int j, int shaded);
        void SetLabel(int num);
        int GetLabel() const;
        int GetValue(int i, int j) const;
        vector<vector<int>> GetGrid() const;
        void SetGrid(vector<vector<int>> grid);

    private:
        int label_;
        int size_ = 28;
        vector<vector<int>> grid_;
    };

}  // namespace naivebayes
