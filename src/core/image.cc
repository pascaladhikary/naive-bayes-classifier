#include <core/image.h>

namespace naivebayes {

    Image::Image() {
        for (int i = 0; i < size_; i++) {
            grid_.emplace_back(0);
            for (int j = 0; j < size_; j++) {
                grid_[i].emplace_back(0);
            }
        }
    }

    void Image::SetValue(int i, int j, int shaded) {
        grid_[i][j] = shaded;
    }

    vector<vector<int>> Image::GetGrid() const{
        return grid_;
    }

    void Image::SetGrid(vector<vector<int>> grid) {
        grid_ = grid;
    }

    void Image::SetLabel(int num) {
        label_ = num;
    }

    int Image::GetLabel() const{
        return label_;
    }

    int Image::GetValue(int i, int j) const{
        return grid_[i][j];
    }
}  // namespace naivebayes