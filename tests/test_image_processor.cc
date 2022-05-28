
#include <fstream>
#include <catch2/catch.hpp>
#include <core/image_processor.h>
#include <core/image.h>
#include <iostream>
#include <core/model.h>

using namespace naivebayes;
using namespace std;

TEST_CASE("Operator Overload") {
    ImageProcessor ip;
    ifstream input_file("/Users/pascaladhikary/Desktop/Cinder/my-projects/naive-bayes-pascaladhikary/data/test_processor.txt");
    if (input_file.is_open()) {
        input_file >> ip;
    }

    SECTION("Test Load Images Fill") {
        vector<Image> images = ip.GetImages();
        REQUIRE(images[0].GetGrid()[0][0] == 1);
        REQUIRE(images[0].GetGrid()[1][0] == 1);
        REQUIRE(images[0].GetGrid()[0][1] == 1);
    }

    SECTION("Test Load Images Unfilled") {
        vector<Image> images = ip.GetImages();
        auto grid = images[0].GetGrid();
        REQUIRE(grid[1][1] == 0);
        REQUIRE(grid[1][2] == 0);
        REQUIRE(grid[1][3] == 0);
    }

    SECTION("Test Load Images Empty") {
        vector<Image> images = ip.GetImages();
        auto grid = images[1].GetGrid();
        bool flag = true;
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                if (grid[i][j] != 0) {
                    flag = false;
                }
            }
        }
        REQUIRE(flag);
    }

    SECTION("Test Load Images Odd Character Non-Blank") {
        vector<Image> images = ip.GetImages();
        auto grid = images[2].GetGrid();
        REQUIRE(images[0].GetGrid()[0][0] == 1);
        REQUIRE(images[0].GetGrid()[1][0] == 1);
        REQUIRE(images[0].GetGrid()[0][1] == 1);
    }
}