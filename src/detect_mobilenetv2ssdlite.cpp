// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <stdio.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "platform.h"
#include "net.h"

#include "benchmark.h"



class Noop : public ncnn::Layer {};
DEFINE_LAYER_CREATOR(Noop)


static int detect_mobilenetv2ssdlite(unsigned char * buffer, int w, int h, int * result, unsigned int result_size, float* result_score)
{
    ncnn::Net mobilenetv2;


    mobilenetv2.register_custom_layer("Silence", Noop_layer_creator);

    // original pretrained model from https://github.com/chuanqi305/MobileNetv2-SSDLite
    // https://github.com/chuanqi305/MobileNetv2-SSDLite/blob/master/ssdlite/voc/deploy.prototxt
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    mobilenetv2.load_param("mobilenetv2_ssdlite_voc.param");
    mobilenetv2.load_model("mobilenetv2_ssdlite_voc.bin");

    const int target_size = 300;

    int img_w = w;
    int img_h = h;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(buffer, ncnn::Mat::PIXEL_BGR, w, h, target_size, target_size);

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.0/127.5,1.0/127.5,1.0/127.5};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = mobilenetv2.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(1);

    double start = ncnn::get_current_time();
    ncnn::Mat out;
    for (int i=0; i<1; i++)
    {
        ex.input("data", in);

        ex.extract("detection_out",out);
    }

    double end = ncnn::get_current_time();

    double time = end - start;

    printf("elapsed time: %f\n", time);

//     printf("%d %d %d\n", out.w, out.h, out.c);
    objects.clear();
    for (int i=0; i<out.h && i<result_size; i++)
    {
        result_score[i] = values[1];
        result[i] = values[0];
    }

    return 0;
}

namespace ncnn {
extern bool g_useqpu;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }


    ncnn::g_useqpu = true;
    fprintf(stderr, "testing g_useqpu\n");

    std::vector<Object> objects;
    detect_mobilenetv2(m, objects);

    /*
    ncnn::g_useqpu = false;
    fprintf(stderr, "testing no g_useqpu\n");

    std::vector<Object> objects1;
    detect_mobilenetv2(m, objects1);*/



    draw_objects(m, objects);

    return 0;
}
