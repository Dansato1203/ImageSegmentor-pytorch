#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <memory>

void detect_whiteline(torch::jit::script::Module &module, torch::Device &device, const cv::Mat &mat, float thre = 0.5, cv::Mat &result_g, cv::Mat result_w){
    torch::NoGradGuard no_grad;
    std::vector<int64_t> shape = {3, 3, mat.rows, mat.cols};
    std::vector<float> imdata(3 * mat.rows * mat.cols);
    for (size_t y = 0; y < mat.rows; ++y) {
        for (size_t x = 0; x < mat.cols; ++x) {
            const auto &pix = mat.at<cv::Vec3b>(y, x);
            imdata[(y * mat.cols + x) + 0 * mat.cols * mat.rows] = pix[0] / 255.;
            imdata[(y * mat.cols + x) + 1 * mat.cols * mat.rows] = pix[1] / 255.;
            imdata[(y * mat.cols + x) + 2 * mat.cols * mat.rows] = pix[2] / 255.;
        }
    }
    torch::Tensor tensor = torch::from_blob(&imdata[0], at::IntList(shape), at::ScalarType::Float).to(device);
    at::Tensor output_g = module.forward({tensor}).toTensor()[0][0];
    at::Tensor output_w = module.forward({tensor}).toTensor()[0][1];
    cv::Mat ou_g(mat.rows, mat.cols, CV_32FC1, output_g.to(torch::kCPU).data_ptr());
    cv::Mat out_w(mat.rows, mat.cols, CV_32FC1, output_w.to(torch::kCPU).data_ptr());
    /*
    cv::Mat result_g(mat.rows, mat.cols, CV_8UC1);
    cv::Mat result_w(mat.rows, mat.cols, CV_8UC1);
    */
    float *ptr_g = out.ptr_g<float>(0);
    float *ptr_w = out.ptr_w<float>(0);
    unsigned char *outp_g = result.ptr_g<unsigned char>(0);
    unsigned char *outp_w = result.ptr_w<unsigned char>(0);
    for (size_t i = 0; i < mat.cols * mat.rows; ++i) {
        if (*ptr_g > thre){
            *outp_g = 255;
        }else{
            *outp_g = 0;
        }

        if (*ptr_w > thre){
            *outp_w = 255;
        }else{
            *outp_w = 0;
        }

        ++ptr_g;
        ++outp_g;
        ++ptr_w;
        ++outp_w;
    }
};

int main(int argc, const char *argv[]) {
    cv::Mat mat;
    cv::Mat result_g, result_w;
    if (argc != 3) {
        std::cerr << "usage: example-app <path-to-exported-script-module> <image>\n";
        return -1;
    }

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::kCUDA;
        std::cerr << "+++ CUDA is available +++" << std::endl;
    } else {
        std::cerr << "CUDA is NOT available" << std::endl;
    }

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[1]);
        module.to(device);
        for (size_t k = 0; k < 10; ++k) {
            mat = cv::imread(argv[2]);
            cv::resize(mat, mat, cv::Size(320, 240));
            cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);

            auto t0 = std::chrono::system_clock::now();
            cv::Mat out = detect_whiteline(module, device, mat, 0.5, result_g, result_w);
            auto t1 = std::chrono::system_clock::now();
            std::cout << "processing time: " << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
                      << " us" << std::endl;

            cv::imshow("green", result_g);
            cv::imshow("white", result_w;
            cv::waitKey(0);
            //cv::imwrite("out.png", out);
        }
    }
    catch (const c10::Error &e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    std::cout << "ok\n";
}
