#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>

struct FaceLandmarks {
    cv::Rect faceBox;
    cv::Point2f leftEye;
    cv::Point2f rightEye;
    cv::Point2f eyebrowInner;
    cv::Point2f noseTip;
    cv::Point2f noseBase;
    cv::Point2f chinTip;
};

struct ProcessedFace {
    std::filesystem::path sourcePath;
    std::string identity;
    cv::Mat grayscale;
    cv::Mat poseNormalized;
    cv::Mat illuminationNormalized;
    cv::Mat normalized;
    FaceLandmarks landmarks;
    double sp = 0.0;
    double si = 0.0;
};

class FacePipeline {
public:
    FacePipeline(std::filesystem::path cascadeDir,
                 std::filesystem::path outputDir,
                 bool illuminationNormalization,
                 std::string identityDelimiter);

    ProcessedFace processImage(const std::filesystem::path& imagePath) const;
    bool illuminationNormalizationEnabled() const noexcept;
    const std::filesystem::path& outputDir() const noexcept;

private:
    mutable cv::CascadeClassifier faceCascade_;
    mutable cv::CascadeClassifier leftEyeCascade_;
    mutable cv::CascadeClassifier rightEyeCascade_;
    std::filesystem::path outputDir_;
    bool illuminationNormalization_ = true;
    std::string identityDelimiter_;

    static std::string identityFromPath(const std::filesystem::path& path, const std::string& delimiter);
    static cv::Rect chooseLargest(const std::vector<cv::Rect>& boxes);
    static cv::Rect clampRect(const cv::Rect& rect, const cv::Size& bounds);
    static cv::Point2f rectCenter(const cv::Rect& rect);
    static std::optional<cv::Rect> detectSingle(cv::CascadeClassifier& cascade, const cv::Mat& gray, const cv::Rect& roi);
    static cv::Point2f rotatePoint(const cv::Point2f& point, const cv::Point2f& center, double angleDegrees);
    static double computePointsAngle(const cv::Point2f& pt1, const cv::Point2f& pt2);
    static cv::Mat correctBandPerspective(const cv::Mat& src, const cv::Point2f& pt1, const cv::Point2f& pt2, const cv::Point2f& pt3);
    static cv::Point2f clampPoint(const cv::Point2f& point, const cv::Size& bounds);
    static cv::Mat weightedGaussian(const cv::Mat& patch, const cv::Mat& gaussianKernel);
    static cv::Mat mirroredPatch(const cv::Mat& input, const cv::Rect& rect);
    static cv::Mat buildGaussianKernel(int size);
    static double computeSigmoid(double value);

    FaceLandmarks detectLandmarks(const cv::Mat& gray, const std::filesystem::path& imagePath) const;
    cv::Mat normalizePose(const cv::Mat& gray, FaceLandmarks& landmarks) const;
    cv::Mat normalizeIllumination(const cv::Mat& normalized) const;
    double computeSp(const FaceLandmarks& landmarks) const;
    double computeSi(const cv::Mat& gray, const FaceLandmarks& landmarks, const std::filesystem::path& imagePath) const;
    double computeMassCenter(const cv::Mat& patch) const;
    void writeDebugArtifacts(const ProcessedFace& processed) const;
};
