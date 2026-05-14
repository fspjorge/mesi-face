#include "face_pipeline.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "stasm_lib.h"
#include "stasm_landmarks.h"

using namespace std;
namespace fs = std::filesystem;

namespace {
constexpr int kNormalizedWidth = 200;
constexpr int kNormalizedHeight = 240;

double clamp01(double value) {
    return std::max(0.0, std::min(1.0, value));
}

cv::Point2f midpoint(const cv::Point2f& a, const cv::Point2f& b) {
    return cv::Point2f((a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f);
}

vector<cv::Point2f*> landmarkRefs(FaceLandmarks& landmarks) {
    return {
        &landmarks.leftEye,
        &landmarks.rightEye,
        &landmarks.leftBrowOuter,
        &landmarks.rightBrowOuter,
        &landmarks.eyebrowInner,
        &landmarks.noseTip,
        &landmarks.noseBase,
        &landmarks.mouthTop,
        &landmarks.mouthBottom,
        &landmarks.mouthLeftCorner,
        &landmarks.mouthRightCorner,
        &landmarks.chinTip,
    };
}

vector<cv::Point2f> illuminationAnchors(const FaceLandmarks& landmarks) {
    return {
        midpoint(landmarks.stasmShape.at(0), landmarks.stasmShape.at(39)),
        midpoint(landmarks.stasmShape.at(3), landmarks.stasmShape.at(39)),
        landmarks.eyebrowInner,
        midpoint(landmarks.eyebrowInner, landmarks.noseTip),
        landmarks.noseTip,
        landmarks.chinTip,
        midpoint(landmarks.stasmShape.at(14), landmarks.stasmShape.at(43)),
        midpoint(landmarks.stasmShape.at(10), landmarks.stasmShape.at(43)),
    };
}
}

FacePipeline::FacePipeline(fs::path cascadeDir,
                           fs::path outputDir,
                           bool illuminationNormalization,
                           string identityDelimiter)
    : stasmDataDir_(std::move(cascadeDir)),
      outputDir_(std::move(outputDir)),
      illuminationNormalization_(illuminationNormalization),
      identityDelimiter_(std::move(identityDelimiter)) {
    if (!fs::exists(stasmDataDir_ / "haarcascade_frontalface_alt2.xml")) {
        throw runtime_error("STASM data directory is missing haarcascade_frontalface_alt2.xml: " + stasmDataDir_.string());
    }
    if (!fs::exists(stasmDataDir_ / "haarcascade_mcs_lefteye.xml")) {
        throw runtime_error("STASM data directory is missing haarcascade_mcs_lefteye.xml: " + stasmDataDir_.string());
    }
    if (!fs::exists(stasmDataDir_ / "haarcascade_mcs_righteye.xml")) {
        throw runtime_error("STASM data directory is missing haarcascade_mcs_righteye.xml: " + stasmDataDir_.string());
    }
    if (!stasm_init(stasmDataDir_.string().c_str(), 0)) {
        throw runtime_error("stasm_init failed: " + string(stasm_lasterr()));
    }

    fs::create_directories(outputDir_ / "normalized");
    fs::create_directories(outputDir_ / "histograms");
}

bool FacePipeline::illuminationNormalizationEnabled() const noexcept {
    return illuminationNormalization_;
}

const fs::path& FacePipeline::outputDir() const noexcept {
    return outputDir_;
}

ProcessedFace FacePipeline::processImage(const fs::path& imagePath) const {
    const cv::Mat gray = cv::imread(imagePath.string(), cv::IMREAD_GRAYSCALE);
    if (gray.empty()) {
        throw runtime_error("Cannot load image: " + imagePath.string());
    }

    ProcessedFace processed;
    processed.sourcePath = imagePath;
    processed.identity = identityFromPath(imagePath, identityDelimiter_);
    processed.grayscale = gray;
    processed.landmarks = detectLandmarks(gray, imagePath);
    processed.sp = computeSp(processed.landmarks);
    processed.si = computeSi(gray, processed.landmarks, imagePath);
    processed.poseNormalized = normalizePose(gray, processed.landmarks);
    processed.normalized = processed.poseNormalized.clone();
    if (illuminationNormalization_) {
        processed.illuminationNormalized = normalizeIllumination(processed.normalized);
        processed.normalized = processed.illuminationNormalized.clone();
    } else {
        processed.illuminationNormalized = processed.poseNormalized.clone();
    }
    writeDebugArtifacts(processed);
    return processed;
}

string FacePipeline::identityFromPath(const fs::path& path, const string& delimiter) {
    const string stem = path.stem().string();
    const size_t pos = delimiter.empty() ? string::npos : stem.find(delimiter);
    return pos == string::npos ? stem : stem.substr(0, pos);
}

cv::Rect FacePipeline::clampRect(const cv::Rect& rect, const cv::Size& bounds) {
    const int x = max(0, rect.x);
    const int y = max(0, rect.y);
    const int maxWidth = max(0, bounds.width - x);
    const int maxHeight = max(0, bounds.height - y);
    return cv::Rect(x, y, min(rect.width, maxWidth), min(rect.height, maxHeight));
}

cv::Point2f FacePipeline::rotatePoint(const cv::Point2f& point, const cv::Point2f& center, double angleDegrees) {
    const double angleRadians = angleDegrees * CV_PI / 180.0;
    const cv::Point2f shifted(point.x - center.x, point.y - center.y);
    return cv::Point2f(
        static_cast<float>((shifted.x * cos(angleRadians)) - (shifted.y * sin(angleRadians)) + center.x),
        static_cast<float>((shifted.x * sin(angleRadians)) + (shifted.y * cos(angleRadians)) + center.y));
}

double FacePipeline::computePointsAngle(const cv::Point2f& pt1, const cv::Point2f& pt2) {
    return atan2(pt2.y - pt1.y, pt2.x - pt1.x) * 180.0 / CV_PI;
}

cv::Point2f FacePipeline::clampPoint(const cv::Point2f& point, const cv::Size& bounds) {
    return cv::Point2f(
        static_cast<float>(std::clamp(point.x, 0.0f, static_cast<float>(bounds.width - 1))),
        static_cast<float>(std::clamp(point.y, 0.0f, static_cast<float>(bounds.height - 1))));
}

cv::Mat FacePipeline::correctBandPerspective(const cv::Mat& src,
                                             const cv::Point2f& pt1,
                                             const cv::Point2f& pt2,
                                             const cv::Point2f& pt3) {
    cv::Point2f srcVertices[3] = {pt1, pt2, pt3};
    cv::Point2f dstVertices[3] = {
        cv::Point2f(src.cols * 0.5f - 1.0f, pt1.y),
        cv::Point2f(src.cols * 0.5f - 1.0f, pt2.y),
        cv::Point2f(static_cast<float>(src.cols - 1), pt2.y)
    };

    const cv::Mat affine = cv::getAffineTransform(srcVertices, dstVertices);
    cv::Mat warp;
    cv::warpAffine(src, warp, affine, src.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return warp;
}

cv::Mat FacePipeline::mirroredPatch(const cv::Mat& input, const cv::Rect& rect) {
    cv::Rect safe = clampRect(rect, input.size());
    if (safe.width <= 0 || safe.height <= 0) {
        return cv::Mat();
    }
    return input(safe).clone();
}

cv::Mat FacePipeline::buildGaussianKernel(int size) {
    cv::Mat kernel(size, size, CV_32F);
    const int half = size / 2;
    const double sigma = static_cast<double>(size) / 10.0;
    for (int row = 0; row < size; ++row) {
        for (int col = 0; col < size; ++col) {
            const int x = col - half;
            const int y = row - half;
            kernel.at<float>(row, col) = static_cast<float>(exp(-(static_cast<double>(x * x + y * y)) / sigma));
        }
    }
    return kernel;
}

cv::Mat FacePipeline::weightedGaussian(const cv::Mat& patch, const cv::Mat& gaussianKernel) {
    cv::Scalar meanScalar = cv::mean(patch);
    const double threshold = meanScalar[0];
    int gtCount = 0;
    int ltCount = 0;
    for (int row = 0; row < patch.rows; ++row) {
        for (int col = 0; col < patch.cols; ++col) {
            if (patch.at<float>(row, col) > threshold) {
                ++gtCount;
            } else {
                ++ltCount;
            }
        }
    }

    const bool moreThanThreshold = gtCount > ltCount;
    cv::Mat weighted = cv::Mat::zeros(gaussianKernel.size(), CV_32F);
    double scaleFactor = 0.0;
    for (int row = 0; row < patch.rows; ++row) {
        for (int col = 0; col < patch.cols; ++col) {
            const float patchValue = patch.at<float>(row, col);
            const bool discard = ((patchValue > threshold) && !moreThanThreshold)
                || ((patchValue < threshold) && moreThanThreshold);
            if (!discard) {
                const float gaussianValue = gaussianKernel.at<float>(row, col);
                weighted.at<float>(row, col) = gaussianValue;
                scaleFactor += gaussianValue;
            }
        }
    }

    if (scaleFactor > 0.0) {
        weighted /= static_cast<float>(scaleFactor);
    }
    return weighted;
}

double FacePipeline::computeSigmoid(double value) {
    return 1.0 / (1.0 + exp(-value / 160.0));
}

FaceLandmarks FacePipeline::detectLandmarks(const cv::Mat& gray, const fs::path& imagePath) const {
    int foundFace = 0;
    float landmarksRaw[2 * stasm_NLANDMARKS] = {};
    if (!stasm_search_single(&foundFace,
                             landmarksRaw,
                             reinterpret_cast<const char*>(gray.data),
                             gray.cols,
                             gray.rows,
                             imagePath.string().c_str(),
                             stasmDataDir_.string().c_str())) {
        throw runtime_error("stasm_search_single failed for " + imagePath.string() + ": " + string(stasm_lasterr()));
    }
    if (!foundFace) {
        throw runtime_error("No face detected in image: " + imagePath.string());
    }

    stasm_force_points_into_image(landmarksRaw, gray.cols, gray.rows);
    stasm_convert_shape(landmarksRaw, 68);

    constexpr int kConvertedLandmarks = 68;
    vector<cv::Point2f> shape(kConvertedLandmarks);
    for (int i = 0; i < kConvertedLandmarks; ++i) {
        shape[i] = cv::Point2f(landmarksRaw[i * 2], landmarksRaw[i * 2 + 1]);
    }

    float minX = static_cast<float>(gray.cols - 1);
    float minY = static_cast<float>(gray.rows - 1);
    float maxX = 0.0f;
    float maxY = 0.0f;
    for (const cv::Point2f& point : shape) {
        minX = std::min(minX, point.x);
        minY = std::min(minY, point.y);
        maxX = std::max(maxX, point.x);
        maxY = std::max(maxY, point.y);
    }

    FaceLandmarks landmarks;
    landmarks.stasmShape = shape;
    landmarks.faceBox = clampRect(
        cv::Rect(static_cast<int>(std::floor(minX)),
                 static_cast<int>(std::floor(minY)),
                 std::max(1, static_cast<int>(std::ceil(maxX - minX))),
                 std::max(1, static_cast<int>(std::ceil(maxY - minY)))),
        gray.size());

    landmarks.leftEye = shape[31];
    landmarks.rightEye = shape[36];
    landmarks.leftBrowOuter = shape[18];
    landmarks.rightBrowOuter = shape[29];
    landmarks.eyebrowInner = shape[24];
    landmarks.noseTip = shape[67];
    landmarks.noseBase = shape[41];
    landmarks.mouthTop = shape[51];
    landmarks.mouthBottom = shape[57];
    landmarks.mouthLeftCorner = shape[48];
    landmarks.mouthRightCorner = shape[54];
    landmarks.chinTip = shape[7];
    return landmarks;
}

cv::Mat FacePipeline::normalizePose(const cv::Mat& gray, FaceLandmarks& landmarks) const {
    const double angle = computePointsAngle(landmarks.leftEye, landmarks.rightEye);
    const cv::Point2f imageCenter(gray.cols * 0.5f, gray.rows * 0.5f);
    const cv::Mat rotation = cv::getRotationMatrix2D(imageCenter, angle, 1.0);

    cv::Mat rotated;
    cv::warpAffine(gray, rotated, rotation, gray.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    for (cv::Point2f* point : landmarkRefs(landmarks)) {
        *point = rotatePoint(*point, imageCenter, -angle);
    }
    for (cv::Point2f& point : landmarks.stasmShape) {
        point = rotatePoint(point, imageCenter, -angle);
    }

    const double dl = cv::norm(landmarks.leftEye - landmarks.noseTip);
    const double dr = cv::norm(landmarks.rightEye - landmarks.noseTip);
    if (dl > dr) {
        cv::flip(rotated, rotated, 1);
        for (cv::Point2f* point : landmarkRefs(landmarks)) {
            point->x = static_cast<float>(rotated.cols - 1) - point->x;
        }
        for (cv::Point2f& point : landmarks.stasmShape) {
            point.x = static_cast<float>(rotated.cols - 1) - point.x;
        }
        std::swap(landmarks.leftEye, landmarks.rightEye);
        std::swap(landmarks.leftBrowOuter, landmarks.rightBrowOuter);
        std::swap(landmarks.mouthLeftCorner, landmarks.mouthRightCorner);
    }

    const cv::Point2f lipTop = landmarks.mouthTop;
    const cv::Point2f lipBottom = landmarks.mouthBottom;

    int x1 = 0;
    int x2 = 0;
    if (dl > dr) {
        x1 = static_cast<int>(std::round(landmarks.stasmShape.at(13).x));
        x2 = static_cast<int>(std::round(landmarks.stasmShape.at(1).x));
    } else {
        x1 = static_cast<int>(std::round(landmarks.stasmShape.at(1).x));
        x2 = static_cast<int>(std::round(landmarks.stasmShape.at(13).x));
    }
    int y1 = static_cast<int>(std::round(landmarks.stasmShape.at(23).y - 40.0f));
    int y2 = static_cast<int>(std::round(landmarks.stasmShape.at(7).y + 5.0f));

    cv::Rect cropRect(x1, y1, std::max(1, x2 - x1), std::max(1, y2 - y1));
    cropRect = clampRect(cropRect, rotated.size());
    if (cropRect.width <= 0 || cropRect.height <= 0) {
        throw runtime_error("Unable to compute a valid pose crop");
    }

    cv::Mat crop = rotated(cropRect).clone();
    auto toCrop = [&](const cv::Point2f& point) {
        return clampPoint(cv::Point2f(point.x - cropRect.x, point.y - cropRect.y), crop.size());
    };

    const cv::Point2f noseTip = toCrop(landmarks.noseTip);
    const cv::Point2f noseTop = toCrop(midpoint(landmarks.stasmShape.at(24), landmarks.stasmShape.at(18)));
    const cv::Point2f topCenter(noseTop.x, 0.0f);
    const cv::Point2f noseBase = toCrop(landmarks.noseBase);
    const cv::Point2f lipTopCrop = toCrop(lipTop);
    const cv::Point2f lipBottomCrop = toCrop(lipBottom);
    const cv::Point2f chinTip = toCrop(landmarks.chinTip);
    const cv::Point2f bottomCenter(chinTip.x, static_cast<float>(crop.rows - 1));

    cv::Mat band1 = correctBandPerspective(crop, topCenter, noseTop, cv::Point2f(static_cast<float>(crop.cols - 1), noseTop.y));
    cv::Mat band2 = correctBandPerspective(crop, noseTop, noseTip, cv::Point2f(static_cast<float>(crop.cols - 1) - std::abs(noseTop.x - noseTip.x), noseTip.y));
    cv::Mat band3 = correctBandPerspective(crop, noseTip, noseBase, cv::Point2f(static_cast<float>(crop.cols - 1) - std::abs(noseTip.x - noseBase.x), noseBase.y));
    cv::Mat band4 = correctBandPerspective(crop, noseBase, lipTopCrop, cv::Point2f(static_cast<float>(crop.cols - 1) - std::abs(noseBase.x - lipTopCrop.x), lipTopCrop.y));
    cv::Mat band5 = correctBandPerspective(crop, lipTopCrop, lipBottomCrop, cv::Point2f(static_cast<float>(crop.cols - 1) - std::abs(lipTopCrop.x - lipBottomCrop.x), lipBottomCrop.y));
    cv::Mat band6 = correctBandPerspective(crop, lipBottomCrop, chinTip, cv::Point2f(static_cast<float>(crop.cols - 1) - std::abs(lipBottomCrop.x - chinTip.x), chinTip.y));
    cv::Mat band7 = correctBandPerspective(crop, chinTip, bottomCenter, cv::Point2f(static_cast<float>(crop.cols - 1) - std::abs(chinTip.x - bottomCenter.x), bottomCenter.y));

    cv::Mat stretch(crop.rows, crop.cols, CV_8U);
    for (int row = 0; row < crop.rows; ++row) {
        if (row < noseTop.y) {
            band1.row(row).copyTo(stretch.row(row));
        } else if (row < noseTip.y) {
            band2.row(row).copyTo(stretch.row(row));
        } else if (row < noseBase.y) {
            band3.row(row).copyTo(stretch.row(row));
        } else if (row < lipTopCrop.y) {
            band4.row(row).copyTo(stretch.row(row));
        } else if (row < lipBottomCrop.y) {
            band5.row(row).copyTo(stretch.row(row));
        } else if (row < chinTip.y) {
            band6.row(row).copyTo(stretch.row(row));
        } else {
            band7.row(row).copyTo(stretch.row(row));
        }
    }

    cv::Mat mirror(crop.rows, crop.cols, CV_8U);
    for (int row = 0; row < crop.rows; ++row) {
        for (int col = 0; col < crop.cols; ++col) {
            if (col < crop.cols / 2) {
                const int srcRow = std::clamp(row + 1, 0, stretch.rows - 1);
                const int srcCol = std::clamp(crop.cols / 2 + (crop.cols / 2 - col - 1), 0, stretch.cols - 1);
                mirror.at<uchar>(row, col) = stretch.at<uchar>(srcRow, srcCol);
            } else {
                const int srcCol = std::clamp(col - 1, 0, stretch.cols - 1);
                mirror.at<uchar>(row, col) = stretch.at<uchar>(row, srcCol);
            }
        }
    }

    cv::Mat normalized;
    cv::resize(mirror, normalized, cv::Size(kNormalizedWidth, kNormalizedHeight), 0.0, 0.0, cv::INTER_LINEAR);
    landmarks.faceBox = cv::Rect(0, 0, kNormalizedWidth, kNormalizedHeight);
    return normalized;
}

cv::Mat FacePipeline::normalizeIllumination(const cv::Mat& normalized) const {
    cv::Mat srcFloat;
    normalized.convertTo(srcFloat, CV_32F);

    const int filterSizes[] = {3, 9, 15};
    cv::Mat accumulated = cv::Mat::zeros(srcFloat.size(), CV_32F);

    for (int filterSize : filterSizes) {
        const cv::Mat gaussian = buildGaussianKernel(filterSize);
        cv::Mat filtered = cv::Mat::zeros(srcFloat.size(), CV_32F);
        const int half = filterSize / 2;

        for (int row = 0; row < srcFloat.rows; ++row) {
            for (int col = 0; col < srcFloat.cols; ++col) {
                cv::Rect rect(col - half, row - half, filterSize, filterSize);
                cv::Mat patch = mirroredPatch(srcFloat, rect);
                if (patch.empty()) {
                    continue;
                }
                if (patch.rows != filterSize || patch.cols != filterSize) {
                    cv::copyMakeBorder(patch,
                                       patch,
                                       std::max(0, half - row),
                                       std::max(0, row + half + 1 - srcFloat.rows),
                                       std::max(0, half - col),
                                       std::max(0, col + half + 1 - srcFloat.cols),
                                       cv::BORDER_REFLECT_101);
                    patch = patch(cv::Rect(0, 0, filterSize, filterSize)).clone();
                }

                cv::Mat weighted = weightedGaussian(patch, gaussian);
                filtered.at<float>(row, col) = static_cast<float>(cv::sum(weighted.mul(patch))[0]);
            }
        }

        cv::Mat qi;
        cv::log(srcFloat + 1.0f, qi);
        cv::Mat logFiltered;
        cv::log(filtered + 1.0f, logFiltered);
        accumulated += (qi - logFiltered) / static_cast<float>(log(10.0));
    }

    cv::normalize(accumulated, accumulated, 0.0, 255.0, cv::NORM_MINMAX);
    cv::Mat result;
    accumulated.convertTo(result, CV_8U);
    return result;
}

double FacePipeline::computeSp(const FaceLandmarks& landmarks) const {
    const double theta = atan2(landmarks.leftEye.y - landmarks.rightEye.y,
                               landmarks.leftEye.x - landmarks.rightEye.x);
    const double roll = min(abs((2.0 * theta) / CV_PI), 1.0);

    const double dl = cv::norm(landmarks.leftEye - landmarks.noseTip);
    const double dr = cv::norm(landmarks.rightEye - landmarks.noseTip);
    const double yaw = max(dl, dr) > 0.0 ? (max(dl, dr) - min(dl, dr)) / max(dl, dr) : 1.0;

    const double eu = cv::norm(landmarks.eyebrowInner - landmarks.noseTip);
    const double ed = cv::norm(landmarks.noseBase - landmarks.chinTip);
    const double pitch = max(eu, ed) > 0.0 ? (max(eu, ed) - min(eu, ed)) / max(eu, ed) : 1.0;

    constexpr double alpha = 0.1;
    constexpr double beta = 0.6;
    constexpr double gamma = 0.3;
    return clamp01(alpha * (1.0 - roll) + beta * (1.0 - yaw) + gamma * (1.0 - pitch));
}

double FacePipeline::computeMassCenter(const cv::Mat& patch) const {
    cv::Mat hist;
    const int channels[] = {0};
    const int histSize[] = {256};
    float range[] = {0.0f, 256.0f};
    const float* ranges[] = {range};

    cv::calcHist(&patch, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);

    double weighted = 0.0;
    double total = 0.0;
    for (int i = 0; i < 256; ++i) {
        const double count = hist.at<float>(i, 0);
        weighted += static_cast<double>(i) * count;
        total += count;
    }
    return total > 0.0 ? weighted / total : 0.0;
}

double FacePipeline::computeSi(const cv::Mat& gray, const FaceLandmarks& landmarks, const fs::path& imagePath) const {
    const int patchSize = max(8, static_cast<int>(round(min(landmarks.faceBox.width, landmarks.faceBox.height) * 0.10)));
    vector<double> massCenters;
    const auto anchors = illuminationAnchors(landmarks);
    const string stem = imagePath.stem().string();

    for (size_t i = 0; i < anchors.size(); ++i) {
        const cv::Rect patchRect(static_cast<int>(round(anchors[i].x - patchSize * 0.5f)),
                                 static_cast<int>(round(anchors[i].y - patchSize * 0.5f)),
                                 patchSize,
                                 patchSize);
        const cv::Rect safePatch = clampRect(patchRect, gray.size());
        if (safePatch.width <= 0 || safePatch.height <= 0) {
            massCenters.push_back(0.0);
            continue;
        }

        const cv::Mat patch = gray(safePatch).clone();
        massCenters.push_back(computeMassCenter(patch));
        cv::imwrite((outputDir_ / "histograms" / (stem + "_patch_" + to_string(i + 1) + ".png")).string(), patch);
    }

    if (massCenters.empty()) {
        return 0.0;
    }

    const double mean = accumulate(massCenters.begin(), massCenters.end(), 0.0) / static_cast<double>(massCenters.size());
    double variance = 0.0;
    for (double value : massCenters) {
        variance += (value - mean) * (value - mean);
    }
    variance /= static_cast<double>(massCenters.size());

    const double stddev = sqrt(variance);
    return clamp01(1.0 - computeSigmoid(stddev));
}

void FacePipeline::writeDebugArtifacts(const ProcessedFace& processed) const {
    const auto stem = processed.sourcePath.stem().string();
    cv::imwrite((outputDir_ / "normalized" / (stem + "_pose.png")).string(), processed.poseNormalized);
    cv::imwrite((outputDir_ / "normalized" / (stem + "_normalized.png")).string(), processed.normalized);
    if (!processed.illuminationNormalized.empty()) {
        cv::imwrite((outputDir_ / "normalized" / (stem + "_sqi.png")).string(), processed.illuminationNormalized);
    }
}
