#include "face_matcher.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

using namespace std;

namespace {
constexpr int kCorrelationBlock = 40;
constexpr double kEpsilon = 1e-9;
}

IdentificationResult FaceMatcher::identify(const ProcessedFace& query, const vector<ProcessedFace>& gallery) const {
    if (gallery.empty()) {
        throw runtime_error("Gallery is empty");
    }

    IdentificationResult result;
    result.query = query;

    for (const ProcessedFace& candidate : gallery) {
        const double correlation = computeGlobalCorrelation(query.normalized, candidate.normalized);
        result.ranking.push_back(MatchScore{
            candidate.identity,
            candidate.sourcePath,
            correlation,
            1.0 - correlation,
        });
    }

    sort(result.ranking.begin(), result.ranking.end(), [](const MatchScore& lhs, const MatchScore& rhs) {
        if (lhs.correlation == rhs.correlation) {
            return lhs.imagePath.string() < rhs.imagePath.string();
        }
        return lhs.correlation > rhs.correlation;
    });

    result.reliability = computeReliability(result.ranking);
    return result;
}

TrainingSummary FaceMatcher::train(const vector<ProcessedFace>& gallery) const {
    TrainingSummary summary;
    summary.sampleCount = gallery.size();
    if (gallery.empty()) {
        return summary;
    }

    for (const ProcessedFace& candidate : gallery) {
        const IdentificationResult result = identify(candidate, gallery);
        summary.srr1Values.push_back(result.reliability.srr1);
    }

    summary.meanSrr1 = computeMean(summary.srr1Values);
    summary.stddevSrr1 = computeStdDev(summary.srr1Values, summary.meanSrr1);
    summary.threshold = summary.meanSrr1 > kEpsilon
        ? abs(pow(summary.meanSrr1, 2.0) - summary.stddevSrr1) / summary.meanSrr1
        : 0.0;
    return summary;
}

double FaceMatcher::computeLocalCorrelation(const cv::Mat& imageA, const cv::Mat& imageB) {
    CV_Assert(imageA.size() == imageB.size());
    CV_Assert(imageA.type() == CV_8U && imageB.type() == CV_8U);

    const cv::Scalar meanA = cv::mean(imageA);
    const cv::Scalar meanB = cv::mean(imageB);

    double numerator = 0.0;
    double denomA = 0.0;
    double denomB = 0.0;

    for (int row = 0; row < imageA.rows; ++row) {
        for (int col = 0; col < imageA.cols; ++col) {
            const double diffA = static_cast<double>(imageA.at<uchar>(row, col)) - meanA[0];
            const double diffB = static_cast<double>(imageB.at<uchar>(row, col)) - meanB[0];
            numerator += diffA * diffB;
            denomA += diffA * diffA;
            denomB += diffB * diffB;
        }
    }

    const double denominator = sqrt(denomA * denomB);
    return denominator > kEpsilon ? numerator / denominator : 0.0;
}

double FaceMatcher::computeGlobalCorrelation(const cv::Mat& regionA, const cv::Mat& regionB) {
    if (regionA.empty() || regionB.empty()) {
        return 0.0;
    }

    const int blockSize = min({kCorrelationBlock, regionA.cols, regionA.rows, regionB.cols, regionB.rows});
    if (blockSize <= 0) {
        return 0.0;
    }

    double sum = 0.0;
    int count = 0;

    for (int y = 0; y + blockSize <= regionA.rows; y += blockSize) {
        for (int x = 0; x + blockSize <= regionA.cols; x += blockSize) {
            const cv::Mat blockA = regionA(cv::Rect(x, y, blockSize, blockSize));
            double localMax = -1.0;

            for (int offsetY = -blockSize; offsetY <= blockSize; offsetY += blockSize) {
                for (int offsetX = -blockSize; offsetX <= blockSize; offsetX += blockSize) {
                    const int bx = x + offsetX;
                    const int by = y + offsetY;
                    if (bx < 0 || by < 0 || bx + blockSize > regionB.cols || by + blockSize > regionB.rows) {
                        continue;
                    }

                    const cv::Mat blockB = regionB(cv::Rect(bx, by, blockSize, blockSize));
                    localMax = max(localMax, computeLocalCorrelation(blockA, blockB));
                }
            }

            if (localMax >= -1.0) {
                sum += localMax;
                ++count;
            }
        }
    }

    return count > 0 ? sum / static_cast<double>(count) : 0.0;
}

double FaceMatcher::computeQls(double distance, double maxDistance) {
    const double clampedDistance = max(0.0, 1.0 - distance);
    const double clampedMaxDistance = max(kEpsilon, 1.0 - maxDistance);
    const double a = 2.0 + sqrt(3.0);
    const double b = 7.0 - (4.0 * sqrt(3.0));
    const double exponent = clampedDistance / clampedMaxDistance;
    return (1.0 - pow(b, exponent)) / (a * pow(b, exponent) + 1.0);
}

ReliabilityMetrics FaceMatcher::computeReliability(const vector<MatchScore>& ranking) {
    ReliabilityMetrics metrics;
    if (ranking.empty()) {
        return metrics;
    }

    metrics.dmax = ranking.front().distance;
    for (const MatchScore& match : ranking) {
        metrics.dmax = max(metrics.dmax, match.distance);
    }
    metrics.dmax = max(metrics.dmax, kEpsilon);

    const string bestIdentity = ranking.front().identity;
    metrics.dg1 = computeQls(ranking.front().distance, metrics.dmax);
    metrics.dgG = computeQls(ranking.back().distance, metrics.dmax);

    bool foundDifferentIdentity = false;
    double nb = 0.0;
    for (const MatchScore& match : ranking) {
        const double qls = computeQls(match.distance, metrics.dmax);
        if (!foundDifferentIdentity && match.identity != bestIdentity) {
            metrics.dg2 = qls;
            foundDifferentIdentity = true;
        }
        if (match.identity != bestIdentity && qls < (2.0 * metrics.dg1)) {
            nb += 1.0;
        }
    }

    if (!foundDifferentIdentity) {
        metrics.dg2 = metrics.dgG;
    }

    metrics.phi1 = metrics.dgG > kEpsilon ? (metrics.dg2 - metrics.dg1) / metrics.dgG : 0.0;
    metrics.phi2 = 1.0 - (nb / static_cast<double>(ranking.size()));

    constexpr double criticalPoint = 0.1;
    const double s1 = metrics.phi1 > criticalPoint ? 1.0 - criticalPoint : criticalPoint;
    const double s2 = metrics.phi2 > criticalPoint ? 1.0 - criticalPoint : criticalPoint;
    metrics.srr1 = s1 > kEpsilon ? abs(metrics.phi1 - criticalPoint) / s1 : 0.0;
    metrics.srr2 = s2 > kEpsilon ? abs(metrics.phi2 - criticalPoint) / s2 : 0.0;
    return metrics;
}

double FaceMatcher::computeMean(const vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    return accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

double FaceMatcher::computeStdDev(const vector<double>& values, double mean) {
    if (values.empty()) {
        return 0.0;
    }

    double variance = 0.0;
    for (double value : values) {
        variance += (value - mean) * (value - mean);
    }
    variance /= static_cast<double>(values.size());
    return sqrt(variance);
}
