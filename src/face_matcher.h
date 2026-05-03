#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "face_pipeline.h"

struct MatchScore {
    std::string identity;
    std::filesystem::path imagePath;
    double correlation = 0.0;
    double distance = 0.0;
};

struct ReliabilityMetrics {
    double phi1 = 0.0;
    double phi2 = 0.0;
    double srr1 = 0.0;
    double srr2 = 0.0;
    double dg1 = 0.0;
    double dg2 = 0.0;
    double dgG = 0.0;
    double dmax = 0.0;
};

struct IdentificationResult {
    ProcessedFace query;
    std::vector<MatchScore> ranking;
    ReliabilityMetrics reliability;
};

struct TrainingSummary {
    std::size_t sampleCount = 0;
    double meanSrr1 = 0.0;
    double stddevSrr1 = 0.0;
    double threshold = 0.0;
    std::vector<double> srr1Values;
};

class FaceMatcher {
public:
    IdentificationResult identify(const ProcessedFace& query, const std::vector<ProcessedFace>& gallery) const;
    TrainingSummary train(const std::vector<ProcessedFace>& gallery) const;

private:
    static double computeLocalCorrelation(const cv::Mat& imageA, const cv::Mat& imageB);
    static double computeGlobalCorrelation(const cv::Mat& regionA, const cv::Mat& regionB);
    static double computeQls(double distance, double maxDistance);
    static ReliabilityMetrics computeReliability(const std::vector<MatchScore>& ranking);
    static double computeMean(const std::vector<double>& values);
    static double computeStdDev(const std::vector<double>& values, double mean);
};
