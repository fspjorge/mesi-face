#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "face_matcher.h"
#include "face_pipeline.h"

struct BatchRunSummary {
    std::string label;
    std::filesystem::path csvPath;
    std::size_t gallerySize = 0;
    std::size_t queryCount = 0;
    double meanTopCorrelation = 0.0;
    double meanSrr1 = 0.0;
    double meanSrr2 = 0.0;
};

bool hasImageExtension(const std::filesystem::path& path);
std::vector<std::filesystem::path> collectImages(const std::filesystem::path& dir);
std::vector<ProcessedFace> loadGallery(const FacePipeline& pipeline, const std::filesystem::path& dir);
BatchRunSummary runBatchExperiment(const FacePipeline& pipeline,
                                   const FaceMatcher& matcher,
                                   const std::filesystem::path& galleryDir,
                                   const std::filesystem::path& queryDir,
                                   const std::filesystem::path& outputDir,
                                   const std::string& label);
std::string formatIdentificationSummary(const IdentificationResult& result);
void writeBatchCsv(const std::filesystem::path& outputPath, const std::vector<IdentificationResult>& results);
