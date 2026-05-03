#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "face_matcher.h"
#include "face_pipeline.h"

bool hasImageExtension(const std::filesystem::path& path);
std::vector<std::filesystem::path> collectImages(const std::filesystem::path& dir);
std::vector<ProcessedFace> loadGallery(const FacePipeline& pipeline, const std::filesystem::path& dir);
std::string formatIdentificationSummary(const IdentificationResult& result);
void writeBatchCsv(const std::filesystem::path& outputPath, const std::vector<IdentificationResult>& results);
