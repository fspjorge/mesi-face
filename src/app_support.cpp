#include "app_support.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

using namespace std;
namespace fs = std::filesystem;

bool hasImageExtension(const fs::path& path) {
    static const vector<string> extensions = {".jpg", ".jpeg", ".png", ".bmp", ".pgm"};
    const string ext = path.extension().string();
    for (const string& candidate : extensions) {
        if (_stricmp(ext.c_str(), candidate.c_str()) == 0) {
            return true;
        }
    }
    return false;
}

vector<fs::path> collectImages(const fs::path& dir) {
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        throw runtime_error("Directory not found: " + dir.string());
    }

    vector<fs::path> images;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file() && hasImageExtension(entry.path())) {
            images.push_back(entry.path());
        }
    }

    sort(images.begin(), images.end());
    return images;
}

vector<ProcessedFace> loadGallery(const FacePipeline& pipeline, const fs::path& dir) {
    const vector<fs::path> files = collectImages(dir);
    if (files.empty()) {
        throw runtime_error("No images found in gallery: " + dir.string());
    }

    vector<ProcessedFace> gallery;
    gallery.reserve(files.size());
    for (const fs::path& path : files) {
        gallery.push_back(pipeline.processImage(path));
    }
    return gallery;
}

string formatIdentificationSummary(const IdentificationResult& result) {
    if (result.ranking.empty()) {
        throw runtime_error("No ranking produced");
    }

    const MatchScore& top = result.ranking.front();
    ostringstream stream;
    stream << fixed << setprecision(6);
    stream << "input_image=" << result.query.sourcePath.filename().string() << "\r\n";
    stream << "identity=" << result.query.identity << "\r\n";
    stream << "top_match=" << top.identity << "\r\n";
    stream << "top_match_file=" << top.imagePath.filename().string() << "\r\n";
    stream << "top_correlation=" << top.correlation << "\r\n";
    stream << "SP=" << result.query.sp << "\r\n";
    stream << "SI=" << result.query.si << "\r\n";
    stream << "SRR1=" << result.reliability.srr1 << "\r\n";
    stream << "SRR2=" << result.reliability.srr2 << "\r\n";
    stream << "ranking:\r\n";
    for (const MatchScore& match : result.ranking) {
        stream << "  " << match.identity << ',' << match.imagePath.filename().string() << ',' << match.correlation << "\r\n";
    }
    return stream.str();
}

void writeBatchCsv(const fs::path& outputPath, const vector<IdentificationResult>& results) {
    ofstream stream(outputPath);
    if (!stream) {
        throw runtime_error("Unable to write batch results: " + outputPath.string());
    }

    stream << "input_image,input_identity,top_match,top_match_file,top_correlation,SP,SI,SRR1,SRR2,status\n";
    stream << fixed << setprecision(6);
    for (const IdentificationResult& result : results) {
        const MatchScore& top = result.ranking.front();
        stream
            << result.query.sourcePath.filename().string() << ','
            << result.query.identity << ','
            << top.identity << ','
            << top.imagePath.filename().string() << ','
            << top.correlation << ','
            << result.query.sp << ','
            << result.query.si << ','
            << result.reliability.srr1 << ','
            << result.reliability.srr2 << ",ok\n";
    }
}
