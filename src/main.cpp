#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

#include "app_support.h"
#include "face_matcher.h"
#include "face_pipeline.h"

using namespace std;
namespace fs = std::filesystem;

namespace {
struct CliOptions {
    string mode;
    fs::path galleryDir;
    fs::path queryDir;
    fs::path queryImage;
    fs::path cascadeDir = "data";
    fs::path outputDir = "output";
    string identityDelimiter = "_";
    bool illuminationNormalization = true;
};

void printUsage() {
    cout
        << "Usage:\n"
        << "  face_cli train --gallery-dir <dir> [--cascade-dir <dir>] [--output-dir <dir>]\n"
        << "  face_cli identify --gallery-dir <dir> --query-image <file> [--cascade-dir <dir>] [--output-dir <dir>]\n"
        << "  face_cli batch --gallery-dir <dir> --query-dir <dir> [--cascade-dir <dir>] [--output-dir <dir>]\n"
        << "Options:\n"
        << "  --illumination on|off\n"
        << "  --identity-delimiter <text>\n";
}

CliOptions parseArgs(int argc, char** argv) {
    CliOptions options;
    if (argc < 2) {
        options.mode = "identify";
        options.galleryDir = "data";
        options.queryImage = "testface.jpg";
        options.cascadeDir = "data";
        options.outputDir = "output-vs-debug";
        return options;
    }

    options.mode = argv[1];

    map<string, string> args;
    for (int i = 2; i < argc; ++i) {
        const string key = argv[i];
        if (key == "--help" || key == "-h") {
            printUsage();
            std::exit(0);
        }
        if (i + 1 >= argc) {
            throw runtime_error("Missing value for argument: " + key);
        }
        args[key] = argv[++i];
    }

    if (args.count("--gallery-dir")) {
        options.galleryDir = args["--gallery-dir"];
    }
    if (args.count("--query-dir")) {
        options.queryDir = args["--query-dir"];
    }
    if (args.count("--query-image")) {
        options.queryImage = args["--query-image"];
    }
    if (args.count("--cascade-dir")) {
        options.cascadeDir = args["--cascade-dir"];
    }
    if (args.count("--output-dir")) {
        options.outputDir = args["--output-dir"];
    }
    if (args.count("--identity-delimiter")) {
        options.identityDelimiter = args["--identity-delimiter"];
    }
    if (args.count("--illumination")) {
        const string value = args["--illumination"];
        if (value == "on") {
            options.illuminationNormalization = true;
        } else if (value == "off") {
            options.illuminationNormalization = false;
        } else {
            throw runtime_error("Invalid value for --illumination. Use on or off.");
        }
    }

    return options;
}

}

int main(int argc, char** argv) {
    try {
        const CliOptions options = parseArgs(argc, argv);
        const FacePipeline pipeline(options.cascadeDir,
                                    options.outputDir,
                                    options.illuminationNormalization,
                                    options.identityDelimiter);
        const FaceMatcher matcher;

        if (options.mode == "train") {
            if (options.galleryDir.empty()) {
                throw runtime_error("train mode requires --gallery-dir");
            }
            const vector<ProcessedFace> gallery = loadGallery(pipeline, options.galleryDir);
            const TrainingSummary summary = matcher.train(gallery);
            cout << fixed << setprecision(6);
            cout << "samples=" << summary.sampleCount << '\n';
            cout << "mean_srr1=" << summary.meanSrr1 << '\n';
            cout << "stddev_srr1=" << summary.stddevSrr1 << '\n';
            cout << "threshold=" << summary.threshold << '\n';
            return 0;
        }

        if (options.mode == "identify") {
            if (options.galleryDir.empty() || options.queryImage.empty()) {
                throw runtime_error("identify mode requires --gallery-dir and --query-image");
            }
            const vector<ProcessedFace> gallery = loadGallery(pipeline, options.galleryDir);
            const ProcessedFace query = pipeline.processImage(options.queryImage);
            const IdentificationResult result = matcher.identify(query, gallery);
            cout << formatIdentificationSummary(result);
            return 0;
        }

        if (options.mode == "batch") {
            if (options.galleryDir.empty() || options.queryDir.empty()) {
                throw runtime_error("batch mode requires --gallery-dir and --query-dir");
            }
            const vector<ProcessedFace> gallery = loadGallery(pipeline, options.galleryDir);
            const vector<fs::path> queryFiles = collectImages(options.queryDir);
            if (queryFiles.empty()) {
                throw runtime_error("No query images found in: " + options.queryDir.string());
            }

            vector<IdentificationResult> results;
            results.reserve(queryFiles.size());
            for (const fs::path& queryFile : queryFiles) {
                results.push_back(matcher.identify(pipeline.processImage(queryFile), gallery));
            }

            fs::create_directories(options.outputDir);
            const fs::path csvPath = options.outputDir / "batch_results.csv";
            writeBatchCsv(csvPath, results);
            cout << "batch_results=" << csvPath.string() << '\n';
            cout << "samples=" << results.size() << '\n';
            return 0;
        }

        throw runtime_error("Unknown mode: " + options.mode);
    } catch (const exception& ex) {
        cerr << "error: " << ex.what() << '\n';
        printUsage();
        return 1;
    }
}
