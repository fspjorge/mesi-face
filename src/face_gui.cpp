#include <windows.h>
#include <commdlg.h>
#include <commctrl.h>
#include <shellapi.h>
#include <shlobj.h>

#include <filesystem>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <iomanip>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "app_support.h"

using namespace std;
namespace fs = std::filesystem;

namespace {
constexpr int kIdGalleryEdit = 1001;
constexpr int kIdGalleryBrowse = 1002;
constexpr int kIdImageEdit = 1003;
constexpr int kIdImageBrowse = 1004;
constexpr int kIdRun = 1005;
constexpr int kIdIllumination = 1006;
constexpr int kIdResultText = 1007;
constexpr int kIdOriginalImage = 1008;
constexpr int kIdPoseImage = 1009;
constexpr int kIdOpenOutput = 1010;
constexpr int kIdSqiImage = 1011;
constexpr int kIdProgressBar = 1012;
constexpr int kIdStatusText = 1013;
constexpr int kIdFeretRootEdit = 1014;
constexpr int kIdFeretRootBrowse = 1015;
constexpr int kIdFeretSubsetCombo = 1016;
constexpr int kIdFeretRun = 1017;
constexpr int kIdFeretSummaryText = 1018;

fs::path defaultStasmDataDir() {
    return fs::path("third_party") / "stasm" / "data";
}

struct GuiState {
    HWND mainWindow = nullptr;
    HWND galleryEdit = nullptr;
    HWND imageEdit = nullptr;
    HWND illuminationCheck = nullptr;
    HWND runButton = nullptr;
    HWND progressBar = nullptr;
    HWND statusText = nullptr;
    HWND resultText = nullptr;
    HWND originalImage = nullptr;
    HWND poseImage = nullptr;
    HWND sqiImage = nullptr;
    HWND feretRootEdit = nullptr;
    HWND feretSubsetCombo = nullptr;
    HWND feretRunButton = nullptr;
    HWND feretSummaryText = nullptr;
    HBITMAP originalBitmap = nullptr;
    HBITMAP poseBitmap = nullptr;
    HBITMAP sqiBitmap = nullptr;
    fs::path outputDir = "output-gui";
};

wstring toWide(const string& value) {
    if (value.empty()) {
        return L"";
    }
    const int size = MultiByteToWideChar(CP_UTF8, 0, value.c_str(), -1, nullptr, 0);
    wstring result(size - 1, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, value.c_str(), -1, result.data(), size);
    return result;
}

string toUtf8(const wstring& value) {
    if (value.empty()) {
        return "";
    }
    const int size = WideCharToMultiByte(CP_UTF8, 0, value.c_str(), -1, nullptr, 0, nullptr, nullptr);
    string result(size - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, value.c_str(), -1, result.data(), size, nullptr, nullptr);
    return result;
}

wstring getWindowTextString(HWND hwnd) {
    const int length = GetWindowTextLengthW(hwnd);
    wstring value(length, L'\0');
    GetWindowTextW(hwnd, value.data(), length + 1);
    return value;
}

void setWindowTextUtf8(HWND hwnd, const string& text) {
    const wstring wide = toWide(text);
    SetWindowTextW(hwnd, wide.c_str());
}

void setBitmapOnControl(HWND control, HBITMAP& target, HBITMAP bitmap) {
    HBITMAP oldBitmap = reinterpret_cast<HBITMAP>(SendMessageW(control, STM_SETIMAGE, IMAGE_BITMAP, reinterpret_cast<LPARAM>(bitmap)));
    if (oldBitmap != nullptr && oldBitmap != target) {
        DeleteObject(oldBitmap);
    }
    target = bitmap;
}

HBITMAP matToBitmap(const cv::Mat& input) {
    cv::Mat bgr;
    if (input.channels() == 1) {
        cv::cvtColor(input, bgr, cv::COLOR_GRAY2BGR);
    } else {
        bgr = input.clone();
    }

    BITMAPINFO bmi{};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = bgr.cols;
    bmi.bmiHeader.biHeight = -bgr.rows;
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 24;
    bmi.bmiHeader.biCompression = BI_RGB;

    void* bits = nullptr;
    HDC hdc = GetDC(nullptr);
    HBITMAP bitmap = CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, &bits, nullptr, 0);
    ReleaseDC(nullptr, hdc);
    if (!bitmap || !bits) {
        throw runtime_error("Failed to create bitmap");
    }

    const int stride = ((bgr.cols * 3 + 3) & ~3);
    for (int row = 0; row < bgr.rows; ++row) {
        memcpy(static_cast<unsigned char*>(bits) + row * stride, bgr.ptr(row), bgr.cols * 3);
    }
    return bitmap;
}

wstring openFileDialog(HWND owner) {
    wchar_t buffer[MAX_PATH] = {};
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = owner;
    ofn.lpstrFilter = L"Image Files\0*.jpg;*.jpeg;*.png;*.bmp;*.pgm\0All Files\0*.*\0";
    ofn.lpstrFile = buffer;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
    if (GetOpenFileNameW(&ofn)) {
        return buffer;
    }
    return L"";
}

wstring browseFolder(HWND owner, const wchar_t* title) {
    BROWSEINFOW bi{};
    bi.hwndOwner = owner;
    bi.lpszTitle = title;
    PIDLIST_ABSOLUTE pidl = SHBrowseForFolderW(&bi);
    if (!pidl) {
        return L"";
    }
    wchar_t path[MAX_PATH] = {};
    SHGetPathFromIDListW(pidl, path);
    CoTaskMemFree(pidl);
    return path;
}

void showError(HWND owner, const string& message) {
    MessageBoxW(owner, toWide(message).c_str(), L"FACE GUI Error", MB_ICONERROR | MB_OK);
}

void pumpUiMessages() {
    MSG message{};
    while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
}

void setProgress(GuiState& state, int percent, const wchar_t* status) {
    SendMessageW(state.progressBar, PBM_SETPOS, percent, 0);
    SetWindowTextW(state.statusText, status);
    UpdateWindow(state.progressBar);
    UpdateWindow(state.statusText);
    pumpUiMessages();
}

int selectedComboIndex(HWND combo) {
    const LRESULT index = SendMessageW(combo, CB_GETCURSEL, 0, 0);
    return index == CB_ERR ? 0 : static_cast<int>(index);
}

string feretSubsetName(int index) {
    switch (index) {
    case 1: return "fa";
    case 2: return "fc";
    case 3: return "qr";
    default: return "all";
    }
}

string buildGuiSummary(const IdentificationResult& result) {
    const MatchScore& top = result.ranking.front();
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(6);
    stream << "Top match: " << top.identity << " (" << top.imagePath.filename().string() << ")\r\n";
    stream << "Correlation: " << top.correlation << "\r\n";
    stream << "SP: " << result.query.sp << "\r\n";
    stream << "SI: " << result.query.si << "\r\n";
    stream << "SRR1: " << result.reliability.srr1 << "\r\n";
    stream << "SRR2: " << result.reliability.srr2 << "\r\n\r\n";
    stream << "Ranking\r\n";
    for (const MatchScore& match : result.ranking) {
        stream << match.identity << " | " << match.imagePath.filename().string() << " | " << match.correlation << "\r\n";
    }
    return stream.str();
}

string buildFeretSummary(const vector<BatchRunSummary>& summaries, const fs::path& rootDir) {
    ostringstream stream;
    stream << fixed << setprecision(6);
    stream << "FERET root: " << rootDir.string() << "\r\n\r\n";
    for (const BatchRunSummary& summary : summaries) {
        stream << "[" << summary.label << "]\r\n";
        stream << "gallery=" << summary.gallerySize << "\r\n";
        stream << "queries=" << summary.queryCount << "\r\n";
        stream << "mean_top_correlation=" << summary.meanTopCorrelation << "\r\n";
        stream << "mean_srr1=" << summary.meanSrr1 << "\r\n";
        stream << "mean_srr2=" << summary.meanSrr2 << "\r\n";
        stream << "csv=" << summary.csvPath.string() << "\r\n\r\n";
    }
    return stream.str();
}

void runFeretExperiment(GuiState& state) {
    const fs::path feretRoot = toUtf8(getWindowTextString(state.feretRootEdit));
    const bool illumination = SendMessageW(state.illuminationCheck, BM_GETCHECK, 0, 0) == BST_CHECKED;
    const int subsetIndex = selectedComboIndex(state.feretSubsetCombo);

    if (feretRoot.empty()) {
        throw runtime_error("FERET root is empty");
    }

    const vector<string> subsets = subsetIndex == 0
        ? vector<string>{"fa", "fc", "qr"}
        : vector<string>{feretSubsetName(subsetIndex)};

    EnableWindow(state.feretRunButton, FALSE);
    setProgress(state, 5, L"Preparing FERET experiment...");

    vector<BatchRunSummary> summaries;
    for (size_t i = 0; i < subsets.size(); ++i) {
        const string& subset = subsets[i];
        const fs::path subsetRoot = feretRoot / subset;
        const fs::path galleryDir = subsetRoot / "gallery";
        const fs::path probeDir = subsetRoot / "probe";
        const fs::path outputDir = subsetRoot / "output";

        const wstring status = wstring(L"Running subset ") + toWide(subset) + L"...";
        setProgress(state, 15 + static_cast<int>(i * 25), status.c_str());
        FacePipeline pipeline(defaultStasmDataDir(), outputDir, illumination, "_");
        FaceMatcher matcher;
        summaries.push_back(runBatchExperiment(pipeline, matcher, galleryDir, probeDir, outputDir, subset));
    }

    setWindowTextUtf8(state.feretSummaryText, buildFeretSummary(summaries, feretRoot));
    setProgress(state, 100, L"FERET done.");
    EnableWindow(state.feretRunButton, TRUE);
}

void runIdentify(GuiState& state) {
    const fs::path galleryDir = toUtf8(getWindowTextString(state.galleryEdit));
    const fs::path imagePath = toUtf8(getWindowTextString(state.imageEdit));
    const bool illumination = SendMessageW(state.illuminationCheck, BM_GETCHECK, 0, 0) == BST_CHECKED;

    EnableWindow(state.runButton, FALSE);
    setProgress(state, 5, L"Initializing pipeline...");
    FacePipeline pipeline(defaultStasmDataDir(), state.outputDir, illumination, "_");
    setProgress(state, 25, L"Loading gallery...");
    const vector<ProcessedFace> gallery = loadGallery(pipeline, galleryDir);
    setProgress(state, 55, L"Processing query image...");
    const ProcessedFace query = pipeline.processImage(imagePath);
    setProgress(state, 75, L"Matching against gallery...");
    const FaceMatcher matcher;
    const IdentificationResult result = matcher.identify(query, gallery);
    setProgress(state, 90, L"Rendering results...");

    const cv::Mat original = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
    if (original.empty()) {
        EnableWindow(state.runButton, TRUE);
        throw runtime_error("Cannot reload original image for display");
    }

    setWindowTextUtf8(state.resultText, buildGuiSummary(result));
    setBitmapOnControl(state.originalImage, state.originalBitmap, matToBitmap(original));
    setBitmapOnControl(state.poseImage, state.poseBitmap, matToBitmap(query.poseNormalized));
    setBitmapOnControl(state.sqiImage, state.sqiBitmap, matToBitmap(query.illuminationNormalized));
    setProgress(state, 100, L"Done.");
    EnableWindow(state.runButton, TRUE);
}

LRESULT CALLBACK windowProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam) {
    GuiState* state = reinterpret_cast<GuiState*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));

    switch (message) {
    case WM_NCCREATE:
        SetWindowLongPtrW(hwnd, GWLP_USERDATA,
                          reinterpret_cast<LONG_PTR>(reinterpret_cast<CREATESTRUCTW*>(lParam)->lpCreateParams));
        return TRUE;
    case WM_COMMAND:
        if (!state) {
            return 0;
        }
        switch (LOWORD(wParam)) {
        case kIdGalleryBrowse: {
            const wstring folder = browseFolder(hwnd, L"Select gallery folder");
            if (!folder.empty()) {
                SetWindowTextW(state->galleryEdit, folder.c_str());
            }
            return 0;
        }
        case kIdImageBrowse: {
            const wstring file = openFileDialog(hwnd);
            if (!file.empty()) {
                SetWindowTextW(state->imageEdit, file.c_str());
            }
            return 0;
        }
        case kIdRun:
            try {
                runIdentify(*state);
            } catch (const exception& ex) {
                EnableWindow(state->runButton, TRUE);
                setProgress(*state, 0, L"Processing failed.");
                showError(hwnd, ex.what());
            }
            return 0;
        case kIdOpenOutput:
            ShellExecuteW(hwnd, L"open", toWide(state->outputDir.string()).c_str(), nullptr, nullptr, SW_SHOWNORMAL);
            return 0;
        case kIdFeretRootBrowse: {
            const wstring folder = browseFolder(hwnd, L"Select FERET root folder");
            if (!folder.empty()) {
                SetWindowTextW(state->feretRootEdit, folder.c_str());
            }
            return 0;
        }
        case kIdFeretRun:
            try {
                runFeretExperiment(*state);
            } catch (const exception& ex) {
                EnableWindow(state->feretRunButton, TRUE);
                setProgress(*state, 0, L"FERET failed.");
                showError(hwnd, ex.what());
            }
            return 0;
        default:
            return 0;
        }
    case WM_DESTROY:
        if (state) {
            if (state->originalBitmap) {
                DeleteObject(state->originalBitmap);
            }
            if (state->poseBitmap) {
                DeleteObject(state->poseBitmap);
            }
            if (state->sqiBitmap) {
                DeleteObject(state->sqiBitmap);
            }
        }
        PostQuitMessage(0);
        return 0;
    default:
        return DefWindowProcW(hwnd, message, wParam, lParam);
    }
}

void createControls(HWND hwnd, GuiState& state) {
    state.mainWindow = hwnd;
    InitCommonControls();

    CreateWindowW(L"STATIC", L"Gallery:", WS_VISIBLE | WS_CHILD, 24, 22, 80, 24, hwnd, nullptr, nullptr, nullptr);
    state.galleryEdit = CreateWindowW(L"EDIT", L"data",
                                      WS_VISIBLE | WS_CHILD | WS_BORDER | ES_AUTOHSCROLL,
                                      110, 18, 520, 26, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(kIdGalleryEdit)), nullptr, nullptr);
    CreateWindowW(L"BUTTON", L"Browse",
                  WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
                  646, 18, 110, 26, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(kIdGalleryBrowse)), nullptr, nullptr);

    CreateWindowW(L"STATIC", L"Query Image:", WS_VISIBLE | WS_CHILD, 24, 60, 92, 24, hwnd, nullptr, nullptr, nullptr);
    state.imageEdit = CreateWindowW(L"EDIT", L"testface.jpg",
                                    WS_VISIBLE | WS_CHILD | WS_BORDER | ES_AUTOHSCROLL,
                                    110, 56, 520, 26, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(kIdImageEdit)), nullptr, nullptr);
    CreateWindowW(L"BUTTON", L"Browse",
                  WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
                  646, 56, 110, 26, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(kIdImageBrowse)), nullptr, nullptr);

    state.illuminationCheck = CreateWindowW(L"BUTTON", L"Apply illumination normalization",
                                            WS_VISIBLE | WS_CHILD | BS_AUTOCHECKBOX,
                                            24, 98, 260, 24, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(kIdIllumination)), nullptr, nullptr);
    SendMessageW(state.illuminationCheck, BM_SETCHECK, BST_CHECKED, 0);

    state.runButton = CreateWindowW(L"BUTTON", L"Run Identify",
                                    WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
                                    320, 92, 140, 32, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(kIdRun)), nullptr, nullptr);
    CreateWindowW(L"BUTTON", L"Open Output Folder",
                  WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
                  474, 92, 170, 32, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(kIdOpenOutput)), nullptr, nullptr);
    state.progressBar = CreateWindowW(PROGRESS_CLASSW, nullptr,
                                      WS_VISIBLE | WS_CHILD | PBS_SMOOTH,
                                      672, 94, 300, 22, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(kIdProgressBar)), nullptr, nullptr);
    SendMessageW(state.progressBar, PBM_SETRANGE, 0, MAKELPARAM(0, 100));
    SendMessageW(state.progressBar, PBM_SETPOS, 0, 0);
    state.statusText = CreateWindowW(L"STATIC", L"Idle",
                                     WS_VISIBLE | WS_CHILD,
                                     996, 94, 180, 22, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(kIdStatusText)), nullptr, nullptr);

    CreateWindowW(L"STATIC", L"Original", WS_VISIBLE | WS_CHILD, 24, 152, 120, 24, hwnd, nullptr, nullptr, nullptr);
    state.originalImage = CreateWindowW(L"STATIC", nullptr,
                                        WS_VISIBLE | WS_CHILD | SS_BITMAP | SS_CENTERIMAGE | WS_BORDER,
                                        24, 180, 300, 340, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(kIdOriginalImage)), nullptr, nullptr);

    CreateWindowW(L"STATIC", L"Pose Normalized", WS_VISIBLE | WS_CHILD, 348, 152, 140, 24, hwnd, nullptr, nullptr, nullptr);
    state.poseImage = CreateWindowW(L"STATIC", nullptr,
                                    WS_VISIBLE | WS_CHILD | SS_BITMAP | SS_CENTERIMAGE | WS_BORDER,
                                    348, 180, 300, 340, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(kIdPoseImage)), nullptr, nullptr);

    CreateWindowW(L"STATIC", L"SQI / Matching View", WS_VISIBLE | WS_CHILD, 672, 152, 160, 24, hwnd, nullptr, nullptr, nullptr);
    state.sqiImage = CreateWindowW(L"STATIC", nullptr,
                                   WS_VISIBLE | WS_CHILD | SS_BITMAP | SS_CENTERIMAGE | WS_BORDER,
                                   672, 180, 300, 340, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(kIdSqiImage)), nullptr, nullptr);

    CreateWindowW(L"STATIC", L"Results", WS_VISIBLE | WS_CHILD, 996, 152, 120, 24, hwnd, nullptr, nullptr, nullptr);
    state.resultText = CreateWindowW(L"EDIT", L"",
                                     WS_VISIBLE | WS_CHILD | WS_BORDER | ES_MULTILINE | ES_AUTOVSCROLL | ES_READONLY | WS_VSCROLL,
                                     996, 180, 320, 340, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(kIdResultText)), nullptr, nullptr);

    CreateWindowW(L"STATIC", L"FERET Root:", WS_VISIBLE | WS_CHILD, 24, 548, 100, 24, hwnd, nullptr, nullptr, nullptr);
    state.feretRootEdit = CreateWindowW(L"EDIT", L"experiments\\feret",
                                        WS_VISIBLE | WS_CHILD | WS_BORDER | ES_AUTOHSCROLL,
                                        110, 544, 520, 26, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(kIdFeretRootEdit)), nullptr, nullptr);
    CreateWindowW(L"BUTTON", L"Browse",
                  WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
                  646, 544, 110, 26, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(kIdFeretRootBrowse)), nullptr, nullptr);

    CreateWindowW(L"STATIC", L"Subset:", WS_VISIBLE | WS_CHILD, 780, 548, 60, 24, hwnd, nullptr, nullptr, nullptr);
    state.feretSubsetCombo = CreateWindowW(WC_COMBOBOXW, nullptr,
                                           WS_VISIBLE | WS_CHILD | CBS_DROPDOWNLIST | WS_VSCROLL,
                                           840, 542, 130, 300, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(kIdFeretSubsetCombo)), nullptr, nullptr);
    SendMessageW(state.feretSubsetCombo, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(L"All"));
    SendMessageW(state.feretSubsetCombo, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(L"fa"));
    SendMessageW(state.feretSubsetCombo, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(L"fc"));
    SendMessageW(state.feretSubsetCombo, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(L"qr"));
    SendMessageW(state.feretSubsetCombo, CB_SETCURSEL, 0, 0);

    state.feretRunButton = CreateWindowW(L"BUTTON", L"Run FERET",
                                         WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
                                         984, 542, 120, 32, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(kIdFeretRun)), nullptr, nullptr);

    CreateWindowW(L"STATIC", L"FERET Summary", WS_VISIBLE | WS_CHILD, 24, 586, 120, 24, hwnd, nullptr, nullptr, nullptr);
    state.feretSummaryText = CreateWindowW(L"EDIT", L"Run FERET to generate a summary here.",
                                           WS_VISIBLE | WS_CHILD | WS_BORDER | ES_MULTILINE | ES_AUTOVSCROLL | ES_READONLY | WS_VSCROLL,
                                           24, 612, 1294, 100, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(kIdFeretSummaryText)), nullptr, nullptr);
}
}

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int showCommand) {
    GuiState state;

    const wchar_t* className = L"FaceGuiWindowClass";
    WNDCLASSW wc{};
    wc.lpfnWndProc = windowProc;
    wc.hInstance = instance;
    wc.lpszClassName = className;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);

    RegisterClassW(&wc);

    HWND hwnd = CreateWindowExW(
        0,
        className,
        L"FACE GUI",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, 1370, 760,
        nullptr, nullptr, instance, &state);

    if (!hwnd) {
        return 1;
    }

    createControls(hwnd, state);
    ShowWindow(hwnd, showCommand);
    UpdateWindow(hwnd);

    MSG message{};
    while (GetMessageW(&message, nullptr, 0, 0)) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }

    return 0;
}
